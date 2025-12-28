//! GPU-accelerated brainwallet processing using Metal

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

/// GPU command timeout (10 seconds should be plenty for any batch)
const GPU_TIMEOUT: Duration = Duration::from_secs(10);

/// Poll interval when waiting for GPU completion
const GPU_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Metal shader source - embedded at compile time
const SHADER_SOURCE: &str = include_str!("brainwallet.metal");

pub const OUTPUT_SIZE: usize = 152;

pub const MAX_PASSPHRASE_LEN: usize = 128;
pub const PASSPHRASE_STRIDE: usize = 16 + MAX_PASSPHRASE_LEN;

/// GPU tier configuration based on hardware
#[derive(Clone)]
struct GpuTier {
    threads_per_dispatch: usize,
    threadgroup_size: usize,
}

impl GpuTier {
    fn detect(device: &Device) -> Result<Self, String> {
        let name = device.name().to_string();
        let name_lower = name.to_lowercase();
        let gpu_mem_mb = device.recommended_max_working_set_size() / (1024 * 1024);
        
        println!("[GPU] Device: {}", name);
        println!("[GPU] Recommended working set: {} MB", gpu_mem_mb);
        
        // Detect Apple Silicon
        let is_apple_silicon = name_lower.contains("apple");
        
        // Determine tier based on GPU capabilities
        // Thresholds adjusted for accurate Apple Silicon detection
        let (tier_name, threads) = if name_lower.contains("ultra") || gpu_mem_mb >= 80000 {
            ("ULTRA", 262_144)
        } else if name_lower.contains("max") || gpu_mem_mb >= 40000 {
            ("MAX", 131_072)
        } else if name_lower.contains("pro") || (is_apple_silicon && gpu_mem_mb >= 18000) {
            ("PRO", 65_536)
        } else if is_apple_silicon && gpu_mem_mb >= 8000 {
            // M1/M2 base chips have ~10-11GB working set
            ("M1/M2", 49_152)
        } else {
            ("BASE", 32_768)
        };
        
        println!("[GPU] {} tier detected", tier_name);
        
        Ok(Self {
            threads_per_dispatch: threads,
            threadgroup_size: 256,
        })
    }
}

/// GPU Brainwallet processor - Apple Metal accelerated
/// 
/// Processes passphrases in batches, computing SHA256 → secp256k1 → HASH160/Keccak256
/// entirely on GPU for maximum throughput.
pub struct GpuBrainwallet {
    /// Metal device (kept alive for buffer/pipeline lifetime)
    #[allow(dead_code)]
    device: Device,
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    tier: GpuTier,
    
    // Buffers
    input_buffer: Buffer,
    count_buffer: Buffer,
    output_buffer: Buffer,
    
    // Stats
    total_processed: AtomicU64,
    should_stop: Arc<AtomicBool>,
}

// Metal types are thread-safe on Apple Silicon
unsafe impl Send for GpuBrainwallet {}
unsafe impl Sync for GpuBrainwallet {}

/// Thread-safe wrapper for raw pointer address (used in parallel copy)
/// Stores as usize to make it Send+Sync safe
#[derive(Clone, Copy)]
struct PtrAddr(usize);
unsafe impl Send for PtrAddr {}
unsafe impl Sync for PtrAddr {}

impl PtrAddr {
    #[inline]
    fn new(ptr: *mut u8) -> Self {
        Self(ptr as usize)
    }
    
    #[inline]
    fn get(&self) -> *mut u8 {
        self.0 as *mut u8
    }
}

impl GpuBrainwallet {
    /// Create a new GPU brainwallet processor
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;
        
        println!("🖥️  GPU: {}", device.name());
        println!("   Max threads per threadgroup: {}", device.max_threads_per_threadgroup().width);
        
        let tier = GpuTier::detect(&device)?;
        
        // Compile shader
        let library = device
            .new_library_with_source(SHADER_SOURCE, &metal::CompileOptions::new())
            .map_err(|e| format!("Failed to compile shader: {}", e))?;
        
        let function = library
            .get_function("process_brainwallet_batch", None)
            .map_err(|e| format!("Failed to get kernel function: {}", e))?;
        
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;
        
        let queue = device.new_command_queue();
        
        let storage = MTLResourceOptions::StorageModeShared;
        
        // Allocate buffers
        let input_size = tier.threads_per_dispatch * PASSPHRASE_STRIDE;
        let output_size = tier.threads_per_dispatch * OUTPUT_SIZE;
        
        let input_buffer = device.new_buffer(input_size as u64, storage);
        let count_buffer = device.new_buffer(4, storage);  // u32
        let output_buffer = device.new_buffer(output_size as u64, storage);
        
        println!("   Threads per dispatch: {}", tier.threads_per_dispatch);
        println!("   Input buffer: {} KB", input_size / 1024);
        println!("   Output buffer: {} KB", output_size / 1024);
        println!("✅ GPU brainwallet processor initialized");
        
        Ok(Self {
            device,
            pipeline,
            queue,
            tier,
            input_buffer,
            count_buffer,
            output_buffer,
            total_processed: AtomicU64::new(0),
            should_stop: Arc::new(AtomicBool::new(false)),
        })
    }
    
    /// Get maximum batch size
    pub fn max_batch_size(&self) -> usize {
        self.tier.threads_per_dispatch
    }
    
    /// Process a batch and return raw bytes (TRUE zero-copy)
    /// 
    /// Returns a direct slice into the GPU output buffer.
    /// This is the most efficient method - no heap allocation.
    /// Data is valid until the next call to any process method.
    /// 
    /// PERFORMANCE: Uses parallel copy via rayon for large batches (>1024)
    pub fn process_batch_raw(&self, passphrases: &[&[u8]]) -> Result<&[u8], String> {
        use rayon::prelude::*;
        
        if passphrases.is_empty() {
            return Ok(&[]);
        }
        
        let count = passphrases.len().min(self.tier.threads_per_dispatch);
        
        // Copy passphrases to input buffer with 16-byte aligned header
        // PARALLEL COPY for large batches to avoid CPU bottleneck
        unsafe {
            let input_ptr = self.input_buffer.contents() as *mut u8;
            
            // For batches > 1024, use parallel copy via rayon
            // This significantly speeds up buffer preparation for large batch sizes
            if count > 1024 {
                // Store pointer address as usize for thread safety
                let ptr_addr = PtrAddr::new(input_ptr);
                
                // Each thread writes to its own non-overlapping region (STRIDE * i)
                // This is safe because regions don't overlap
                passphrases.par_iter()
                    .take(count)
                    .enumerate()
                    .for_each(move |(i, passphrase)| {
                        let offset = i * PASSPHRASE_STRIDE;
                        let len = passphrase.len().min(MAX_PASSPHRASE_LEN);
                        
                        let ptr = ptr_addr.get();
                        
                        // Length byte at offset 0
                        *ptr.add(offset) = len as u8;
                        
                        // Zero padding bytes 1-15
                        std::ptr::write_bytes(ptr.add(offset + 1), 0, 15);
                        
                        // Copy passphrase data starting at byte 16
                        std::ptr::copy_nonoverlapping(
                            passphrase.as_ptr(),
                            ptr.add(offset + 16),
                            len,
                        );
                    });
            } else {
                // Sequential copy for small batches (overhead not worth it)
                for (i, passphrase) in passphrases.iter().take(count).enumerate() {
                    let offset = i * PASSPHRASE_STRIDE;
                    let len = passphrase.len().min(MAX_PASSPHRASE_LEN);
                    
                    *input_ptr.add(offset) = len as u8;
                    std::ptr::write_bytes(input_ptr.add(offset + 1), 0, 15);
                    std::ptr::copy_nonoverlapping(
                        passphrase.as_ptr(),
                        input_ptr.add(offset + 16),
                        len,
                    );
                }
            }
            
            let count_ptr = self.count_buffer.contents() as *mut u32;
            *count_ptr = count as u32;
        }
        
        // Dispatch
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.input_buffer), 0);
        encoder.set_buffer(1, Some(&self.count_buffer), 0);
        encoder.set_buffer(2, Some(&self.output_buffer), 0);
        
        let grid_size = MTLSize::new(count as u64, 1, 1);
        let threadgroup_size = MTLSize::new(self.tier.threadgroup_size as u64, 1, 1);
        
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        
        command_buffer.commit();
        
        // Wait with timeout to prevent infinite blocking
        let deadline = Instant::now() + GPU_TIMEOUT;
        loop {
            let status = command_buffer.status();
            match status {
                metal::MTLCommandBufferStatus::Completed => break,
                metal::MTLCommandBufferStatus::Error => {
                    return Err(format!(
                        "GPU command buffer failed (status: {:?}). \
                         Possible causes: shader error, buffer overflow, or invalid data.",
                        status
                    ));
                }
                _ => {
                    if Instant::now() > deadline {
                        return Err(format!(
                            "GPU timeout after {:?}. The GPU may be overloaded or the batch too large.",
                            GPU_TIMEOUT
                        ));
                    }
                    thread::sleep(GPU_POLL_INTERVAL);
                }
            }
        }
        
        self.total_processed.fetch_add(count as u64, Ordering::Relaxed);
        
        // Return raw output buffer
        unsafe {
            let output_ptr = self.output_buffer.contents() as *const u8;
            Ok(std::slice::from_raw_parts(output_ptr, count * OUTPUT_SIZE))
        }
    }
    
    /// Get total processed count
    pub fn total_processed(&self) -> u64 {
        self.total_processed.load(Ordering::Relaxed)
    }
    
    /// Signal to stop processing (for graceful shutdown)
    /// 
    /// Sets the stop flag. Use `should_stop()` to check this flag
    /// in your processing loop.
    pub fn stop(&self) {
        self.should_stop.store(true, Ordering::SeqCst);
    }
    
    /// Check if stop was signaled
    /// 
    /// Returns true if `stop()` was called. Use this to implement
    /// graceful shutdown in your processing loop.
    pub fn should_stop(&self) -> bool {
        self.should_stop.load(Ordering::SeqCst)
    }
    
    /// Reset the stop flag
    pub fn reset_stop(&self) {
        self.should_stop.store(false, Ordering::SeqCst);
    }
}

impl Drop for GpuBrainwallet {
    fn drop(&mut self) {
        // Metal automatically releases resources
        println!("🛑 GPU brainwallet processor shut down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_init() {
        if metal::Device::system_default().is_none() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let gpu = GpuBrainwallet::new();
        assert!(gpu.is_ok(), "GPU initialization failed: {:?}", gpu.err());
    }
    
    #[test]
    fn test_gpu_stop_control() {
        if metal::Device::system_default().is_none() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let gpu = GpuBrainwallet::new().unwrap();
        
        // Initial state: not stopped
        assert!(!gpu.should_stop());
        
        // Signal stop
        gpu.stop();
        assert!(gpu.should_stop());
        
        // Reset stop
        gpu.reset_stop();
        assert!(!gpu.should_stop());
    }
    
    #[test]
    fn test_gpu_process_batch_raw() {
        if metal::Device::system_default().is_none() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let gpu = GpuBrainwallet::new().unwrap();
        
        let passphrases: Vec<&[u8]> = vec![
            b"password",
            b"test123",
            b"hello world",
        ];
        
        let raw_output = gpu.process_batch_raw(&passphrases).unwrap();
        
        // Should have 3 results, each OUTPUT_SIZE bytes
        assert_eq!(raw_output.len(), 3 * OUTPUT_SIZE);
        
        // Check that results are different (h160_c is first 20 bytes)
        let h160_0 = &raw_output[0..20];
        let h160_1 = &raw_output[OUTPUT_SIZE..OUTPUT_SIZE + 20];
        let h160_2 = &raw_output[2 * OUTPUT_SIZE..2 * OUTPUT_SIZE + 20];
        
        // All results should be valid (non-zero)
        assert!(h160_0.iter().any(|&b| b != 0), "Result 0 should be valid");
        assert!(h160_1.iter().any(|&b| b != 0), "Result 1 should be valid");
        assert!(h160_2.iter().any(|&b| b != 0), "Result 2 should be valid");
        
        // Results should be different
        assert_ne!(h160_0, h160_1);
        assert_ne!(h160_1, h160_2);
    }
    
    #[test]
    fn test_gpu_known_vector() {
        if metal::Device::system_default().is_none() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let gpu = GpuBrainwallet::new().unwrap();
        
        // Known test vector: SHA256("password") is a well-known private key
        let passphrases: Vec<&[u8]> = vec![b"password"];
        let raw_output = gpu.process_batch_raw(&passphrases).unwrap();
        
        assert_eq!(raw_output.len(), OUTPUT_SIZE);
        
        // Result should be valid (non-zero h160_c)
        let h160_c = &raw_output[0..20];
        assert!(h160_c.iter().any(|&b| b != 0), "Result should be valid");
        
        println!("password -> h160_c: {}", hex::encode(h160_c));
        println!("password -> h160_u: {}", hex::encode(&raw_output[20..40]));
    }
}

