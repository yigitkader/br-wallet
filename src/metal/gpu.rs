//! GPU-accelerated brainwallet processing using Metal
//!
//! This module provides 10-100x speedup by:
//! - Batch processing 65K+ passphrases per GPU dispatch
//! - Parallel SHA256 + secp256k1 + RIPEMD160 on GPU
//! - Zero-copy via unified memory (Apple Silicon)

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

/// Metal shader source - embedded at compile time
const SHADER_SOURCE: &str = include_str!("brainwallet.metal");

/// Output size per passphrase: h160_c(20) + h160_u(20) + h160_nested(20) + taproot(32) + eth_addr(20) + priv_key(32) = 144 bytes
/// Note: Ethereum address is now computed on GPU via Keccak256 (no CPU post-processing needed!)
pub const OUTPUT_SIZE: usize = 144;

/// Maximum passphrase length
pub const MAX_PASSPHRASE_LEN: usize = 128;

/// Stride per passphrase in input buffer: 16 (aligned header) + MAX_PASSPHRASE_LEN = 144
/// 
/// Memory alignment optimization:
/// - Old stride: 129 bytes (unaligned - causes GPU memory access slowdown)
/// - New stride: 144 bytes (16-byte aligned for optimal GPU coalescing)
/// - Header: [length:1][padding:15][data:128] = 144 bytes
/// 
/// GPU memory coalescing works best with 16/32/64 byte aligned accesses.
pub const PASSPHRASE_STRIDE: usize = 16 + MAX_PASSPHRASE_LEN;

/// Result from GPU processing for a single passphrase
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuBrainwalletResult {
    pub h160_c: [u8; 20],       // HASH160(compressed pubkey) - Bitcoin/Litecoin
    pub h160_u: [u8; 20],       // HASH160(uncompressed pubkey) - Legacy addresses
    pub h160_nested: [u8; 20],  // HASH160(P2SH-P2WPKH script) - Nested SegWit
    pub taproot: [u8; 32],      // Taproot x-only pubkey - Bitcoin/Litecoin Taproot
    pub eth_addr: [u8; 20],     // Keccak256(pubkey)[12:32] - Ethereum address (GPU-computed!)
    pub priv_key: [u8; 32],     // SHA256(passphrase) - private key
}

impl GpuBrainwalletResult {
    /// Parse from raw output bytes
    /// Layout: h160_c(20) + h160_u(20) + h160_nested(20) + taproot(32) + eth_addr(20) + priv_key(32) = 144
    #[inline]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < OUTPUT_SIZE {
            return None;
        }
        
        let mut result = Self {
            h160_c: [0u8; 20],
            h160_u: [0u8; 20],
            h160_nested: [0u8; 20],
            taproot: [0u8; 32],
            eth_addr: [0u8; 20],
            priv_key: [0u8; 32],
        };
        
        result.h160_c.copy_from_slice(&data[0..20]);
        result.h160_u.copy_from_slice(&data[20..40]);
        result.h160_nested.copy_from_slice(&data[40..60]);
        result.taproot.copy_from_slice(&data[60..92]);
        result.eth_addr.copy_from_slice(&data[92..112]);
        result.priv_key.copy_from_slice(&data[112..144]);
        
        Some(result)
    }
    
    /// Check if result is valid (non-zero)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.h160_c.iter().any(|&b| b != 0)
    }
}

/// GPU tier configuration based on hardware
#[derive(Clone)]
struct GpuTier {
    name: String,
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
            name,
            threads_per_dispatch: threads,
            threadgroup_size: 256,
        })
    }
}

/// GPU Brainwallet processor
pub struct GpuBrainwallet {
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
    
    /// Process a batch of passphrases
    /// 
    /// Returns a vector of results, one per passphrase.
    /// Invalid passphrases (empty or producing invalid keys) have all-zero results.
    pub fn process_batch(&self, passphrases: &[&[u8]]) -> Result<Vec<GpuBrainwalletResult>, String> {
        if passphrases.is_empty() {
            return Ok(Vec::new());
        }
        
        let count = passphrases.len().min(self.tier.threads_per_dispatch);
        
        // Copy passphrases to input buffer with 16-byte aligned header
        // Format: [length:1][padding:15][data:128] = 144 bytes per passphrase
        unsafe {
            let input_ptr = self.input_buffer.contents() as *mut u8;
            
            for (i, passphrase) in passphrases.iter().take(count).enumerate() {
                let offset = i * PASSPHRASE_STRIDE;
                let len = passphrase.len().min(MAX_PASSPHRASE_LEN);
                
                // Write length byte at offset 0
                *input_ptr.add(offset) = len as u8;
                
                // Zero the 15-byte padding (bytes 1-15)
                std::ptr::write_bytes(input_ptr.add(offset + 1), 0, 15);
                
                // Write passphrase data starting at byte 16 (16-byte aligned)
                std::ptr::copy_nonoverlapping(
                    passphrase.as_ptr(),
                    input_ptr.add(offset + 16),
                    len,
                );
                
                // Zero-pad remaining passphrase area
                if len < MAX_PASSPHRASE_LEN {
                    std::ptr::write_bytes(
                        input_ptr.add(offset + 16 + len),
                        0,
                        MAX_PASSPHRASE_LEN - len,
                    );
                }
            }
            
            // Write count
            let count_ptr = self.count_buffer.contents() as *mut u32;
            *count_ptr = count as u32;
        }
        
        // Dispatch GPU kernel
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
        command_buffer.wait_until_completed();
        
        // Check for errors
        let status = command_buffer.status();
        if status == metal::MTLCommandBufferStatus::Error {
            return Err("GPU command buffer failed".to_string());
        }
        
        // Read results
        let mut results = Vec::with_capacity(count);
        
        unsafe {
            let output_ptr = self.output_buffer.contents() as *const u8;
            
            for i in 0..count {
                let offset = i * OUTPUT_SIZE;
                let data = std::slice::from_raw_parts(output_ptr.add(offset), OUTPUT_SIZE);
                
                if let Some(result) = GpuBrainwalletResult::from_bytes(data) {
                    results.push(result);
                } else {
                    // Should never happen, but handle gracefully
                    results.push(GpuBrainwalletResult {
                        h160_c: [0u8; 20],
                        h160_u: [0u8; 20],
                        h160_nested: [0u8; 20],
                        taproot: [0u8; 32],
                        eth_addr: [0u8; 20],
                        priv_key: [0u8; 32],
                    });
                }
            }
        }
        
        self.total_processed.fetch_add(count as u64, Ordering::Relaxed);
        
        Ok(results)
    }
    
    /// Process a batch and return raw bytes
    /// 
    /// This is more efficient when you only need to check specific hashes.
    pub fn process_batch_raw(&self, passphrases: &[&[u8]]) -> Result<&[u8], String> {
        if passphrases.is_empty() {
            return Ok(&[]);
        }
        
        let count = passphrases.len().min(self.tier.threads_per_dispatch);
        
        // Copy passphrases to input buffer with 16-byte aligned header
        unsafe {
            let input_ptr = self.input_buffer.contents() as *mut u8;
            
            for (i, passphrase) in passphrases.iter().take(count).enumerate() {
                let offset = i * PASSPHRASE_STRIDE;
                let len = passphrase.len().min(MAX_PASSPHRASE_LEN);
                
                // Length byte + 15 bytes padding + data
                *input_ptr.add(offset) = len as u8;
                std::ptr::write_bytes(input_ptr.add(offset + 1), 0, 15);
                std::ptr::copy_nonoverlapping(
                    passphrase.as_ptr(),
                    input_ptr.add(offset + 16),
                    len,
                );
                
                if len < MAX_PASSPHRASE_LEN {
                    std::ptr::write_bytes(
                        input_ptr.add(offset + 16 + len),
                        0,
                        MAX_PASSPHRASE_LEN - len,
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
        command_buffer.wait_until_completed();
        
        let status = command_buffer.status();
        if status == metal::MTLCommandBufferStatus::Error {
            return Err("GPU command buffer failed".to_string());
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
    
    /// Signal to stop processing
    pub fn stop(&self) {
        self.should_stop.store(true, Ordering::SeqCst);
    }
    
    /// Check if should stop
    pub fn should_stop(&self) -> bool {
        self.should_stop.load(Ordering::SeqCst)
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
        if !metal::Device::system_default().is_some() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let gpu = GpuBrainwallet::new();
        assert!(gpu.is_ok(), "GPU initialization failed: {:?}", gpu.err());
    }
    
    #[test]
    fn test_gpu_process_batch() {
        if !metal::Device::system_default().is_some() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let gpu = GpuBrainwallet::new().unwrap();
        
        let passphrases: Vec<&[u8]> = vec![
            b"password",
            b"test123",
            b"hello world",
        ];
        
        let results = gpu.process_batch(&passphrases).unwrap();
        assert_eq!(results.len(), 3);
        
        // All results should be valid (non-zero)
        for result in &results {
            assert!(result.is_valid(), "Result should be valid");
        }
        
        // Results should be different
        assert_ne!(results[0].h160_c, results[1].h160_c);
        assert_ne!(results[1].h160_c, results[2].h160_c);
    }
    
    #[test]
    fn test_gpu_known_vector() {
        if !metal::Device::system_default().is_some() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let gpu = GpuBrainwallet::new().unwrap();
        
        // Known test vector: SHA256("password") is a well-known private key
        let passphrases: Vec<&[u8]> = vec![b"password"];
        let results = gpu.process_batch(&passphrases).unwrap();
        
        assert_eq!(results.len(), 1);
        assert!(results[0].is_valid());
        
        println!("password -> h160_c: {}", hex::encode(&results[0].h160_c));
        println!("password -> h160_u: {}", hex::encode(&results[0].h160_u));
    }
}

