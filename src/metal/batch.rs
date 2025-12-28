//! Batch processor for GPU-accelerated brainwallet cracking
//!
//! This module provides high-level batch processing that integrates
//! the GPU with the existing comparer system.
//!
//! ## Performance Architecture
//!
//! The key optimization is **zero-allocation matching**:
//! - GPU returns raw bytes (184 bytes per passphrase, including GLV bonus)
//! - We iterate over raw bytes without allocating BrainwalletResult for each
//! - Only when a match is found do we allocate and copy data
//! - This eliminates millions of heap allocations per second
//!
//! ## GLV Endomorphism (FREE 2x THROUGHPUT!)
//!
//! From ONE EC computation k*G, we get TWO valid keypairs:
//! - Primary: (k, P) where P = k*G
//! - GLV: (λ*k mod n, φ(P)) where φ(P) = (β*x, y)
//!
//! This essentially DOUBLES our address checking throughput for free!

use std::sync::Arc;

use super::gpu::{GpuBrainwallet, OUTPUT_SIZE};

/// Result from batch processing - only created on MATCH
/// 
/// ⚠️ PERFORMANCE: This struct is only created when a hash matches.
/// For the 99.99%+ non-matching passphrases, we use zero-copy RawGpuResult.
#[derive(Clone, Debug)]
pub struct BrainwalletResult {
    /// Original passphrase (used in format functions)
    #[allow(dead_code)]
    pub passphrase: Vec<u8>,
    /// HASH160(compressed pubkey) - for P2PKH and Native SegWit
    pub h160_c: [u8; 20],
    /// HASH160(uncompressed pubkey) - for Legacy P2PKH
    pub h160_u: [u8; 20],
    /// HASH160(P2SH-P2WPKH script) - for Nested SegWit
    pub h160_nested: [u8; 20],
    /// Taproot x-only pubkey (32 bytes)
    pub taproot: [u8; 32],
    /// Ethereum address (20 bytes) - Keccak256(pubkey)[12:32]
    pub eth_addr: [u8; 20],
    /// Private key (32 bytes) - SHA256(passphrase)
    pub priv_key: [u8; 32],
    
    // GLV Endomorphism bonus (FREE 2x throughput!)
    /// HASH160(GLV compressed pubkey) - Bitcoin/Litecoin
    pub glv_h160_c: [u8; 20],
    /// GLV Ethereum address - Keccak256(GLV pubkey)[12:32]
    pub glv_eth_addr: [u8; 20],
}

impl BrainwalletResult {
    /// Check if result is valid (non-zero hashes)
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        self.h160_c.iter().any(|&b| b != 0)
    }
}

/// Zero-copy view into GPU output for a single passphrase
/// 
/// This is used for efficient matching without heap allocation.
/// Layout: Primary(144) + GLV(40) = 184 bytes
///   Primary: h160_c(20) + h160_u(20) + h160_nested(20) + taproot(32) + eth_addr(20) + priv_key(32)
///   GLV: glv_h160_c(20) + glv_eth_addr(20)
#[derive(Clone, Copy)]
pub struct RawGpuResult<'a> {
    data: &'a [u8],
}

impl<'a> RawGpuResult<'a> {
    /// Create from raw GPU output slice (must be OUTPUT_SIZE bytes)
    #[inline]
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() >= OUTPUT_SIZE {
            Some(Self { data: &data[..OUTPUT_SIZE] })
        } else {
            None
        }
    }
    
    /// Check if valid (non-zero h160_c)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.data[..20].iter().any(|&b| b != 0)
    }
    
    /// Get h160_c (compressed pubkey hash) - zero-copy
    #[inline]
    pub fn h160_c(&self) -> &[u8; 20] {
        self.data[0..20].try_into().unwrap()
    }
    
    /// Get h160_u (uncompressed pubkey hash) - zero-copy
    #[inline]
    pub fn h160_u(&self) -> &[u8; 20] {
        self.data[20..40].try_into().unwrap()
    }
    
    /// Get h160_nested (P2SH-P2WPKH) - zero-copy
    #[inline]
    pub fn h160_nested(&self) -> &[u8; 20] {
        self.data[40..60].try_into().unwrap()
    }
    
    /// Get taproot pubkey - zero-copy
    #[inline]
    pub fn taproot(&self) -> &[u8; 32] {
        self.data[60..92].try_into().unwrap()
    }
    
    // =========================================================================
    // GLV Endomorphism accessors (FREE 2x throughput!)
    // =========================================================================
    
    /// Get GLV h160_c (compressed GLV pubkey hash) - zero-copy
    #[inline]
    pub fn glv_h160_c(&self) -> &[u8; 20] {
        self.data[144..164].try_into().unwrap()
    }
    
    /// Get GLV Ethereum address - zero-copy
    #[inline]
    pub fn glv_eth_addr(&self) -> &[u8; 20] {
        self.data[164..184].try_into().unwrap()
    }
    
    /// Get Ethereum address - zero-copy
    #[inline]
    pub fn eth_addr(&self) -> &[u8; 20] {
        self.data[92..112].try_into().unwrap()
    }
    
    /// Get private key - zero-copy
    #[inline]
    pub fn priv_key(&self) -> &[u8; 32] {
        self.data[112..144].try_into().unwrap()
    }
    
    /// Convert to owned BrainwalletResult (allocates!)
    /// Only call this when a match is confirmed.
    #[inline]
    pub fn to_owned(&self, passphrase: &[u8]) -> BrainwalletResult {
        BrainwalletResult {
            passphrase: passphrase.to_vec(),
            h160_c: *self.h160_c(),
            h160_u: *self.h160_u(),
            h160_nested: *self.h160_nested(),
            taproot: *self.taproot(),
            eth_addr: *self.eth_addr(),
            priv_key: *self.priv_key(),
            glv_h160_c: *self.glv_h160_c(),
            glv_eth_addr: *self.glv_eth_addr(),
        }
    }
}

/// Raw GPU output buffer - TRUE zero-copy access to batch results
/// 
/// This struct holds a REFERENCE to the GPU buffer (no heap allocation).
/// Data is valid until the next GPU dispatch.
pub struct RawBatchOutput<'a> {
    /// Direct reference to GPU output buffer (no copy!)
    data: &'a [u8],
    /// Number of valid results
    count: usize,
}

impl<'a> RawBatchOutput<'a> {
    /// Get number of results
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if empty
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Get raw result at index (zero-copy)
    #[inline]
    pub fn get(&self, index: usize) -> Option<RawGpuResult<'a>> {
        if index >= self.count {
            return None;
        }
        let offset = index * OUTPUT_SIZE;
        if offset + OUTPUT_SIZE > self.data.len() {
            return None;
        }
        RawGpuResult::new(&self.data[offset..offset + OUTPUT_SIZE])
    }
    
    /// Iterate over all results (zero-copy)
    #[inline]
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = RawGpuResult<'a>> + '_ {
        (0..self.count).filter_map(move |i| self.get(i))
    }
}

/// Batch processor that wraps GPU operations
pub struct BatchProcessor {
    gpu: Arc<GpuBrainwallet>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new() -> Result<Self, String> {
        let gpu = GpuBrainwallet::new()?;
        Ok(Self {
            gpu: Arc::new(gpu),
        })
    }
    
    /// Get maximum batch size supported by GPU
    pub fn max_batch_size(&self) -> usize {
        self.gpu.max_batch_size()
    }
    
    /// Process a batch and return raw output (TRUE zero-copy)
    /// 
    /// ⚠️ PERFORMANCE: This is the fastest method.
    /// Returns a direct reference to GPU buffer - NO heap allocation!
    /// Data is valid until the next call to process_raw.
    /// 
    /// Iterate over RawBatchOutput without allocating BrainwalletResult for each passphrase.
    /// Only call RawGpuResult::to_owned() when a match is confirmed.
    pub fn process_raw<'a>(&'a self, passphrases: &[&[u8]]) -> Result<RawBatchOutput<'a>, String> {
        // Use process_batch_raw for TRUE zero-copy (returns slice into GPU buffer)
        let raw_data = self.gpu.process_batch_raw(passphrases)?;
        let count = passphrases.len().min(self.gpu.max_batch_size());
        
        Ok(RawBatchOutput { data: raw_data, count })
    }
    
    /// Process a batch of passphrases (legacy API - allocates for each result)
    /// 
    /// ⚠️ PERFORMANCE WARNING: This method allocates BrainwalletResult for EVERY
    /// passphrase, including the 99.99%+ that don't match. Use process_raw() instead.
    #[allow(dead_code)]
    pub fn process(&self, passphrases: &[&[u8]]) -> Result<Vec<BrainwalletResult>, String> {
        let gpu_results = self.gpu.process_batch(passphrases)?;
        
        let mut results = Vec::with_capacity(gpu_results.len());
        
        for (passphrase, gpu_result) in passphrases.iter().zip(gpu_results.iter()) {
            results.push(BrainwalletResult {
                passphrase: passphrase.to_vec(),
                h160_c: gpu_result.h160_c,
                h160_u: gpu_result.h160_u,
                h160_nested: gpu_result.h160_nested,
                taproot: gpu_result.taproot,
                eth_addr: gpu_result.eth_addr,
                priv_key: gpu_result.priv_key,
                glv_h160_c: gpu_result.glv_h160_c,
                glv_eth_addr: gpu_result.glv_eth_addr,
            });
        }
        
        Ok(results)
    }
    
    /// Get total processed count
    pub fn total_processed(&self) -> u64 {
        self.gpu.total_processed()
    }
}

/// Helper to collect passphrases from mmap into batches
pub struct PassphraseBatcher<'a> {
    data: &'a [u8],
    position: usize,
    batch_size: usize,
}

impl<'a> PassphraseBatcher<'a> {
    /// Create a new batcher from memory-mapped file
    pub fn new(data: &'a [u8], batch_size: usize) -> Self {
        Self {
            data,
            position: 0,
            batch_size,
        }
    }
    
    /// Get next batch of passphrases
    /// 
    /// Returns slices into the original data (zero-copy)
    pub fn next_batch(&mut self) -> Option<Vec<&'a [u8]>> {
        if self.position >= self.data.len() {
            return None;
        }
        
        let mut batch = Vec::with_capacity(self.batch_size);
        
        while batch.len() < self.batch_size && self.position < self.data.len() {
            // Find end of line
            let start = self.position;
            let mut end = start;
            
            while end < self.data.len() && self.data[end] != b'\n' {
                end += 1;
            }
            
            // Get line and clean it - ONLY strip line endings, preserve whitespace!
            let mut line = &self.data[start..end];
            
            // Strip ONLY line endings (CRLF, CR) - preserve all other whitespace!
            // ⚠️ Brainwallet passphrases can have leading/trailing spaces!
            if line.ends_with(b"\r") {
                line = &line[..line.len() - 1];
            }
            
            // ⚠️ NO WHITESPACE TRIMMING - spaces/tabs are part of the passphrase!
            // Skip truly empty lines (only after CR stripping)
            if !line.is_empty() {
                batch.push(line);
            }
            
            // Move past newline
            self.position = end + 1;
        }
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
    
    /// Reset to beginning
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.position = 0;
    }
    
    /// Get current position (bytes processed)
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batcher() {
        let data = b"password\ntest123\nhello world\n";
        let mut batcher = PassphraseBatcher::new(data, 2);
        
        let batch1 = batcher.next_batch().unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0], b"password");
        assert_eq!(batch1[1], b"test123");
        
        let batch2 = batcher.next_batch().unwrap();
        assert_eq!(batch2.len(), 1);
        assert_eq!(batch2[0], b"hello world");
        
        assert!(batcher.next_batch().is_none());
    }
    
    #[test]
    fn test_batcher_crlf() {
        let data = b"password\r\ntest123\r\n";
        let mut batcher = PassphraseBatcher::new(data, 10);
        
        let batch = batcher.next_batch().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], b"password");
        assert_eq!(batch[1], b"test123");
    }
    
    #[test]
    fn test_batcher_preserves_whitespace() {
        // ⚠️ CRITICAL: Whitespace must be preserved for brainwallet accuracy!
        let data = b" password \n\ttest\t\n  hello world  \n";
        let mut batcher = PassphraseBatcher::new(data, 10);
        
        let batch = batcher.next_batch().unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0], b" password ");  // Leading/trailing spaces preserved
        assert_eq!(batch[1], b"\ttest\t");    // Tabs preserved
        assert_eq!(batch[2], b"  hello world  ");  // Multiple spaces preserved
    }
    
    #[test]
    fn test_batch_processor() {
        if !metal::Device::system_default().is_some() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let processor = BatchProcessor::new().unwrap();
        
        let passphrases: Vec<&[u8]> = vec![b"password", b"test"];
        let results = processor.process(&passphrases).unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].is_valid());
        assert!(results[1].is_valid());
    }
    
    #[test]
    fn test_batch_processor_raw() {
        if !metal::Device::system_default().is_some() {
            println!("Skipping test - no Metal device");
            return;
        }
        
        let processor = BatchProcessor::new().unwrap();
        
        let passphrases: Vec<&[u8]> = vec![b"password", b"test", b"hello"];
        let raw_output = processor.process_raw(&passphrases).unwrap();
        
        // Check length
        assert_eq!(raw_output.len(), 3);
        
        // Test zero-copy access
        let result0 = raw_output.get(0).unwrap();
        let result1 = raw_output.get(1).unwrap();
        let result2 = raw_output.get(2).unwrap();
        
        assert!(result0.is_valid());
        assert!(result1.is_valid());
        assert!(result2.is_valid());
        
        // Test zero-copy hash access
        assert!(result0.h160_c().iter().any(|&b| b != 0));
        assert!(result0.eth_addr().iter().any(|&b| b != 0));
        
        // Test to_owned only when needed
        let owned = result0.to_owned(b"password");
        assert_eq!(owned.passphrase, b"password");
        assert_eq!(owned.h160_c, *result0.h160_c());
    }
    
    #[test]
    fn test_raw_gpu_result_zero_copy() {
        // Test that RawGpuResult doesn't allocate
        // Layout: Primary(144) + GLV(40) = 184 bytes
        let data = [0u8; OUTPUT_SIZE];
        let raw = RawGpuResult::new(&data).unwrap();
        
        // These should all be zero-copy references
        // Primary
        assert_eq!(raw.h160_c().len(), 20);
        assert_eq!(raw.h160_u().len(), 20);
        assert_eq!(raw.h160_nested().len(), 20);
        assert_eq!(raw.taproot().len(), 32);
        assert_eq!(raw.eth_addr().len(), 20);
        assert_eq!(raw.priv_key().len(), 32);
        
        // GLV bonus (FREE 2x throughput!)
        assert_eq!(raw.glv_h160_c().len(), 20);
        assert_eq!(raw.glv_eth_addr().len(), 20);
        
        // Total should be 184
        assert!(!raw.is_valid()); // all zeros
    }
}

