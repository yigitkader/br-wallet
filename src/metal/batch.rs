//! Batch processor for GPU-accelerated brainwallet cracking
//!
//! This module provides high-level batch processing that integrates
//! the GPU with the existing comparer system.

use std::sync::Arc;

use super::gpu::{GpuBrainwallet, OUTPUT_SIZE};

/// Result from batch processing - matches CPU wallet format
#[derive(Clone, Debug)]
pub struct BrainwalletResult {
    /// Original passphrase
    pub passphrase: Vec<u8>,
    /// HASH160(compressed pubkey) - for P2PKH and Native SegWit
    pub h160_c: [u8; 20],
    /// HASH160(uncompressed pubkey) - for Legacy P2PKH
    pub h160_u: [u8; 20],
    /// HASH160(P2SH-P2WPKH script) - for Nested SegWit
    pub h160_nested: [u8; 20],
    /// Taproot x-only pubkey (32 bytes)
    pub taproot: [u8; 32],
    /// Uncompressed public key (64 bytes, X||Y without 0x04 prefix)
    /// Used for Ethereum Keccak256 address derivation on CPU
    pub pubkey_u: [u8; 64],
    /// Private key (32 bytes) - SHA256(passphrase), avoids recomputation
    pub priv_key: [u8; 32],
}

impl BrainwalletResult {
    /// Check if result is valid (non-zero hashes)
    pub fn is_valid(&self) -> bool {
        self.h160_c.iter().any(|&b| b != 0)
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
    
    /// Process a batch of passphrases
    /// 
    /// Returns results for all passphrases (including invalid ones with zero hashes)
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
                pubkey_u: gpu_result.pubkey_u,
                priv_key: gpu_result.priv_key,
            });
        }
        
        Ok(results)
    }
    
    /// Process a batch and check against target sets
    /// 
    /// This is the most efficient method - no intermediate allocations.
    /// Returns only matching passphrases with their results.
    pub fn process_and_match<F>(
        &self,
        passphrases: &[&[u8]],
        mut matcher: F,
    ) -> Result<Vec<BrainwalletResult>, String>
    where
        F: FnMut(&[u8; 20], &[u8; 20], &[u8; 20], &[u8; 32]) -> bool,
    {
        // Get raw output from GPU
        let raw_output = self.gpu.process_batch_raw(passphrases)?;
        
        let mut matches = Vec::new();
        
        for (i, passphrase) in passphrases.iter().enumerate() {
            let offset = i * OUTPUT_SIZE;
            if offset + OUTPUT_SIZE > raw_output.len() {
                break;
            }
            
            let data = &raw_output[offset..offset + OUTPUT_SIZE];
            
            // Parse inline
            let h160_c: [u8; 20] = data[0..20].try_into().unwrap();
            let h160_u: [u8; 20] = data[20..40].try_into().unwrap();
            let h160_nested: [u8; 20] = data[40..60].try_into().unwrap();
            let taproot: [u8; 32] = data[60..92].try_into().unwrap();
            let pubkey_u: [u8; 64] = data[92..156].try_into().unwrap();
            let priv_key: [u8; 32] = data[156..188].try_into().unwrap();
            
            // Skip invalid results
            if h160_c.iter().all(|&b| b == 0) {
                continue;
            }
            
            // Check if any hash matches
            if matcher(&h160_c, &h160_u, &h160_nested, &taproot) {
                matches.push(BrainwalletResult {
                    passphrase: passphrase.to_vec(),
                    h160_c,
                    h160_u,
                    h160_nested,
                    taproot,
                    pubkey_u,
                    priv_key,
                });
            }
        }
        
        Ok(matches)
    }
    
    /// Get total processed count
    pub fn total_processed(&self) -> u64 {
        self.gpu.total_processed()
    }
    
    /// Signal to stop processing
    pub fn stop(&self) {
        self.gpu.stop();
    }
    
    /// Check if should stop
    pub fn should_stop(&self) -> bool {
        self.gpu.should_stop()
    }
    
    /// Get reference to underlying GPU processor
    pub fn gpu(&self) -> &GpuBrainwallet {
        &self.gpu
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
            
            // Get line and clean it
            let mut line = &self.data[start..end];
            
            // Strip line endings (CRLF, CR)
            if line.ends_with(b"\r") {
                line = &line[..line.len() - 1];
            }
            
            // Trim leading whitespace
            while !line.is_empty() && (line[0] == b' ' || line[0] == b'\t') {
                line = &line[1..];
            }
            
            // Trim trailing whitespace
            while !line.is_empty() && (line[line.len() - 1] == b' ' || line[line.len() - 1] == b'\t') {
                line = &line[..line.len() - 1];
            }
            
            // Skip empty lines
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
    pub fn reset(&mut self) {
        self.position = 0;
    }
    
    /// Get current position (bytes processed)
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
}

