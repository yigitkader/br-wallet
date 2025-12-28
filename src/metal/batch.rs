//! GPU batch processor for brainwallet cracking
//!
//! Zero-allocation matching via RawGpuResult (152 bytes per passphrase).
//! BrainwalletResult only allocated on match.

use std::sync::Arc;
use super::gpu::{GpuBrainwallet, OUTPUT_SIZE};

/// Result of brainwallet processing containing all derived addresses
#[derive(Clone, Debug)]
pub struct BrainwalletResult {
    /// Original passphrase used to derive this result
    pub passphrase: Vec<u8>,
    /// HASH160 of compressed public key (for P2PKH compressed, Native SegWit)
    pub h160_c: [u8; 20],
    /// HASH160 of uncompressed public key (for P2PKH uncompressed)
    pub h160_u: [u8; 20],
    /// HASH160 of P2SH-P2WPKH witness script (for 3xxx addresses)
    pub h160_nested: [u8; 20],
    /// Ethereum address (last 20 bytes of Keccak256)
    pub eth_addr: [u8; 20],
    /// Private key (SHA256 of passphrase)
    pub priv_key: [u8; 32],
    /// GLV-derived HASH160 of compressed public key
    pub glv_h160_c: [u8; 20],
    /// GLV-derived Ethereum address
    pub glv_eth_addr: [u8; 20],
}

impl BrainwalletResult {
    /// Check if result is valid (non-zero hash)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.h160_c.iter().any(|&b| b != 0)
    }
    
    /// Get passphrase as UTF-8 string (lossy conversion)
    #[inline]
    pub fn passphrase_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.passphrase)
    }
}

/// Zero-copy view into GPU output (152 bytes)
/// Layout: h160_c(20) + h160_u(20) + h160_nested(20) + eth_addr(20) + priv_key(32) + glv_h160_c(20) + glv_eth_addr(20)
#[derive(Clone, Copy)]
pub struct RawGpuResult<'a> {
    data: &'a [u8],
}

impl<'a> RawGpuResult<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() >= OUTPUT_SIZE {
            Some(Self { data: &data[..OUTPUT_SIZE] })
        } else {
            None
        }
    }
    
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.data[..20].iter().any(|&b| b != 0)
    }
    
    #[inline]
    pub fn h160_c(&self) -> &[u8; 20] {
        self.data[0..20].try_into().unwrap()
    }
    
    #[inline]
    pub fn h160_u(&self) -> &[u8; 20] {
        self.data[20..40].try_into().unwrap()
    }
    
    #[inline]
    pub fn h160_nested(&self) -> &[u8; 20] {
        self.data[40..60].try_into().unwrap()
    }
    
    #[inline]
    pub fn eth_addr(&self) -> &[u8; 20] {
        self.data[60..80].try_into().unwrap()
    }
    
    #[inline]
    pub fn priv_key(&self) -> &[u8; 32] {
        self.data[80..112].try_into().unwrap()
    }
    
    #[inline]
    pub fn glv_h160_c(&self) -> &[u8; 20] {
        self.data[112..132].try_into().unwrap()
    }
    
    #[inline]
    pub fn glv_eth_addr(&self) -> &[u8; 20] {
        self.data[132..152].try_into().unwrap()
    }
    
    #[inline]
    pub fn to_owned(&self, passphrase: &[u8]) -> BrainwalletResult {
        BrainwalletResult {
            passphrase: passphrase.to_vec(),
            h160_c: *self.h160_c(),
            h160_u: *self.h160_u(),
            h160_nested: *self.h160_nested(),
            eth_addr: *self.eth_addr(),
            priv_key: *self.priv_key(),
            glv_h160_c: *self.glv_h160_c(),
            glv_eth_addr: *self.glv_eth_addr(),
        }
    }
}

/// Zero-copy batch output from GPU processing
pub struct RawBatchOutput<'a> {
    data: &'a [u8],
    count: usize,
}

impl<'a> RawBatchOutput<'a> {
    /// Number of results in this batch
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if batch is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Get result at index (zero-copy)
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
    pub fn iter(&self) -> impl Iterator<Item = RawGpuResult<'a>> + '_ {
        (0..self.count).filter_map(move |i| self.get(i))
    }
}

pub struct BatchProcessor {
    gpu: Arc<GpuBrainwallet>,
}

impl BatchProcessor {
    pub fn new() -> Result<Self, String> {
        let gpu = GpuBrainwallet::new()?;
        Ok(Self { gpu: Arc::new(gpu) })
    }
    
    pub fn max_batch_size(&self) -> usize {
        self.gpu.max_batch_size()
    }
    
    pub fn process_raw<'a>(&'a self, passphrases: &[&[u8]]) -> Result<RawBatchOutput<'a>, String> {
        let raw_data = self.gpu.process_batch_raw(passphrases)?;
        let count = passphrases.len().min(self.gpu.max_batch_size());
        Ok(RawBatchOutput { data: raw_data, count })
    }
    
    /// Process batch and return full BrainwalletResult structs
    /// 
    /// Use `process_raw` for zero-allocation matching in hot paths.
    /// This method is useful when you need the full result with passphrase.
    pub fn process(&self, passphrases: &[&[u8]]) -> Result<Vec<BrainwalletResult>, String> {
        let raw_output = self.process_raw(passphrases)?;
        
        let mut results = Vec::with_capacity(raw_output.len());
        for (i, passphrase) in passphrases.iter().enumerate() {
            if let Some(raw) = raw_output.get(i) {
                results.push(raw.to_owned(passphrase));
            }
        }
        Ok(results)
    }
    
    pub fn total_processed(&self) -> u64 {
        self.gpu.total_processed()
    }
}

pub struct PassphraseBatcher<'a> {
    data: &'a [u8],
    position: usize,
    batch_size: usize,
}

impl<'a> PassphraseBatcher<'a> {
    pub fn new(data: &'a [u8], batch_size: usize) -> Self {
        Self { data, position: 0, batch_size }
    }
    
    pub fn next_batch(&mut self) -> Option<Vec<&'a [u8]>> {
        if self.position >= self.data.len() {
            return None;
        }
        
        let mut batch = Vec::with_capacity(self.batch_size);
        
        while batch.len() < self.batch_size && self.position < self.data.len() {
            let start = self.position;
            let mut end = start;
            
            while end < self.data.len() && self.data[end] != b'\n' {
                end += 1;
            }
            
            let mut line = &self.data[start..end];
            if line.ends_with(b"\r") {
                line = &line[..line.len() - 1];
            }
            
            if !line.is_empty() {
                batch.push(line);
            }
            
            self.position = end + 1;
        }
        
        if batch.is_empty() { None } else { Some(batch) }
    }
    
    /// Reset batcher to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }
    
    /// Get current position in bytes
    pub fn position(&self) -> usize {
        self.position
    }
    
    /// Get total data size in bytes
    pub fn total_size(&self) -> usize {
        self.data.len()
    }
    
    /// Calculate progress as percentage (0.0 - 100.0)
    pub fn progress_percent(&self) -> f64 {
        if self.data.is_empty() {
            return 100.0;
        }
        (self.position as f64 / self.data.len() as f64) * 100.0
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
    fn test_batcher_methods() {
        let data = b"line1\nline2\nline3\n";
        let mut batcher = PassphraseBatcher::new(data, 2);
        
        // Test total_size
        assert_eq!(batcher.total_size(), data.len());
        
        // Initial position
        assert_eq!(batcher.position(), 0);
        assert!(batcher.progress_percent() < 0.01);
        
        // After first batch
        let _ = batcher.next_batch();
        assert!(batcher.position() > 0);
        assert!(batcher.progress_percent() > 0.0);
        
        // Reset and verify
        batcher.reset();
        assert_eq!(batcher.position(), 0);
    }
    
    #[test]
    fn test_batch_processor() {
        if metal::Device::system_default().is_none() {
            return;
        }
        
        let processor = BatchProcessor::new().unwrap();
        let passphrases: Vec<&[u8]> = vec![b"password", b"test"];
        let results = processor.process(&passphrases).unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].is_valid());
        
        // Test passphrase_str
        assert_eq!(results[0].passphrase_str(), "password");
        assert_eq!(results[1].passphrase_str(), "test");
    }
    
    #[test]
    fn test_raw_batch_output() {
        if metal::Device::system_default().is_none() {
            return;
        }
        
        let processor = BatchProcessor::new().unwrap();
        let passphrases: Vec<&[u8]> = vec![b"test1", b"test2", b"test3"];
        let raw_output = processor.process_raw(&passphrases).unwrap();
        
        // Test len/is_empty
        assert_eq!(raw_output.len(), 3);
        assert!(!raw_output.is_empty());
        
        // Test iter
        let count = raw_output.iter().count();
        assert_eq!(count, 3);
        
        // Verify all results are valid
        for result in raw_output.iter() {
            assert!(result.is_valid());
        }
    }
    
    #[test]
    fn test_raw_gpu_result() {
        let data = [0u8; OUTPUT_SIZE];
        let raw = RawGpuResult::new(&data).unwrap();
        
        assert_eq!(raw.h160_c().len(), 20);
        assert_eq!(raw.h160_u().len(), 20);
        assert_eq!(raw.h160_nested().len(), 20);
        assert_eq!(raw.eth_addr().len(), 20);
        assert_eq!(raw.priv_key().len(), 32);
        assert_eq!(raw.glv_h160_c().len(), 20);
        assert_eq!(raw.glv_eth_addr().len(), 20);
        assert!(!raw.is_valid());
    }
}
