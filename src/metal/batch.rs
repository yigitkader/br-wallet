//! GPU batch processor for brainwallet cracking
//!
//! Zero-allocation matching via RawGpuResult (152 bytes per passphrase).
//! BrainwalletResult only allocated on match.

use std::sync::Arc;
use super::gpu::{GpuBrainwallet, OUTPUT_SIZE};

#[derive(Clone, Debug)]
pub struct BrainwalletResult {
    #[allow(dead_code)]
    pub passphrase: Vec<u8>,
    pub h160_c: [u8; 20],
    pub eth_addr: [u8; 20],
    pub priv_key: [u8; 32],
    pub glv_h160_c: [u8; 20],
    pub glv_eth_addr: [u8; 20],
}

impl BrainwalletResult {
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        self.h160_c.iter().any(|&b| b != 0)
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
            eth_addr: *self.eth_addr(),
            priv_key: *self.priv_key(),
            glv_h160_c: *self.glv_h160_c(),
            glv_eth_addr: *self.glv_eth_addr(),
        }
    }
}

pub struct RawBatchOutput<'a> {
    data: &'a [u8],
    count: usize,
}

impl<'a> RawBatchOutput<'a> {
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.count
    }
    
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
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
    
    #[inline]
    #[allow(dead_code)]
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
    
    #[allow(dead_code)]
    pub fn process(&self, passphrases: &[&[u8]]) -> Result<Vec<BrainwalletResult>, String> {
        let gpu_results = self.gpu.process_batch(passphrases)?;
        
        let mut results = Vec::with_capacity(gpu_results.len());
        for (passphrase, gpu_result) in passphrases.iter().zip(gpu_results.iter()) {
            results.push(BrainwalletResult {
                passphrase: passphrase.to_vec(),
                h160_c: gpu_result.h160_c,
                eth_addr: gpu_result.eth_addr,
                priv_key: gpu_result.priv_key,
                glv_h160_c: gpu_result.glv_h160_c,
                glv_eth_addr: gpu_result.glv_eth_addr,
            });
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
    
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.position = 0;
    }
    
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
    fn test_batch_processor() {
        if !metal::Device::system_default().is_some() {
            return;
        }
        
        let processor = BatchProcessor::new().unwrap();
        let passphrases: Vec<&[u8]> = vec![b"password", b"test"];
        let results = processor.process(&passphrases).unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].is_valid());
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
