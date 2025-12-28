//! GPU Stress Test

use brwallet::metal::BatchProcessor;

#[test]
fn test_gpu_stress_1m_passphrases() {
    use std::time::Instant;
    
    if metal::Device::system_default().is_none() {
        panic!("No Metal device");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    let batch_size = processor.max_batch_size();
    
    println!("\n=== GPU Stress Test: 1M Passphrases ===");
    
    let total_passphrases = 1_000_000;
    let num_batches = (total_passphrases + batch_size - 1) / batch_size;
    
    let start = Instant::now();
    let mut total_valid = 0usize;
    
    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(total_passphrases);
        let current_batch_size = batch_end - batch_start;
        
        let mut test_data = Vec::with_capacity(current_batch_size);
        for i in batch_start..batch_end {
            test_data.push(format!("stress_test_{}", i));
        }
        let refs: Vec<&[u8]> = test_data.iter().map(|s| s.as_bytes()).collect();
        
        let results = processor.process(&refs).expect("GPU failed");
        
        let valid = results.iter()
            .filter(|r| r.h160_c.iter().any(|&b| b != 0))
            .count();
        total_valid += valid;
        
        if batch_idx % 10 == 0 {
            let progress = (batch_idx + 1) as f64 / num_batches as f64 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }
    }
    
    let elapsed = start.elapsed();
    let rate = total_passphrases as f64 / elapsed.as_secs_f64();
    
    println!("\nProcessed: {} | Valid: {} | Rate: {:.0}/sec", 
        total_passphrases, total_valid, rate);
    
    assert_eq!(total_valid, total_passphrases);
    assert!(elapsed.as_secs() < 120);
    
    println!("✅ Stress test passed!");
}

#[test]
fn test_gpu_memory_stability() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== GPU Memory Stability Test ===");
    
    let cycles = 20;
    let batch_size = 1000;
    
    for cycle in 0..cycles {
        let mut test_data = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            test_data.push(format!("mem_{}_{}", cycle, i));
        }
        let refs: Vec<&[u8]> = test_data.iter().map(|s| s.as_bytes()).collect();
        
        let results = processor.process(&refs).expect("GPU failed");
        
        let valid = results.iter()
            .filter(|r| r.h160_c.iter().any(|&b| b != 0))
            .count();
        
        assert_eq!(valid, batch_size, "Cycle {} failed", cycle);
        
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    
    println!("✅ Memory stability test passed!");
}

#[test]
fn test_gpu_consistency() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== GPU Consistency Test ===");
    
    let test_phrases: Vec<&[u8]> = vec![
        b"test1", b"test2", b"test3",
    ];
    
    let results1 = processor.process(&test_phrases).expect("GPU failed");
    
    for run in 0..10 {
        let results2 = processor.process(&test_phrases).expect("GPU failed");
        
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(r1.h160_c, r2.h160_c, "Run {}: h160_c mismatch {}", run, i);
            assert_eq!(r1.eth_addr, r2.eth_addr, "Run {}: eth_addr mismatch {}", run, i);
        }
    }
    
    println!("✅ Consistency test passed!");
}
