//! GPU Stress Test - Continuous operation test
//!
//! Tests GPU stability under heavy load.

use brwallet::metal::BatchProcessor;

/// Stress test: 1 million passphrases
#[test]
fn test_gpu_stress_1m_passphrases() {
    use std::time::Instant;
    
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    let batch_size = processor.max_batch_size();
    
    println!("\n=== GPU Stress Test: 1M Passphrases ===");
    println!("Batch size: {}", batch_size);
    
    let total_passphrases = 1_000_000;
    let num_batches = (total_passphrases + batch_size - 1) / batch_size;
    
    let start = Instant::now();
    let mut total_valid = 0usize;
    
    for batch_idx in 0..num_batches {
        // Generate batch data
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(total_passphrases);
        let current_batch_size = batch_end - batch_start;
        
        let mut test_data = Vec::with_capacity(current_batch_size);
        for i in batch_start..batch_end {
            test_data.push(format!("stress_test_password_{}", i));
        }
        let refs: Vec<&[u8]> = test_data.iter().map(|s| s.as_bytes()).collect();
        
        // Process on GPU
        let results = processor.process(&refs).expect("GPU process failed");
        
        // Count valid results
        let valid = results.iter()
            .filter(|r| r.h160_c.iter().any(|&b| b != 0))
            .count();
        total_valid += valid;
        
        // Progress every 10 batches
        if batch_idx % 10 == 0 {
            let progress = (batch_idx + 1) as f64 / num_batches as f64 * 100.0;
            println!("  Progress: {:.1}% ({}/{})", progress, batch_idx + 1, num_batches);
        }
    }
    
    let elapsed = start.elapsed();
    let rate = total_passphrases as f64 / elapsed.as_secs_f64();
    
    println!("\n=== Results ===");
    println!("Total processed: {}", total_passphrases);
    println!("Valid results: {}", total_valid);
    println!("Time: {:?}", elapsed);
    println!("Rate: {:.0} passphrases/sec", rate);
    
    // All passphrases should produce valid results
    assert_eq!(total_valid, total_passphrases, "All passphrases should produce valid hashes");
    
    // Should complete in reasonable time (< 120 seconds for 1M on M1)
    assert!(elapsed.as_secs() < 120, "Stress test took too long: {:?}", elapsed);
    
    println!("\n✅ Stress test passed!");
}

/// Memory stability test: Multiple batch cycles
#[test]
fn test_gpu_memory_stability() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== GPU Memory Stability Test ===");
    
    // 20 cycles with small batches
    let cycles = 20;
    let batch_size = 1000;
    
    for cycle in 0..cycles {
        let mut test_data = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            test_data.push(format!("memory_test_{}_{}", cycle, i));
        }
        let refs: Vec<&[u8]> = test_data.iter().map(|s| s.as_bytes()).collect();
        
        let results = processor.process(&refs).expect("GPU process failed");
        
        // Verify results are valid
        let valid = results.iter()
            .filter(|r| r.h160_c.iter().any(|&b| b != 0))
            .count();
        
        assert_eq!(valid, batch_size, "Cycle {} failed: {}/{} valid", cycle, valid, batch_size);
        
        if cycle % 5 == 0 {
            println!("  Cycle {}/{} - OK", cycle + 1, cycles);
        }
        
        // Give GPU a brief rest
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    
    println!("\n✅ Memory stability test passed! ({} cycles)", cycles);
}

/// Consistency test: Same input produces same output
#[test]
fn test_gpu_consistency() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== GPU Consistency Test ===");
    
    let test_phrases: Vec<&[u8]> = vec![
        b"consistency_test_1",
        b"consistency_test_2",
        b"consistency_test_3",
    ];
    
    // First run
    let results1 = processor.process(&test_phrases).expect("GPU process failed");
    
    // Repeat 10 times and compare
    for run in 0..10 {
        let results2 = processor.process(&test_phrases).expect("GPU process failed");
        
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(r1.h160_c, r2.h160_c, 
                "Run {}: h160_c mismatch for phrase {}", run, i);
            assert_eq!(r1.h160_u, r2.h160_u, 
                "Run {}: h160_u mismatch for phrase {}", run, i);
            assert_eq!(r1.h160_nested, r2.h160_nested, 
                "Run {}: h160_nested mismatch for phrase {}", run, i);
            assert_eq!(r1.taproot, r2.taproot, 
                "Run {}: taproot mismatch for phrase {}", run, i);
        }
    }
    
    println!("✅ Consistency test passed! (10 runs identical)");
}
