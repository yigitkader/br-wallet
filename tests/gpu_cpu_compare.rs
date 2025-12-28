//! GPU vs CPU hash comparison test

use brwallet::brainwallet::BtcWallet;
use sha2::{Digest, Sha256};

#[cfg(feature = "gpu")]
use brwallet::metal::BatchProcessor;

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_vs_cpu_hash_comparison() {
    use std::time::Instant;
    
    // Skip if no GPU
    if metal::Device::system_default().is_none() {
        println!("No Metal device - skipping");
        return;
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    // Test passphrases
    let test_phrases: Vec<&[u8]> = vec![
        b"password",
        b"test123",
        b"hello world",
        b"bitcoin",
        b"satoshi nakamoto",
        b"12345678",
        b"qwerty",
        b"letmein",
    ];
    
    println!("\n=== GPU vs CPU Hash Comparison ===\n");
    
    // GPU batch processing
    let gpu_start = Instant::now();
    let gpu_results = processor.process(&test_phrases).expect("GPU process failed");
    let gpu_time = gpu_start.elapsed();
    
    println!("GPU Time: {:?}", gpu_time);
    println!();
    
    // CPU processing  
    let cpu_start = Instant::now();
    for (i, passphrase) in test_phrases.iter().enumerate() {
        let priv_bytes: [u8; 32] = Sha256::digest(passphrase).into();
        
        if let Some(cpu_btc) = BtcWallet::generate(priv_bytes) {
            let gpu_result = &gpu_results[i];
            
            let pass_str = String::from_utf8_lossy(passphrase);
            
            // Compare ALL hashes including TAPROOT
            let h160_c_match = gpu_result.h160_c == cpu_btc.h160_c;
            let h160_u_match = gpu_result.h160_u == cpu_btc.h160_u;
            let h160_nested_match = gpu_result.h160_nested == cpu_btc.h160_nested;
            let taproot_match = gpu_result.taproot == cpu_btc.taproot;
            
            println!("Passphrase: '{}'", pass_str);
            println!("  CPU h160_c: {}", hex::encode(&cpu_btc.h160_c));
            println!("  GPU h160_c: {} {}", 
                hex::encode(&gpu_result.h160_c),
                if h160_c_match { "✓" } else { "✗ MISMATCH" }
            );
            println!("  CPU h160_u: {}", hex::encode(&cpu_btc.h160_u));
            println!("  GPU h160_u: {} {}", 
                hex::encode(&gpu_result.h160_u),
                if h160_u_match { "✓" } else { "✗ MISMATCH" }
            );
            println!("  CPU h160_nested: {}", hex::encode(&cpu_btc.h160_nested));
            println!("  GPU h160_nested: {} {}", 
                hex::encode(&gpu_result.h160_nested),
                if h160_nested_match { "✓" } else { "✗ MISMATCH" }
            );
            println!("  CPU taproot: {}", hex::encode(&cpu_btc.taproot));
            println!("  GPU taproot: {} {}", 
                hex::encode(&gpu_result.taproot),
                if taproot_match { "✓" } else { "✗ MISMATCH" }
            );
            println!();
            
            // Assert ALL matches including TAPROOT
            assert!(h160_c_match, "h160_c mismatch for '{}'", pass_str);
            assert!(h160_u_match, "h160_u mismatch for '{}'", pass_str);
            assert!(h160_nested_match, "h160_nested mismatch for '{}'", pass_str);
            assert!(taproot_match, "TAPROOT mismatch for '{}'", pass_str);
        }
    }
    let cpu_time = cpu_start.elapsed();
    println!("CPU Time: {:?}", cpu_time);
    
    println!("\n=== All hashes match! ===");
}

#[test]
#[cfg(feature = "gpu")]  
fn test_gpu_batch_performance() {
    use std::time::Instant;
    
    if metal::Device::system_default().is_none() {
        println!("No Metal device - skipping");
        return;
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    let batch_size = processor.max_batch_size();
    
    println!("\n=== GPU Batch Performance Test ===");
    println!("Batch size: {}", batch_size);
    
    // Generate test data
    let mut test_data = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        test_data.push(format!("password{}", i));
    }
    let refs: Vec<&[u8]> = test_data.iter().map(|s| s.as_bytes()).collect();
    
    // Benchmark GPU
    let start = Instant::now();
    let results = processor.process(&refs).expect("GPU process failed");
    let elapsed = start.elapsed();
    
    println!("Time for {} passphrases: {:?}", batch_size, elapsed);
    println!("Rate: {:.0} passphrases/sec", batch_size as f64 / elapsed.as_secs_f64());
    
    // Verify some results are non-zero
    let valid_count = results.iter().filter(|r| r.h160_c.iter().any(|&b| b != 0)).count();
    println!("Valid results: {}/{}", valid_count, batch_size);
    
    assert!(valid_count > 0, "Expected some valid results");
}

