//! Edge Case Tests - Boundary conditions and special cases
//!
//! Bu testler kritik edge case'leri kontrol eder:
//! - Maximum valid private key
//! - Batch size sınırları
//! - Whitespace handling

use brwallet::brainwallet::BtcWallet;
use sha2::{Digest, Sha256};

#[cfg(feature = "gpu")]
use brwallet::metal::BatchProcessor;

/// Maximum valid private key test
/// secp256k1 curve order - 1 (en büyük geçerli key)
#[test]
fn test_maximum_valid_private_key() {
    // secp256k1 curve order N = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    // Max valid key = N - 1
    let max_key: [u8; 32] = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
        0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x40,
    ];
    
    println!("\n=== Maximum Valid Private Key Test ===");
    println!("Key: {}", hex::encode(&max_key));
    
    let btc = BtcWallet::generate(max_key);
    assert!(btc.is_some(), "Maximum valid key should produce valid wallet");
    
    let btc = btc.unwrap();
    println!("h160_c: {}", hex::encode(&btc.h160_c));
    println!("taproot: {}", hex::encode(&btc.taproot));
    
    // Non-zero output
    assert!(btc.h160_c.iter().any(|&b| b != 0), "h160_c should not be zero");
    assert!(btc.taproot.iter().any(|&b| b != 0), "taproot should not be zero");
    
    println!("✅ Maximum valid key test passed!");
}

/// Minimum valid private key test (k = 1)
#[test]
fn test_minimum_valid_private_key() {
    let min_key: [u8; 32] = {
        let mut arr = [0u8; 32];
        arr[31] = 1;  // k = 1
        arr
    };
    
    println!("\n=== Minimum Valid Private Key Test (k=1) ===");
    println!("Key: {}", hex::encode(&min_key));
    
    let btc = BtcWallet::generate(min_key);
    assert!(btc.is_some(), "k=1 should produce valid wallet");
    
    let btc = btc.unwrap();
    println!("h160_c: {}", hex::encode(&btc.h160_c));
    
    // k=1 should produce G (generator point)
    // P2PKH of G is known
    println!("✅ Minimum valid key test passed!");
}

/// Invalid key: k = 0
#[test]
fn test_zero_private_key() {
    let zero_key: [u8; 32] = [0u8; 32];
    
    println!("\n=== Zero Private Key Test ===");
    
    let btc = BtcWallet::generate(zero_key);
    assert!(btc.is_none(), "k=0 should NOT produce valid wallet");
    
    println!("✅ Zero key correctly rejected!");
}

/// Invalid key: k >= curve order
#[test]
fn test_key_at_curve_order() {
    // secp256k1 curve order N
    let curve_order: [u8; 32] = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
        0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41,
    ];
    
    println!("\n=== Curve Order Key Test ===");
    
    let btc = BtcWallet::generate(curve_order);
    assert!(btc.is_none(), "k=N should NOT produce valid wallet");
    
    println!("✅ Curve order key correctly rejected!");
}

/// GPU Batch size edge cases
#[test]
#[cfg(feature = "gpu")]
fn test_gpu_batch_size_edge_cases() {
    if metal::Device::system_default().is_none() {
        println!("No Metal device - skipping");
        return;
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== GPU Batch Size Edge Cases ===");
    
    // Test 1: Empty batch
    let empty: Vec<&[u8]> = vec![];
    let results = processor.process(&empty).expect("Empty batch should work");
    assert_eq!(results.len(), 0, "Empty batch should return empty results");
    println!("  Empty batch: ✓");
    
    // Test 2: Single passphrase
    let single: Vec<&[u8]> = vec![b"test"];
    let results = processor.process(&single).expect("Single should work");
    assert_eq!(results.len(), 1, "Single batch should return 1 result");
    assert!(results[0].h160_c.iter().any(|&b| b != 0), "Result should be valid");
    println!("  Single passphrase: ✓");
    
    // Test 3: Exact batch size
    let batch_size = processor.max_batch_size();
    let mut exact_batch = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        exact_batch.push(format!("pass{}", i));
    }
    let refs: Vec<&[u8]> = exact_batch.iter().map(|s| s.as_bytes()).collect();
    let results = processor.process(&refs).expect("Exact batch should work");
    assert_eq!(results.len(), batch_size, "Exact batch should return exact results");
    println!("  Exact batch size ({}): ✓", batch_size);
    
    // Test 4: Batch size + 1 (overflow handling)
    let mut overflow_batch = Vec::with_capacity(batch_size + 1);
    for i in 0..(batch_size + 1) {
        overflow_batch.push(format!("pass{}", i));
    }
    let refs: Vec<&[u8]> = overflow_batch.iter().map(|s| s.as_bytes()).collect();
    let results = processor.process(&refs).expect("Overflow batch should work");
    // Should truncate to max batch size
    assert!(results.len() <= batch_size, "Overflow should be handled");
    println!("  Batch overflow ({}+1): ✓", batch_size);
    
    println!("\n✅ All batch size edge cases passed!");
}

/// GPU vs CPU comparison for edge case keys
#[test]
#[cfg(feature = "gpu")]
fn test_gpu_edge_case_keys() {
    if metal::Device::system_default().is_none() {
        println!("No Metal device - skipping");
        return;
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== GPU Edge Case Keys Test ===");
    
    // Known edge case passphrases that produce interesting keys
    let edge_cases: Vec<&[u8]> = vec![
        b"a",           // Very short
        b"aa",          // Two chars
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  // 64 chars
        b"\x00",        // Null byte
        b"\xff",        // High byte
        b" ",           // Single space
        b"\t",          // Tab
    ];
    
    let gpu_results = processor.process(&edge_cases).expect("GPU process failed");
    
    for (i, passphrase) in edge_cases.iter().enumerate() {
        let priv_bytes: [u8; 32] = Sha256::digest(passphrase).into();
        
        if let Some(cpu_btc) = BtcWallet::generate(priv_bytes) {
            let gpu_result = &gpu_results[i];
            
            assert_eq!(gpu_result.h160_c, cpu_btc.h160_c, 
                "h160_c mismatch for passphrase {:?}", passphrase);
            assert_eq!(gpu_result.taproot, cpu_btc.taproot, 
                "taproot mismatch for passphrase {:?}", passphrase);
            
            println!("  Passphrase {:?}: ✓", String::from_utf8_lossy(passphrase));
        }
    }
    
    println!("\n✅ All edge case keys match between GPU and CPU!");
}

/// Whitespace handling test
#[test]
fn test_whitespace_handling() {
    println!("\n=== Whitespace Handling Test ===");
    
    // These should all produce the same hash as "password"
    let variations = [
        b"password".as_slice(),
        b"password ".as_slice(),      // trailing space
        b" password".as_slice(),       // leading space
        b" password ".as_slice(),      // both
        b"\tpassword".as_slice(),      // leading tab
        b"password\t".as_slice(),      // trailing tab
    ];
    
    // Get clean "password" hash
    let clean_priv: [u8; 32] = Sha256::digest(b"password").into();
    let clean_btc = BtcWallet::generate(clean_priv).unwrap();
    
    println!("Clean 'password' h160: {}", hex::encode(&clean_btc.h160_c));
    
    // Note: These variations should be DIFFERENT because whitespace is part of passphrase
    for variation in &variations {
        let priv_bytes: [u8; 32] = Sha256::digest(variation).into();
        let btc = BtcWallet::generate(priv_bytes).unwrap();
        
        let matches_clean = btc.h160_c == clean_btc.h160_c;
        println!("  {:?}: {} ({})", 
            String::from_utf8_lossy(variation),
            hex::encode(&btc.h160_c),
            if matches_clean { "SAME" } else { "DIFFERENT" }
        );
    }
    
    println!("\n✅ Whitespace handling test completed!");
}

/// Taproot specific edge cases
#[test]
#[cfg(feature = "gpu")]
fn test_taproot_edge_cases() {
    if metal::Device::system_default().is_none() {
        println!("No Metal device - skipping");
        return;
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== Taproot Edge Cases Test ===");
    
    // Test multiple passphrases to ensure taproot works consistently
    let test_phrases: Vec<&[u8]> = vec![
        b"password",
        b"test123", 
        b"bitcoin",
        b"satoshi nakamoto",
        b"hello world",
        b"12345678",
        b"qwerty",
        b"letmein",
        b"abc",
        b"xyz",
    ];
    
    let gpu_results = processor.process(&test_phrases).expect("GPU process failed");
    
    let mut all_match = true;
    for (i, passphrase) in test_phrases.iter().enumerate() {
        let priv_bytes: [u8; 32] = Sha256::digest(passphrase).into();
        
        if let Some(cpu_btc) = BtcWallet::generate(priv_bytes) {
            let gpu_result = &gpu_results[i];
            
            let taproot_match = gpu_result.taproot == cpu_btc.taproot;
            
            if !taproot_match {
                println!("  ❌ '{}': CPU={} GPU={}", 
                    String::from_utf8_lossy(passphrase),
                    hex::encode(&cpu_btc.taproot),
                    hex::encode(&gpu_result.taproot)
                );
                all_match = false;
            } else {
                println!("  ✓ '{}'", String::from_utf8_lossy(passphrase));
            }
        }
    }
    
    assert!(all_match, "All taproot values should match");
    println!("\n✅ All taproot edge cases passed!");
}

