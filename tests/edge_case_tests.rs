//! Edge Case Tests - Boundary conditions and special cases
//!
//! GPU-only tests for:
//! - Maximum/minimum valid private keys
//! - Batch size boundaries
//! - Whitespace handling

use brwallet::metal::BatchProcessor;
use sha2::{Digest, Sha256};
use secp256k1::{PublicKey, SecretKey, SECP256K1};
use ripemd::Ripemd160;

fn hash160(data: &[u8]) -> [u8; 20] {
    Ripemd160::digest(Sha256::digest(data)).into()
}

fn derive_h160_c_from_priv(priv_bytes: [u8; 32]) -> Option<[u8; 20]> {
    let sk = SecretKey::from_slice(&priv_bytes).ok()?;
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    Some(hash160(&pk.serialize()))
}

/// Maximum valid private key test
/// secp256k1 curve order - 1 (largest valid key)
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
    
    let h160 = derive_h160_c_from_priv(max_key);
    assert!(h160.is_some(), "Maximum valid key should produce valid wallet");
    
    let h160 = h160.unwrap();
    println!("h160_c: {}", hex::encode(&h160));
    
    // Non-zero output
    assert!(h160.iter().any(|&b| b != 0), "h160_c should not be zero");
    
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
    
    let h160 = derive_h160_c_from_priv(min_key);
    assert!(h160.is_some(), "k=1 should produce valid wallet");
    
    let h160 = h160.unwrap();
    println!("h160_c: {}", hex::encode(&h160));
    
    // k=1 produces G (generator point)
    println!("✅ Minimum valid key test passed!");
}

/// Invalid key: k = 0
#[test]
fn test_zero_private_key() {
    let zero_key: [u8; 32] = [0u8; 32];
    
    println!("\n=== Zero Private Key Test ===");
    
    let h160 = derive_h160_c_from_priv(zero_key);
    assert!(h160.is_none(), "k=0 should NOT produce valid wallet");
    
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
    
    let h160 = derive_h160_c_from_priv(curve_order);
    assert!(h160.is_none(), "k=N should NOT produce valid wallet");
    
    println!("✅ Curve order key correctly rejected!");
}

/// GPU Batch size edge cases
#[test]
fn test_gpu_batch_size_edge_cases() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
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

/// GPU edge case keys test
#[test]
fn test_gpu_edge_case_keys() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
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
        
        if let Some(ref_h160) = derive_h160_c_from_priv(priv_bytes) {
            let gpu_result = &gpu_results[i];
            
            assert_eq!(gpu_result.h160_c, ref_h160, 
                "h160_c mismatch for passphrase {:?}", passphrase);
            
            println!("  Passphrase {:?}: ✓", String::from_utf8_lossy(passphrase));
        }
    }
    
    println!("\n✅ All edge case keys match reference!");
}

/// Whitespace handling test
#[test]
fn test_whitespace_handling() {
    println!("\n=== Whitespace Handling Test ===");
    
    // These should all produce DIFFERENT hashes (whitespace is part of passphrase)
    let variations: &[&[u8]] = &[
        b"password",
        b"password ",      // trailing space
        b" password",      // leading space
        b" password ",     // both
        b"\tpassword",     // leading tab
        b"password\t",     // trailing tab
    ];
    
    // Get clean "password" hash
    let clean_priv: [u8; 32] = Sha256::digest(b"password").into();
    let clean_h160 = derive_h160_c_from_priv(clean_priv).unwrap();
    
    println!("Clean 'password' h160: {}", hex::encode(&clean_h160));
    
    // Note: These variations should be DIFFERENT because whitespace is part of passphrase
    for variation in variations {
        let priv_bytes: [u8; 32] = Sha256::digest(variation).into();
        let h160 = derive_h160_c_from_priv(priv_bytes).unwrap();
        
        let matches_clean = h160 == clean_h160;
        println!("  {:?}: {} ({})", 
            String::from_utf8_lossy(variation),
            hex::encode(&h160),
            if matches_clean { "SAME" } else { "DIFFERENT" }
        );
    }
    
    println!("\n✅ Whitespace handling test completed!");
}

/// Taproot specific edge cases - GPU
#[test]
fn test_taproot_edge_cases_gpu() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    println!("\n=== Taproot Edge Cases GPU Test ===");
    
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
    
    // All results should have non-zero taproot
    for (i, passphrase) in test_phrases.iter().enumerate() {
        let gpu_result = &gpu_results[i];
        
        assert!(gpu_result.taproot.iter().any(|&b| b != 0), 
            "Taproot should not be zero for '{}'", 
            String::from_utf8_lossy(passphrase));
        
        println!("  ✓ '{}'", String::from_utf8_lossy(passphrase));
    }
    
    println!("\n✅ All taproot edge cases passed!");
}
