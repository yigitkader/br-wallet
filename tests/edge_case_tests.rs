//! Edge Case Tests - Boundary conditions and GPU verification

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

#[test]
fn test_maximum_valid_private_key() {
    let max_key: [u8; 32] = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
        0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x40,
    ];
    
    let h160 = derive_h160_c_from_priv(max_key);
    assert!(h160.is_some(), "Max valid key should produce valid wallet");
    assert!(h160.unwrap().iter().any(|&b| b != 0));
}

#[test]
fn test_minimum_valid_private_key() {
    let min_key: [u8; 32] = {
        let mut arr = [0u8; 32];
        arr[31] = 1;
        arr
    };
    
    let h160 = derive_h160_c_from_priv(min_key);
    assert!(h160.is_some(), "k=1 should produce valid wallet");
}

#[test]
fn test_zero_private_key() {
    let zero_key: [u8; 32] = [0u8; 32];
    let h160 = derive_h160_c_from_priv(zero_key);
    assert!(h160.is_none(), "k=0 should NOT produce valid wallet");
}

#[test]
fn test_key_at_curve_order() {
    let curve_order: [u8; 32] = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
        0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41,
    ];
    
    let h160 = derive_h160_c_from_priv(curve_order);
    assert!(h160.is_none(), "k=N should NOT produce valid wallet");
}

#[test]
fn test_gpu_batch_size_edge_cases() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    let empty: Vec<&[u8]> = vec![];
    let results = processor.process(&empty).expect("Empty batch should work");
    assert_eq!(results.len(), 0);
    
    let single: Vec<&[u8]> = vec![b"test"];
    let results = processor.process(&single).expect("Single should work");
    assert_eq!(results.len(), 1);
    assert!(results[0].h160_c.iter().any(|&b| b != 0));
    
    let batch_size = processor.max_batch_size();
    let mut exact_batch = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        exact_batch.push(format!("pass{}", i));
    }
    let refs: Vec<&[u8]> = exact_batch.iter().map(|s| s.as_bytes()).collect();
    let results = processor.process(&refs).expect("Exact batch should work");
    assert_eq!(results.len(), batch_size);
}

#[test]
fn test_gpu_edge_case_keys() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    let edge_cases: Vec<&[u8]> = vec![
        b"a",
        b"aa",
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        b"\x00",
        b"\xff",
        b" ",
        b"\t",
    ];
    
    let gpu_results = processor.process(&edge_cases).expect("GPU process failed");
    
    for (i, passphrase) in edge_cases.iter().enumerate() {
        let priv_bytes: [u8; 32] = Sha256::digest(passphrase).into();
        
        if let Some(ref_h160) = derive_h160_c_from_priv(priv_bytes) {
            assert_eq!(gpu_results[i].h160_c, ref_h160, 
                "h160_c mismatch for passphrase {:?}", passphrase);
        }
    }
}

#[test]
fn test_whitespace_handling() {
    let variations: &[&[u8]] = &[
        b"password",
        b"password ",
        b" password",
        b" password ",
        b"\tpassword",
        b"password\t",
    ];
    
    let clean_priv: [u8; 32] = Sha256::digest(b"password").into();
    let clean_h160 = derive_h160_c_from_priv(clean_priv).unwrap();
    
    for variation in variations {
        let priv_bytes: [u8; 32] = Sha256::digest(variation).into();
        let h160 = derive_h160_c_from_priv(priv_bytes).unwrap();
        
        if *variation == b"password" {
            assert_eq!(h160, clean_h160);
        } else {
            assert_ne!(h160, clean_h160, "Whitespace should produce different hash");
        }
    }
}

#[test]
fn test_glv_endomorphism() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    let test_passphrases = ["password", "satoshi", "bitcoin", "hello", "test123"];
    
    for passphrase in &test_passphrases {
        let gpu_results = processor.process(&[passphrase.as_bytes()]).expect("GPU failed");
        let result = &gpu_results[0];
        
        assert!(result.h160_c.iter().any(|&b| b != 0), "Primary should be non-zero");
        assert!(result.glv_h160_c.iter().any(|&b| b != 0), "GLV should be non-zero");
        assert_ne!(result.h160_c, result.glv_h160_c, "GLV should differ from primary");
        assert_ne!(result.eth_addr, result.glv_eth_addr, "GLV ETH should differ");
    }
    
    let gpu_results1 = processor.process(&[b"password".as_slice()]).expect("GPU failed");
    let gpu_results2 = processor.process(&[b"password".as_slice()]).expect("GPU failed");
    
    assert_eq!(gpu_results1[0].h160_c, gpu_results2[0].h160_c, "Should be deterministic");
    assert_eq!(gpu_results1[0].glv_h160_c, gpu_results2[0].glv_h160_c);
}
