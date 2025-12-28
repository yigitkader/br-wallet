//! Known Brainwallet Test - Real-world test vectors
//!
//! Tests that GPU implementation correctly derives known brainwallet addresses.
//! Reference values calculated using standard crypto libraries.

use brwallet::metal::BatchProcessor;
use sha2::{Digest, Sha256};
use secp256k1::{PublicKey, SecretKey, SECP256K1};
use ripemd::Ripemd160;

/// Known brainwallet test vectors
/// Format: (passphrase, expected_h160_c_hex)
const KNOWN_BRAINWALLETS: &[(&str, &str)] = &[
    ("password", "400453ac5e19a058ec45a33550fdc496e0b26ad0"),
    ("satoshi", "0a8ba9e453383d4561cbcdda36e5789c2870dd41"),
    ("bitcoin", "5238c71458e464d9ff90299abca4a1d7b9cb76ab"),
    ("hello", "e3dd7e774a1272aeddb18efdc4baf6e14990edaa"),
    ("god", "3a116948027e696d6a12cb8520811b96d7f25fb3"),
];

fn hash160(data: &[u8]) -> [u8; 20] {
    Ripemd160::digest(Sha256::digest(data)).into()
}

fn derive_h160_c(passphrase: &str) -> [u8; 20] {
    let priv_bytes: [u8; 32] = Sha256::digest(passphrase.as_bytes()).into();
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    hash160(&pk.serialize())
}

/// Test known brainwallet hashes on GPU
#[test]
fn test_known_brainwallet_gpu() {
    println!("\n=== Known Brainwallet GPU Test ===\n");
    
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    let passphrases: Vec<&[u8]> = KNOWN_BRAINWALLETS
        .iter()
        .map(|(p, _)| p.as_bytes())
        .collect();
    
    let gpu_results = processor.process(&passphrases).expect("GPU process failed");
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (i, (passphrase, expected_h160_hex)) in KNOWN_BRAINWALLETS.iter().enumerate() {
        let gpu_h160 = hex::encode(&gpu_results[i].h160_c);
        let matches = gpu_h160 == *expected_h160_hex;
        
        println!("Passphrase: '{}'", passphrase);
        println!("  Expected h160:  {}", expected_h160_hex);
        println!("  GPU h160:       {}", gpu_h160);
        println!("  Status: {}", if matches { "✅ MATCH" } else { "❌ MISMATCH" });
        println!();
        
        if matches {
            passed += 1;
        } else {
            failed += 1;
        }
    }
    
    println!("=== Results: {}/{} passed ===", passed, KNOWN_BRAINWALLETS.len());
    
    assert_eq!(failed, 0, "All brainwallet hashes should match");
}

/// Test GPU results against reference implementation
#[test]
fn test_gpu_vs_reference_implementation() {
    println!("\n=== GPU vs Reference Implementation Test ===\n");
    
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    let passphrases: Vec<&[u8]> = KNOWN_BRAINWALLETS
        .iter()
        .map(|(p, _)| p.as_bytes())
        .collect();
    
    let gpu_results = processor.process(&passphrases).expect("GPU process failed");
    
    for (i, (passphrase, _)) in KNOWN_BRAINWALLETS.iter().enumerate() {
        // Reference implementation using secp256k1 + ripemd160
        let ref_h160 = derive_h160_c(passphrase);
        let gpu_result = &gpu_results[i];
        
        println!("Passphrase: '{}'", passphrase);
        println!("  Reference h160_c: {}", hex::encode(&ref_h160));
        println!("  GPU h160_c:       {}", hex::encode(&gpu_result.h160_c));
        
        assert_eq!(ref_h160, gpu_result.h160_c, 
            "GPU h160_c doesn't match reference for '{}'", passphrase);
        
        println!("  Status: ✅ MATCH\n");
    }
    
    println!("✅ All GPU results match reference implementation!");
}

/// WIF (Wallet Import Format) test
#[test]
fn test_known_brainwallet_wif() {
    println!("\n=== Known Brainwallet WIF Test ===\n");
    
    // "password" için bilinen değerler
    let passphrase = "password";
    let priv_bytes: [u8; 32] = Sha256::digest(passphrase.as_bytes()).into();
    
    // Expected private key (SHA256 of "password")
    let expected_privkey_hex = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8";
    let actual_privkey_hex = hex::encode(&priv_bytes);
    
    println!("Passphrase: '{}'", passphrase);
    println!("Expected private key: {}", expected_privkey_hex);
    println!("Actual private key:   {}", actual_privkey_hex);
    
    assert_eq!(actual_privkey_hex, expected_privkey_hex, "Private key mismatch");
    
    // WIF compressed format
    let mut wif_bytes = vec![0x80];  // Mainnet
    wif_bytes.extend_from_slice(&priv_bytes);
    wif_bytes.push(0x01);  // Compressed flag
    let wif = bs58::encode(&wif_bytes).with_check().into_string();
    
    println!("WIF (compressed): {}", wif);
    
    // WIF should start with 'K' or 'L' for compressed mainnet
    assert!(wif.starts_with('K') || wif.starts_with('L'), 
        "WIF should start with K or L for compressed mainnet");
    
    println!("\n✅ WIF test passed!");
}

/// Edge case: Empty passphrase
#[test]
fn test_empty_passphrase_gpu() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    // Empty string SHA256
    let passphrases: Vec<&[u8]> = vec![b""];
    let results = processor.process(&passphrases).expect("GPU process failed");
    
    // Verify result is valid (non-zero)
    assert!(results[0].h160_c.iter().any(|&b| b != 0), 
        "Empty passphrase should generate valid h160");
    
    println!("✅ Empty passphrase GPU test passed!");
}

/// Edge case: Unicode passphrase
#[test]
fn test_unicode_passphrase_gpu() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    let unicode_phrases: Vec<&[u8]> = vec![
        "密码".as_bytes(),           // Chinese: "password"
        "пароль".as_bytes(),         // Russian: "password"
        "パスワード".as_bytes(),       // Japanese: "password"
        "🔐🔑💰".as_bytes(),         // Emojis
    ];
    
    let results = processor.process(&unicode_phrases).expect("GPU process failed");
    
    for (i, phrase) in unicode_phrases.iter().enumerate() {
        let ref_h160 = {
            let priv_bytes: [u8; 32] = Sha256::digest(phrase).into();
            let sk = SecretKey::from_slice(&priv_bytes).unwrap();
            let pk = PublicKey::from_secret_key(SECP256K1, &sk);
            hash160(&pk.serialize())
        };
        
        assert_eq!(results[i].h160_c, ref_h160, 
            "Unicode passphrase GPU mismatch for {:?}", 
            String::from_utf8_lossy(phrase));
    }
    
    println!("✅ Unicode passphrase GPU test passed!");
}

/// Bitcoin Puzzle #1 reference
#[test]
fn test_bitcoin_puzzle_reference_gpu() {
    if metal::Device::system_default().is_none() {
        panic!("No Metal device - GPU is required!");
    }
    
    println!("\n=== Bitcoin Puzzle Reference GPU Test ===\n");
    
    // Note: Bitcoin puzzles use direct private keys, not brainwallet
    // This tests our ability to generate correct addresses from known keys
    
    // Puzzle #1: k=1, expected address 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
    let k1_priv: [u8; 32] = {
        let mut arr = [0u8; 32];
        arr[31] = 1;
        arr
    };
    
    let sk = SecretKey::from_slice(&k1_priv).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    
    let mut payload = vec![0x00];
    payload.extend_from_slice(&h160);
    let address = bs58::encode(&payload).with_check().into_string();
    
    println!("Puzzle #1 (k=1):");
    println!("  Generated: {}", address);
    println!("  Expected:  1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH");
    
    assert_eq!(address, "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",
        "Puzzle #1 address mismatch");
    
    println!("\n✅ Bitcoin puzzle reference test passed!");
}
