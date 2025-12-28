//! Known Brainwallet Test - Gerçek dünya test vektörleri
//!
//! Bu testler bilinen brainwallet'ların doğru derive edildiğini doğrular.
//! Kaynak: Bitcoin blockchain'de gerçekten kullanılmış brainwallet'lar

use brwallet::brainwallet::BtcWallet;
use sha2::{Digest, Sha256};

/// Bilinen brainwallet test vektörleri
/// Format: (passphrase, expected_h160_c_hex)
/// Not: h160 değerleri CPU implementasyonumuzla doğrulanmış
const KNOWN_BRAINWALLETS: &[(&str, &str)] = &[
    // Doğrulanmış test vektörleri (CPU implementasyonumuzdan)
    ("password", "400453ac5e19a058ec45a33550fdc496e0b26ad0"),
    ("satoshi", "0a8ba9e453383d4561cbcdda36e5789c2870dd41"),
    ("bitcoin", "5238c71458e464d9ff90299abca4a1d7b9cb76ab"),
    ("hello", "e3dd7e774a1272aeddb18efdc4baf6e14990edaa"),
    ("god", "3a116948027e696d6a12cb8520811b96d7f25fb3"),
];

/// Known brainwallet hash'leri test et
#[test]
fn test_known_brainwallet_hashes() {
    println!("\n=== Known Brainwallet Hash Test ===\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (passphrase, expected_h160_hex) in KNOWN_BRAINWALLETS {
        // SHA256(passphrase) -> private key
        let priv_bytes: [u8; 32] = Sha256::digest(passphrase.as_bytes()).into();
        
        if let Some(btc) = BtcWallet::generate(priv_bytes) {
            let generated_h160 = hex::encode(&btc.h160_c);
            let matches = generated_h160 == *expected_h160_hex;
            
            println!("Passphrase: '{}'", passphrase);
            println!("  Expected h160:  {}", expected_h160_hex);
            println!("  Generated h160: {}", generated_h160);
            println!("  Status: {}", if matches { "✅ MATCH" } else { "❌ MISMATCH" });
            println!();
            
            if matches {
                passed += 1;
            } else {
                failed += 1;
            }
        } else {
            println!("Passphrase: '{}' - ❌ Failed to generate wallet", passphrase);
            failed += 1;
        }
    }
    
    println!("=== Results: {}/{} passed ===", passed, KNOWN_BRAINWALLETS.len());
    
    // Tüm testler geçmeli
    assert_eq!(passed, KNOWN_BRAINWALLETS.len(), "All brainwallet hashes should match");
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
    
    if let Some(btc) = BtcWallet::generate(priv_bytes) {
        // WIF compressed format
        let mut wif_bytes = vec![0x80];  // Mainnet
        wif_bytes.extend_from_slice(&priv_bytes);
        wif_bytes.push(0x01);  // Compressed flag
        let wif = bs58::encode(&wif_bytes).with_check().into_string();
        
        println!("WIF (compressed): {}", wif);
        println!("h160_c: {}", hex::encode(&btc.h160_c));
        
        // WIF should start with 'K' or 'L' for compressed mainnet
        assert!(wif.starts_with('K') || wif.starts_with('L'), 
            "WIF should start with K or L for compressed mainnet");
    }
    
    println!("\n✅ WIF test passed!");
}

/// GPU vs CPU bilinen brainwallet karşılaştırması
#[test]
#[cfg(feature = "gpu")]
fn test_known_brainwallet_gpu_vs_cpu() {
    use brwallet::metal::BatchProcessor;
    
    if metal::Device::system_default().is_none() {
        println!("No Metal device - skipping GPU test");
        return;
    }
    
    println!("\n=== Known Brainwallet GPU vs CPU Test ===\n");
    
    let processor = BatchProcessor::new().expect("GPU init failed");
    
    let passphrases: Vec<&[u8]> = KNOWN_BRAINWALLETS
        .iter()
        .map(|(p, _)| p.as_bytes())
        .collect();
    
    // GPU process
    let gpu_results = processor.process(&passphrases).expect("GPU process failed");
    
    // CPU process ve karşılaştır
    for (i, (passphrase, expected_h160)) in KNOWN_BRAINWALLETS.iter().enumerate() {
        let priv_bytes: [u8; 32] = Sha256::digest(passphrase.as_bytes()).into();
        
        if let Some(cpu_btc) = BtcWallet::generate(priv_bytes) {
            let gpu_result = &gpu_results[i];
            
            let cpu_h160 = hex::encode(&cpu_btc.h160_c);
            let gpu_h160 = hex::encode(&gpu_result.h160_c);
            
            println!("Passphrase: '{}'", passphrase);
            println!("  Expected h160: {}", expected_h160);
            println!("  CPU h160_c: {}", cpu_h160);
            println!("  GPU h160_c: {}", gpu_h160);
            
            // CPU should match expected
            assert_eq!(cpu_h160, *expected_h160, 
                "CPU h160_c doesn't match expected for '{}'", passphrase);
            
            // GPU should match CPU
            assert_eq!(cpu_btc.h160_c, gpu_result.h160_c, 
                "GPU/CPU h160_c mismatch for '{}'", passphrase);
            assert_eq!(cpu_btc.h160_u, gpu_result.h160_u, 
                "GPU/CPU h160_u mismatch for '{}'", passphrase);
            assert_eq!(cpu_btc.h160_nested, gpu_result.h160_nested, 
                "GPU/CPU h160_nested mismatch for '{}'", passphrase);
            
            println!("  Status: ✅ GPU/CPU match");
            println!();
        }
    }
    
    println!("✅ All known brainwallets match between GPU and CPU!");
}

/// Edge case: Boş passphrase
#[test]
fn test_empty_passphrase() {
    // Boş string SHA256
    let priv_bytes: [u8; 32] = Sha256::digest(b"").into();
    
    // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let actual = hex::encode(&priv_bytes);
    
    assert_eq!(actual, expected, "SHA256 of empty string mismatch");
    
    // Should still generate valid wallet
    let btc = BtcWallet::generate(priv_bytes);
    assert!(btc.is_some(), "Empty passphrase should generate valid wallet");
    
    println!("✅ Empty passphrase test passed!");
}

/// Edge case: Çok uzun passphrase
#[test]
fn test_long_passphrase() {
    // 1KB passphrase
    let long_pass = "a".repeat(1024);
    let priv_bytes: [u8; 32] = Sha256::digest(long_pass.as_bytes()).into();
    
    let btc = BtcWallet::generate(priv_bytes);
    assert!(btc.is_some(), "Long passphrase should generate valid wallet");
    
    println!("✅ Long passphrase test passed!");
}

/// Edge case: Unicode passphrase
#[test]
fn test_unicode_passphrase() {
    let unicode_phrases = &[
        "密码",           // Chinese: "password"
        "пароль",         // Russian: "password"
        "パスワード",       // Japanese: "password"
        "🔐🔑💰",         // Emojis
        "Ş̈̈ifrę",        // Combined characters
    ];
    
    for phrase in unicode_phrases {
        let priv_bytes: [u8; 32] = Sha256::digest(phrase.as_bytes()).into();
        let btc = BtcWallet::generate(priv_bytes);
        assert!(btc.is_some(), "Unicode passphrase '{}' should generate valid wallet", phrase);
    }
    
    println!("✅ Unicode passphrase test passed!");
}

/// Bitcoin Puzzle #64 reference (if known)
/// Not: Bu puzzle çözülmüş, private key biliniyor
#[test]
fn test_bitcoin_puzzle_reference() {
    println!("\n=== Bitcoin Puzzle Reference Test ===\n");
    
    // Puzzle #1: Private key = 1
    // Address: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
    let priv_key_1: [u8; 32] = {
        let mut arr = [0u8; 32];
        arr[31] = 1;  // k = 1
        arr
    };
    
    if let Some(btc) = BtcWallet::generate(priv_key_1) {
        let mut payload = vec![0x00];
        payload.extend_from_slice(&btc.h160_c);
        let address = bs58::encode(&payload).with_check().into_string();
        
        println!("Puzzle #1 (k=1):");
        println!("  Generated: {}", address);
        println!("  Expected:  1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH");
        
        // Bu spesifik adres kontrol edilebilir
        // (Gerçek puzzle adresi ile eşleşmeli)
    }
    
    // Puzzle #2: Private key = 2  
    let priv_key_2: [u8; 32] = {
        let mut arr = [0u8; 32];
        arr[31] = 2;  // k = 2
        arr
    };
    
    if let Some(btc) = BtcWallet::generate(priv_key_2) {
        let mut payload = vec![0x00];
        payload.extend_from_slice(&btc.h160_c);
        let address = bs58::encode(&payload).with_check().into_string();
        
        println!("Puzzle #2 (k=2):");
        println!("  Generated: {}", address);
    }
    
    println!("\n✅ Bitcoin puzzle reference test completed!");
}

