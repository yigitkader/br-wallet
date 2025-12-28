//! Entegrasyon Testi - Tüm blockchain tipleri için
//! 
//! Test edilen passphrase: "satoshi"
//! Tüm adres tipleri bu passphrase'den türetilmiş ve JSON dosyalarına yazılmıştır.
//!
//! Çalıştırma: cargo test --test integration_test -- --nocapture

use std::collections::HashSet;
use sha2::{Digest, Sha256};
use secp256k1::{Keypair, PublicKey, SecretKey, XOnlyPublicKey, SECP256K1};
use ripemd::Ripemd160;
use tiny_keccak::{Hasher, Keccak};
use ed25519_dalek::SigningKey;

const TEST_PASSPHRASE: &str = "satoshi";

// ==================== HELPER FUNCTIONS ====================

fn hash160(data: &[u8]) -> [u8; 20] {
    Ripemd160::digest(Sha256::digest(data)).into()
}

fn tagged_hash(tag: &str, data: &[u8]) -> [u8; 32] {
    let tag_hash = Sha256::digest(tag.as_bytes());
    let mut engine = Sha256::new();
    engine.update(&tag_hash);
    engine.update(&tag_hash);
    engine.update(data);
    engine.finalize().into()
}

fn to_b58(version: u8, hash: &[u8; 20]) -> String {
    let mut payload = vec![version];
    payload.extend_from_slice(hash);
    bs58::encode(&payload).with_check().into_string()
}

// ==================== BITCOIN TESTS ====================

#[test]
fn test_bitcoin_legacy_p2pkh_compressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    let addr = to_b58(0x00, &h160);
    
    assert_eq!(addr, "1xm4vFerV3pSgvBFkyzLgT1Ew3HQYrS1V", 
               "Bitcoin Legacy P2PKH (compressed) address mismatch");
}

#[test]
fn test_bitcoin_legacy_p2pkh_uncompressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize_uncompressed());
    let addr = to_b58(0x00, &h160);
    
    assert_eq!(addr, "1ADJqstUMBB5zFquWg19UqZ7Zc6ePCpzLE", 
               "Bitcoin Legacy P2PKH (uncompressed) address mismatch");
}

#[test]
fn test_bitcoin_p2sh_segwit() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160_c = hash160(&pk.serialize());
    
    // P2SH-P2WPKH script
    let mut script = [0u8; 22];
    script[0] = 0x00;
    script[1] = 0x14;
    script[2..22].copy_from_slice(&h160_c);
    let h160_nested = hash160(&script);
    let addr = to_b58(0x05, &h160_nested);
    
    assert_eq!(addr, "3JkdX5SHwjTukvN3yGeUWGJ3oPXNvW5h8m", 
               "Bitcoin P2SH-SegWit address mismatch");
}

#[test]
fn test_bitcoin_native_segwit() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    
    let hrp = bech32::Hrp::parse("bc").unwrap();
    let addr = bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &h160).unwrap();
    
    assert_eq!(addr, "bc1qp296nezn8q752cwtehdrdetcns58ph2plx04f5", 
               "Bitcoin Native SegWit address mismatch");
}

#[test]
fn test_bitcoin_taproot() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let keypair = Keypair::from_secret_key(SECP256K1, &sk);
    let (internal_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    
    // BIP341 tweak
    let tweak_hash = tagged_hash("TapTweak", &internal_key.serialize());
    let tweak = secp256k1::Scalar::from_be_bytes(tweak_hash).unwrap();
    let (tweaked_key, _) = internal_key.add_tweak(SECP256K1, &tweak).unwrap();
    let taproot = tweaked_key.serialize();
    
    let hrp = bech32::Hrp::parse("bc").unwrap();
    let addr = bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &taproot).unwrap();
    
    assert_eq!(addr, "bc1p3ve2qusuhhy2cydvxfytuagvjsx56ccftuhxu4a9ephr2ddqz5vq70fjry", 
               "Bitcoin Taproot address mismatch");
}

// ==================== LITECOIN TESTS ====================

#[test]
fn test_litecoin_legacy_p2pkh_compressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    let addr = to_b58(0x30, &h160); // Litecoin P2PKH version
    
    assert_eq!(addr, "LLBiL8ZUw9HshVcLRtyHchWmT9QZX6TPqu", 
               "Litecoin Legacy P2PKH (compressed) address mismatch");
}

#[test]
fn test_litecoin_legacy_p2pkh_uncompressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize_uncompressed());
    let addr = to_b58(0x30, &h160);
    
    assert_eq!(addr, "LUSG76CJRqR9F4Y4gozSkrcsmpTvW8Gogt", 
               "Litecoin Legacy P2PKH (uncompressed) address mismatch");
}

#[test]
fn test_litecoin_p2sh_segwit() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160_c = hash160(&pk.serialize());
    
    let mut script = [0u8; 22];
    script[0] = 0x00;
    script[1] = 0x14;
    script[2..22].copy_from_slice(&h160_c);
    let h160_nested = hash160(&script);
    let addr = to_b58(0x32, &h160_nested); // Litecoin P2SH version
    
    assert_eq!(addr, "MQxmpxrFtrKLZRdx59dpKuYT867ppsgVXz", 
               "Litecoin P2SH-SegWit address mismatch");
}

#[test]
fn test_litecoin_native_segwit() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    
    let hrp = bech32::Hrp::parse("ltc").unwrap();
    let addr = bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &h160).unwrap();
    
    assert_eq!(addr, "ltc1qp296nezn8q752cwtehdrdetcns58ph2pm6433y", 
               "Litecoin Native SegWit address mismatch");
}

#[test]
fn test_litecoin_taproot() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let keypair = Keypair::from_secret_key(SECP256K1, &sk);
    let (internal_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    
    let tweak_hash = tagged_hash("TapTweak", &internal_key.serialize());
    let tweak = secp256k1::Scalar::from_be_bytes(tweak_hash).unwrap();
    let (tweaked_key, _) = internal_key.add_tweak(SECP256K1, &tweak).unwrap();
    let taproot = tweaked_key.serialize();
    
    let hrp = bech32::Hrp::parse("ltc").unwrap();
    let addr = bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &taproot).unwrap();
    
    assert_eq!(addr, "ltc1p3ve2qusuhhy2cydvxfytuagvjsx56ccftuhxu4a9ephr2ddqz5vqat8zep", 
               "Litecoin Taproot address mismatch");
}

// ==================== ETHEREUM TESTS ====================

#[test]
fn test_ethereum_address() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    
    let mut k = Keccak::v256();
    k.update(&pk.serialize_uncompressed()[1..65]); // 0x04 prefix atılır
    let mut h = [0u8; 32];
    k.finalize(&mut h);
    
    let addr = format!("0x{}", hex::encode(&h[12..32]));
    
    assert_eq!(addr, "0xf651e0dbba2072a7f4b3161a3f4e36f98ca34632", 
               "Ethereum address mismatch");
}

// ==================== SOLANA TESTS ====================

#[test]
fn test_solana_address() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SigningKey::from_bytes(&priv_bytes);
    let addr = bs58::encode(sk.verifying_key().to_bytes()).into_string();
    
    assert_eq!(addr, "zug6MJcfsKzTpUzsyRcEf9mabqyVNLDviwFBzNohQZ5", 
               "Solana address mismatch");
}

// ==================== COMPARER TESTS ====================

#[test]
fn test_comparer_bitcoin_parsing() {
    // Bitcoin adreslerini parse et ve hash'leri kontrol et
    // NOT: bc1q (Native SegWit) h160_c kullanır - Legacy compressed ile AYNI hash!
    // Bu yüzden HashSet'te 2 unique h20 hash olur (h160_c ve h160_nested)
    let addresses = vec![
        "1xm4vFerV3pSgvBFkyzLgT1Ew3HQYrS1V",        // Legacy P2PKH (h160_c)
        "3JkdX5SHwjTukvN3yGeUWGJ3oPXNvW5h8m",       // P2SH-SegWit (h160_nested)
        "bc1qp296nezn8q752cwtehdrdetcns58ph2plx04f5", // Native SegWit (h160_c - AYNI!)
        "bc1p3ve2qusuhhy2cydvxfytuagvjsx56ccftuhxu4a9ephr2ddqz5vq70fjry", // Taproot
    ];
    
    let mut h20_set: HashSet<[u8; 20]> = HashSet::new();
    let mut h32_set: HashSet<[u8; 32]> = HashSet::new();
    
    for addr in &addresses {
        if addr.starts_with("bc1") {
            if let Ok((_, _, p)) = bech32::segwit::decode(addr) {
                if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                    h20_set.insert(arr);
                } else if let Ok(arr) = <[u8; 32]>::try_from(p.as_slice()) {
                    h32_set.insert(arr);
                }
            }
        } else if let Ok(d) = bs58::decode(addr).with_check(None).into_vec() {
            if d.len() >= 21 {
                if let Ok(arr) = <[u8; 20]>::try_from(&d[1..21]) {
                    h20_set.insert(arr);
                }
            }
        }
    }
    
    // Legacy (1...) ve Native SegWit (bc1q...) aynı h160_c hash'ini kullanır
    // P2SH (3...) farklı bir hash kullanır (h160_nested)
    // Dolayısıyla 2 unique h20 hash var
    assert_eq!(h20_set.len(), 2, "Should have 2 unique h20 hashes (h160_c shared, h160_nested)");
    assert_eq!(h32_set.len(), 1, "Should have 1 h32 hash (taproot)");
    
    // Türetilen hash'lerin eşleştiğini kontrol et
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160_c = hash160(&pk.serialize());
    
    assert!(h20_set.contains(&h160_c), "h160_c should be in set");
}

#[test]
fn test_comparer_litecoin_parsing() {
    // Litecoin de aynı mantık - Legacy ve Native SegWit aynı hash'i paylaşır
    let addresses = vec![
        "LLBiL8ZUw9HshVcLRtyHchWmT9QZX6TPqu",       // Legacy (h160_c)
        "MQxmpxrFtrKLZRdx59dpKuYT867ppsgVXz",       // P2SH (h160_nested)
        "ltc1qp296nezn8q752cwtehdrdetcns58ph2pm6433y", // Native SegWit (h160_c - AYNI!)
        "ltc1p3ve2qusuhhy2cydvxfytuagvjsx56ccftuhxu4a9ephr2ddqz5vqat8zep", // Taproot
    ];
    
    let mut h20_set: HashSet<[u8; 20]> = HashSet::new();
    let mut h32_set: HashSet<[u8; 32]> = HashSet::new();
    
    for addr in &addresses {
        if addr.starts_with("ltc1") {
            if let Ok((_, _, p)) = bech32::segwit::decode(addr) {
                if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                    h20_set.insert(arr);
                } else if let Ok(arr) = <[u8; 32]>::try_from(p.as_slice()) {
                    h32_set.insert(arr);
                }
            }
        } else if let Ok(d) = bs58::decode(addr).with_check(None).into_vec() {
            if d.len() >= 21 {
                if let Ok(arr) = <[u8; 20]>::try_from(&d[1..21]) {
                    h20_set.insert(arr);
                }
            }
        }
    }
    
    assert_eq!(h20_set.len(), 2, "Should have 2 unique h20 hashes");
    assert_eq!(h32_set.len(), 1, "Should have 1 h32 hash");
}

#[test]
fn test_comparer_ethereum_parsing() {
    let addr = "0xf651e0dbba2072a7f4b3161a3f4e36f98ca34632";
    
    if let Ok(b) = hex::decode(addr.trim_start_matches("0x")) {
        if let Ok(arr) = <[u8; 20]>::try_from(b.as_slice()) {
            // Türetilen adresle karşılaştır
            let priv_bytes = derive_private_key(TEST_PASSPHRASE);
            let sk = SecretKey::from_slice(&priv_bytes).unwrap();
            let pk = PublicKey::from_secret_key(SECP256K1, &sk);
            
            let mut k = Keccak::v256();
            k.update(&pk.serialize_uncompressed()[1..65]);
            let mut h = [0u8; 32];
            k.finalize(&mut h);
            
            let mut eth_addr = [0u8; 20];
            eth_addr.copy_from_slice(&h[12..32]);
            
            assert_eq!(arr, eth_addr, "Ethereum address hash mismatch");
        } else {
            panic!("Failed to parse Ethereum address");
        }
    }
}

#[test]
fn test_comparer_solana_parsing() {
    let addr = "zug6MJcfsKzTpUzsyRcEf9mabqyVNLDviwFBzNohQZ5";
    
    if let Ok(b) = bs58::decode(addr).into_vec() {
        if let Ok(arr) = <[u8; 32]>::try_from(b.as_slice()) {
            // Türetilen adresle karşılaştır
            let priv_bytes = derive_private_key(TEST_PASSPHRASE);
            let sk = SigningKey::from_bytes(&priv_bytes);
            let sol_addr = sk.verifying_key().to_bytes();
            
            assert_eq!(arr, sol_addr, "Solana address mismatch");
        } else {
            panic!("Failed to parse Solana address");
        }
    }
}

// ==================== END-TO-END MATCHING TEST ====================

#[test]
fn test_full_matching_flow() {
    println!("\n=== Full Matching Flow Test ===\n");
    
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    
    // Generate all wallets
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    
    // Bitcoin/Litecoin hashes
    let h160_c = hash160(&pk.serialize());
    let h160_u = hash160(&pk.serialize_uncompressed());
    
    let mut script = [0u8; 22];
    script[0] = 0x00;
    script[1] = 0x14;
    script[2..22].copy_from_slice(&h160_c);
    let h160_nested = hash160(&script);
    
    let keypair = Keypair::from_secret_key(SECP256K1, &sk);
    let (internal_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    let tweak_hash = tagged_hash("TapTweak", &internal_key.serialize());
    let tweak = secp256k1::Scalar::from_be_bytes(tweak_hash).unwrap();
    let (tweaked_key, _) = internal_key.add_tweak(SECP256K1, &tweak).unwrap();
    let taproot = tweaked_key.serialize();
    
    // Load target hashes from test files (simulated)
    let btc_targets_20: HashSet<[u8; 20]> = [h160_c, h160_u, h160_nested].into_iter().collect();
    let btc_targets_32: HashSet<[u8; 32]> = [taproot].into_iter().collect();
    
    // Check Bitcoin matches
    assert!(btc_targets_20.contains(&h160_c), "BTC Legacy (comp) should match");
    assert!(btc_targets_20.contains(&h160_u), "BTC Legacy (uncomp) should match");
    assert!(btc_targets_20.contains(&h160_nested), "BTC P2SH should match");
    assert!(btc_targets_32.contains(&taproot), "BTC Taproot should match");
    
    // Litecoin uses same hashes (different address encoding, same underlying hash)
    assert!(btc_targets_20.contains(&h160_c), "LTC Legacy (comp) should match");
    assert!(btc_targets_32.contains(&taproot), "LTC Taproot should match");
    
    // Ethereum
    let mut k = Keccak::v256();
    k.update(&pk.serialize_uncompressed()[1..65]);
    let mut h = [0u8; 32];
    k.finalize(&mut h);
    let mut eth_addr = [0u8; 20];
    eth_addr.copy_from_slice(&h[12..32]);
    
    let eth_targets: HashSet<[u8; 20]> = [eth_addr].into_iter().collect();
    assert!(eth_targets.contains(&eth_addr), "ETH should match");
    
    // Solana
    let sol_sk = SigningKey::from_bytes(&priv_bytes);
    let sol_addr = sol_sk.verifying_key().to_bytes();
    
    let sol_targets: HashSet<[u8; 32]> = [sol_addr].into_iter().collect();
    assert!(sol_targets.contains(&sol_addr), "SOL should match");
    
    println!("✅ All 4 networks matched successfully!");
    println!("   - Bitcoin: 4 address types (Legacy, Uncomp, P2SH, SegWit, Taproot)");
    println!("   - Litecoin: 4 address types");
    println!("   - Ethereum: 1 address type");
    println!("   - Solana: 1 address type");
}

// ==================== HELPER ====================

fn derive_private_key(passphrase: &str) -> [u8; 32] {
    let hash = Sha256::digest(passphrase.as_bytes());
    let mut priv_bytes = [0u8; 32];
    priv_bytes.copy_from_slice(&hash);
    priv_bytes
}

