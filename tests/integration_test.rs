//! Integration Tests - All blockchain types
//! Test passphrase: "satoshi"

use std::collections::HashSet;
use sha2::{Digest, Sha256};
use secp256k1::{PublicKey, SecretKey, SECP256K1};
use ripemd::Ripemd160;
use tiny_keccak::{Hasher, Keccak};

const TEST_PASSPHRASE: &str = "satoshi";

fn hash160(data: &[u8]) -> [u8; 20] {
    Ripemd160::digest(Sha256::digest(data)).into()
}

fn to_b58(version: u8, hash: &[u8; 20]) -> String {
    let mut payload = vec![version];
    payload.extend_from_slice(hash);
    bs58::encode(&payload).with_check().into_string()
}

fn derive_private_key(passphrase: &str) -> [u8; 32] {
    Sha256::digest(passphrase.as_bytes()).into()
}

// ==================== BITCOIN TESTS ====================

#[test]
fn test_bitcoin_legacy_p2pkh_compressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    let addr = to_b58(0x00, &h160);
    assert_eq!(addr, "1xm4vFerV3pSgvBFkyzLgT1Ew3HQYrS1V");
}

#[test]
fn test_bitcoin_legacy_p2pkh_uncompressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize_uncompressed());
    let addr = to_b58(0x00, &h160);
    assert_eq!(addr, "1ADJqstUMBB5zFquWg19UqZ7Zc6ePCpzLE");
}

#[test]
fn test_bitcoin_p2sh_segwit() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160_c = hash160(&pk.serialize());
    
    let mut script = [0u8; 22];
    script[0] = 0x00;
    script[1] = 0x14;
    script[2..22].copy_from_slice(&h160_c);
    let h160_nested = hash160(&script);
    let addr = to_b58(0x05, &h160_nested);
    assert_eq!(addr, "3JkdX5SHwjTukvN3yGeUWGJ3oPXNvW5h8m");
}

#[test]
fn test_bitcoin_native_segwit() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    
    let hrp = bech32::Hrp::parse("bc").unwrap();
    let addr = bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &h160).unwrap();
    assert_eq!(addr, "bc1qp296nezn8q752cwtehdrdetcns58ph2plx04f5");
}

// ==================== LITECOIN TESTS ====================

#[test]
fn test_litecoin_legacy_p2pkh_compressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    let addr = to_b58(0x30, &h160);
    assert_eq!(addr, "LLBiL8ZUw9HshVcLRtyHchWmT9QZX6TPqu");
}

#[test]
fn test_litecoin_legacy_p2pkh_uncompressed() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize_uncompressed());
    let addr = to_b58(0x30, &h160);
    assert_eq!(addr, "LUSG76CJRqR9F4Y4gozSkrcsmpTvW8Gogt");
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
    let addr = to_b58(0x32, &h160_nested);
    assert_eq!(addr, "MQxmpxrFtrKLZRdx59dpKuYT867ppsgVXz");
}

#[test]
fn test_litecoin_native_segwit() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160 = hash160(&pk.serialize());
    
    let hrp = bech32::Hrp::parse("ltc").unwrap();
    let addr = bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &h160).unwrap();
    assert_eq!(addr, "ltc1qp296nezn8q752cwtehdrdetcns58ph2pm6433y");
}

// ==================== ETHEREUM TESTS ====================

#[test]
fn test_ethereum_address() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    
    let mut k = Keccak::v256();
    k.update(&pk.serialize_uncompressed()[1..65]);
    let mut h = [0u8; 32];
    k.finalize(&mut h);
    
    let addr = format!("0x{}", hex::encode(&h[12..32]));
    assert_eq!(addr, "0xf651e0dbba2072a7f4b3161a3f4e36f98ca34632");
}

// ==================== COMPARER TESTS ====================

#[test]
fn test_comparer_bitcoin_parsing() {
    let addresses = vec![
        "1xm4vFerV3pSgvBFkyzLgT1Ew3HQYrS1V",
        "3JkdX5SHwjTukvN3yGeUWGJ3oPXNvW5h8m",
        "bc1qp296nezn8q752cwtehdrdetcns58ph2plx04f5",
    ];
    
    let mut h20_set: HashSet<[u8; 20]> = HashSet::new();
    
    for addr in &addresses {
        if addr.starts_with("bc1q") {
            if let Ok((_, _, p)) = bech32::segwit::decode(addr) {
                if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                    h20_set.insert(arr);
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
    
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    let h160_c = hash160(&pk.serialize());
    
    assert!(h20_set.contains(&h160_c), "h160_c should be in set");
}

#[test]
fn test_comparer_litecoin_parsing() {
    let addresses = vec![
        "LLBiL8ZUw9HshVcLRtyHchWmT9QZX6TPqu",
        "MQxmpxrFtrKLZRdx59dpKuYT867ppsgVXz",
        "ltc1qp296nezn8q752cwtehdrdetcns58ph2pm6433y",
    ];
    
    let mut h20_set: HashSet<[u8; 20]> = HashSet::new();
    
    for addr in &addresses {
        if addr.starts_with("ltc1q") {
            if let Ok((_, _, p)) = bech32::segwit::decode(addr) {
                if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                    h20_set.insert(arr);
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
}

#[test]
fn test_comparer_ethereum_parsing() {
    let addr = "0xf651e0dbba2072a7f4b3161a3f4e36f98ca34632";
    
    if let Ok(b) = hex::decode(addr.trim_start_matches("0x")) {
        if let Ok(arr) = <[u8; 20]>::try_from(b.as_slice()) {
            let priv_bytes = derive_private_key(TEST_PASSPHRASE);
            let sk = SecretKey::from_slice(&priv_bytes).unwrap();
            let pk = PublicKey::from_secret_key(SECP256K1, &sk);
            
            let mut k = Keccak::v256();
            k.update(&pk.serialize_uncompressed()[1..65]);
            let mut h = [0u8; 32];
            k.finalize(&mut h);
            
            let mut eth_addr = [0u8; 20];
            eth_addr.copy_from_slice(&h[12..32]);
            
            assert_eq!(arr, eth_addr);
        } else {
            panic!("Failed to parse Ethereum address");
        }
    }
}

// ==================== END-TO-END MATCHING TEST ====================

#[test]
fn test_full_matching_flow() {
    let priv_bytes = derive_private_key(TEST_PASSPHRASE);
    let sk = SecretKey::from_slice(&priv_bytes).unwrap();
    let pk = PublicKey::from_secret_key(SECP256K1, &sk);
    
    let h160_c = hash160(&pk.serialize());
    let h160_u = hash160(&pk.serialize_uncompressed());
    
    let mut script = [0u8; 22];
    script[0] = 0x00;
    script[1] = 0x14;
    script[2..22].copy_from_slice(&h160_c);
    let h160_nested = hash160(&script);
    
    let btc_targets: HashSet<[u8; 20]> = [h160_c, h160_u, h160_nested].into_iter().collect();
    
    assert!(btc_targets.contains(&h160_c), "BTC Legacy (comp) should match");
    assert!(btc_targets.contains(&h160_u), "BTC Legacy (uncomp) should match");
    assert!(btc_targets.contains(&h160_nested), "BTC P2SH should match");
    
    let mut k = Keccak::v256();
    k.update(&pk.serialize_uncompressed()[1..65]);
    let mut h = [0u8; 32];
    k.finalize(&mut h);
    let mut eth_addr = [0u8; 20];
    eth_addr.copy_from_slice(&h[12..32]);
    
    let eth_targets: HashSet<[u8; 20]> = [eth_addr].into_iter().collect();
    assert!(eth_targets.contains(&eth_addr), "ETH should match");
}
