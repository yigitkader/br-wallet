//! GPU-Accelerated Brainwallet Cracker - Metal Implementation
//!
//! This module provides high-performance brainwallet cracking using Apple Metal GPU.
//! ALL cryptographic operations (SHA256, secp256k1, RIPEMD160, Keccak256) run on GPU.
//!
//! Supported chains:
//! - Bitcoin: P2PKH, P2SH-P2WPKH, Native SegWit, Taproot
//! - Litecoin: Same address types with LTC prefixes
//! - Ethereum: Keccak256-based addresses
//!
//! ⚠️ GPU-ONLY: This implementation requires Apple Metal GPU.
//! CPU fallback has been removed for maximum performance.

use crate::comparer::Comparer;
use crate::metal::{PipelinedBatchProcessor, BrainwalletResult, PassphraseBatcher, RawBatchOutputOwned};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};
use std::time::{Duration, Instant};

/// Adaptive sampling ile satır sayısını tahmin eder
///
/// Dosyanın 3 farklı bölgesinden (başlangıç, orta, son) örnek alarak
/// daha doğru bir tahmin yapar. Değişken uzunluklu satırlar için önemli.
fn estimate_line_count(mmap: &Mmap) -> u64 {
    const SAMPLE_SIZE: usize = 1_048_576; // 1 MB per sample

    if mmap.len() == 0 {
        return 0;
    }

    // 3 farklı bölgeden sample al: başlangıç, orta, son
    let positions: [usize; 3] = [
        0,                                       // Başlangıç
        mmap.len() / 2,                          // Orta
        mmap.len().saturating_sub(SAMPLE_SIZE),  // Son
    ];

    let mut total_lines = 0u64;
    let mut total_bytes = 0usize;

    for &start in &positions {
        let end = (start + SAMPLE_SIZE).min(mmap.len());
        if end <= start {
            continue;
        }

        let sample = &mmap[start..end];
        let lines = sample.iter().filter(|&&b| b == b'\n').count();

        total_lines += lines as u64;
        total_bytes += sample.len();
    }

    if total_lines == 0 {
        return if mmap.len() > 0 { 1 } else { 0 };
    }

    (mmap.len() as u64 * total_lines) / total_bytes as u64
}

/// GPU-accelerated brainwallet cracking with PIPELINED processing
///
/// ALL operations happen on GPU:
/// - SHA256 (passphrase → private key)
/// - secp256k1 scalar multiplication (private key → public key)
/// - RIPEMD160 (public key → Bitcoin/Litecoin addresses)
/// - Keccak256 (public key → Ethereum address)
///
/// ## Pipeline Architecture
/// 
/// Uses double-buffering to overlap GPU and CPU work:
/// ```text
/// Time  │ GPU Buffer 0      │ GPU Buffer 1      │ CPU
/// ──────┼───────────────────┼───────────────────┼─────────────────
///   0   │ Process Batch 0   │                   │ Prepare Batch 1
///   1   │                   │ Process Batch 1   │ Check Batch 0, Prep 2
///   2   │ Process Batch 2   │                   │ Check Batch 1, Prep 3
/// ```
///
/// # Panics
/// Panics if GPU initialization fails. This is intentional - GPU is required.
pub fn start_cracking(dict: &str, comparer: &Comparer) {
    // Initialize pipelined GPU processor
    let processor = match PipelinedBatchProcessor::new() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ GPU initialization failed: {}", e);
            eprintln!("   This application requires Apple Metal GPU.");
            eprintln!("   Make sure you're running on macOS with Metal support.");
            std::process::exit(1);
        }
    };

    let batch_size = processor.max_batch_size();

    println!("🚀 GPU Mode: Metal Accelerated (PIPELINED)");
    println!("   Batch size: {} passphrases/dispatch", batch_size);
    println!("   Double-buffering: GPU and CPU work overlap");

    // Open dictionary file
    let file = match std::fs::File::open(dict) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ Cannot open dictionary: {}", e);
            std::process::exit(1);
        }
    };

    let mmap = match unsafe { Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Cannot memory-map dictionary: {}", e);
            std::process::exit(1);
        }
    };

    // Open log file
    let log = Mutex::new(BufWriter::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open("found.txt")
            .expect("Cannot open found.txt"),
    ));

    // Progress bar
    let estimated_lines = estimate_line_count(&mmap);
    println!("📊 Estimated lines: {}", estimated_lines);

    let pb = ProgressBar::new(estimated_lines);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (~{eta} remaining) {msg}")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message("GPU scanning (pipelined)...");

    let counter = AtomicU64::new(0);
    let mut batcher = PassphraseBatcher::new(&mmap, batch_size);
    
    // Throttled progress updates (every 500ms max)
    let mut last_pb_update = Instant::now();
    const PB_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

    // ========================================================================
    // PIPELINED PROCESSING LOOP
    // ========================================================================
    // 
    // Key insight: While GPU processes batch N, CPU can:
    // 1. Check results from batch N-1 (parallel with rayon)
    // 2. Prepare batch N+1 (read from mmap)
    //
    // This eliminates idle time on both GPU and CPU.
    
    // Track previous batch for result processing
    let mut prev_batch: Option<Vec<&[u8]>> = None;
    let mut pending_output: Option<RawBatchOutputOwned> = None;
    
    // Submit first batch to GPU (prime the pipeline)
    if let Some(batch) = batcher.next_batch() {
        if let Err(e) = processor.submit(&batch) {
            eprintln!("⚠️  GPU submit error: {}", e);
        } else {
            prev_batch = Some(batch);
        }
    }
    
    // Main pipelined loop
    loop {
        // Get next batch while GPU is processing
        let current_batch = batcher.next_batch();
        
        // Wait for GPU results from previous submission
        if prev_batch.is_some() {
            match processor.wait_results() {
                Ok(output) => {
                    pending_output = Some(output);
                }
                Err(e) => {
                    eprintln!("⚠️  GPU processing error: {}", e);
                    pending_output = None;
                }
            }
        }
        
        // Submit current batch to GPU (non-blocking)
        if let Some(ref batch) = current_batch {
            if let Err(e) = processor.submit(batch) {
                eprintln!("⚠️  GPU submit error: {}", e);
            }
        }
        
        // Process previous results on CPU while GPU works on current batch
        if let (Some(batch), Some(raw_output)) = (&prev_batch, &pending_output) {
            let batch_len = batch.len();
            
            // PARALLEL matching using rayon
            let all_matches: Vec<String> = batch
                .par_iter()
                .enumerate()
                .filter_map(|(i, passphrase)| {
                    let raw = raw_output.get(i)?;
                    
                    if !raw.is_valid() {
                        return None;
                    }
                    
                    let mut rep = String::new();
                    
                    // Lazily create result only when needed
                    let mut result: Option<BrainwalletResult> = None;
                    let ensure_result = || raw.to_owned(passphrase);

                    // Check Bitcoin - Primary keypair
                    if comparer.btc_on {
                        if comparer.btc_20.contains(raw.h160_c())
                            || comparer.btc_20.contains(raw.h160_u())
                            || comparer.btc_20.contains(raw.h160_nested())
                        {
                            let r = result.get_or_insert_with(&ensure_result);
                            rep.push_str(&format_btc_match(r, false));
                        }
                        // Check GLV INDEPENDENTLY
                        if comparer.btc_20.contains(raw.glv_h160_c()) {
                            let r = result.get_or_insert_with(&ensure_result);
                            rep.push_str(&format_btc_match(r, true));
                        }
                    }

                    // Check Litecoin - Primary keypair
                    if comparer.ltc_on {
                        if comparer.ltc_20.contains(raw.h160_c())
                            || comparer.ltc_20.contains(raw.h160_u())
                            || comparer.ltc_20.contains(raw.h160_nested())
                        {
                            let r = result.get_or_insert_with(&ensure_result);
                            rep.push_str(&format_ltc_match(r, false));
                        }
                        // Check GLV INDEPENDENTLY
                        if comparer.ltc_20.contains(raw.glv_h160_c()) {
                            let r = result.get_or_insert_with(&ensure_result);
                            rep.push_str(&format_ltc_match(r, true));
                        }
                    }

                    // Check Ethereum - Primary keypair
                    if comparer.eth_on {
                        if comparer.eth_20.contains(raw.eth_addr()) {
                            let r = result.get_or_insert_with(&ensure_result);
                            rep.push_str(&format_eth_match(r, false));
                        }
                        // Check GLV INDEPENDENTLY
                        if comparer.eth_20.contains(raw.glv_eth_addr()) {
                            let r = result.get_or_insert_with(&ensure_result);
                            rep.push_str(&format_eth_match(r, true));
                        }
                    }

                    if rep.is_empty() { None } else { Some(rep) }
                })
                .collect();

            // Write matches - IMMEDIATE FLUSH after each match to prevent data loss
            // If program crashes, we don't want to lose found matches
            if !all_matches.is_empty() {
                let mut f = log.lock().unwrap();
                for rep in &all_matches {
                    let _ = f.write_all(rep.as_bytes());
                    // Flush immediately after each match - critical for data safety
                    let _ = f.flush();
                    pb.println(format!("\n{}", rep));
                }
            }

            // Update progress
            let current = counter.fetch_add(batch_len as u64, Ordering::Relaxed);
            let now = Instant::now();
            if now.duration_since(last_pb_update) >= PB_UPDATE_INTERVAL {
                pb.set_position(current + batch_len as u64);
                last_pb_update = now;
            }
        }
        
        // Move current to previous for next iteration
        prev_batch = current_batch;
        pending_output = None;
        
        // Exit if no more batches
        if prev_batch.is_none() {
            break;
        }
    }

    // Finalize
    let final_count = counter.load(Ordering::Relaxed);
    pb.set_position(final_count);

    if let Ok(mut f) = log.lock() {
        let _ = f.flush();
    }

    let processed = processor.total_processed();
    pb.finish_with_message(format!(
        "Complete! {} lines scanned (GPU processed: {})",
        final_count, processed
    ));
}

// ============================================================================
// FORMATTING FUNCTIONS - All use GPU-computed values (NO recomputation!)
// ============================================================================

/// GLV endomorphism lambda constant
/// λ = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
const GLV_LAMBDA: [u8; 32] = [
    0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0,
    0xa5, 0x26, 0x1c, 0x02, 0x88, 0x12, 0x64, 0x5a,
    0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
    0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72,
];

/// Compute GLV private key: (λ * k) mod n
/// This gives the private key for the GLV-derived address
#[inline]
fn compute_glv_privkey(priv_key: &[u8; 32]) -> [u8; 32] {
    // Use k256 for modular multiplication
    use k256::elliptic_curve::scalar::ScalarPrimitive;
    use k256::Secp256k1;
    
    // Parse original private key as scalar
    let k_primitive: ScalarPrimitive<Secp256k1> = match ScalarPrimitive::from_slice(priv_key) {
        Ok(s) => s,
        Err(_) => return [0u8; 32], // Invalid key
    };
    let k = k256::Scalar::from(&k_primitive);
    
    // Parse lambda as scalar
    let lambda_primitive: ScalarPrimitive<Secp256k1> = match ScalarPrimitive::from_slice(&GLV_LAMBDA) {
        Ok(s) => s,
        Err(_) => return [0u8; 32],
    };
    let lambda = k256::Scalar::from(&lambda_primitive);
    
    // GLV key = (λ * k) mod n
    let glv_scalar = lambda * k;
    
    // Convert back to bytes
    let mut result = [0u8; 32];
    result.copy_from_slice(&glv_scalar.to_bytes());
    result
}

/// Format Bitcoin match - all values from GPU
/// If is_glv is true, computes the correct GLV private key
/// Shows ALL address types that can be derived from this private key
fn format_btc_match(result: &BrainwalletResult, is_glv: bool) -> String {
    let priv_key = if is_glv {
        compute_glv_privkey(&result.priv_key)
    } else {
        result.priv_key
    };
    
    let hrp = bech32::Hrp::parse("bc").unwrap();
    let match_type = if is_glv { "BITCOIN GLV MATCH" } else { "BITCOIN MATCH" };

    if is_glv {
        // GLV: only compressed addresses available from GPU
        let wif_c = compute_wif(&priv_key, 0x80, true);
        let h160_c = &result.glv_h160_c;
        
        // Compute P2SH-P2WPKH from glv_h160_c
        let h160_nested = compute_p2sh_p2wpkh_hash(h160_c);
        
        format!(
            "=== {} ===\n\
             Passphrase: {}\n\
             WIF: {}\n\
             \n\
             P2PKH (compressed): {}\n\
             P2SH-P2WPKH:        {}\n\
             Native SegWit:      {}\n\
             =====================\n\n",
            match_type,
            result.passphrase_str(),
            wif_c,
            to_base58check(0x00, h160_c),
            to_base58check(0x05, &h160_nested),
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, h160_c).unwrap_or_default(),
        )
    } else {
        // Primary: all address types available
        let wif_c = compute_wif(&priv_key, 0x80, true);
        let wif_u = compute_wif(&priv_key, 0x80, false);
        
        format!(
            "=== {} ===\n\
             Passphrase: {}\n\
             WIF (compressed):   {}\n\
             WIF (uncompressed): {}\n\
             \n\
             P2PKH (compressed):   {}\n\
             P2PKH (uncompressed): {}\n\
             P2SH-P2WPKH:          {}\n\
             Native SegWit:        {}\n\
             =====================\n\n",
            match_type,
            result.passphrase_str(),
            wif_c,
            wif_u,
            to_base58check(0x00, &result.h160_c),
            to_base58check(0x00, &result.h160_u),
            to_base58check(0x05, &result.h160_nested),
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &result.h160_c).unwrap_or_default(),
        )
    }
}

/// Format Litecoin match - all values from GPU
/// If is_glv is true, computes the correct GLV private key
/// Shows ALL address types that can be derived from this private key
fn format_ltc_match(result: &BrainwalletResult, is_glv: bool) -> String {
    let priv_key = if is_glv {
        compute_glv_privkey(&result.priv_key)
    } else {
        result.priv_key
    };
    
    let hrp = bech32::Hrp::parse("ltc").unwrap();
    let match_type = if is_glv { "LITECOIN GLV MATCH" } else { "LITECOIN MATCH" };

    if is_glv {
        // GLV: only compressed addresses available from GPU
        let wif_c = compute_wif(&priv_key, 0xB0, true);
        let h160_c = &result.glv_h160_c;
        
        // Compute P2SH-P2WPKH from glv_h160_c
        let h160_nested = compute_p2sh_p2wpkh_hash(h160_c);
        
        format!(
            "=== {} ===\n\
             Passphrase: {}\n\
             WIF: {}\n\
             \n\
             P2PKH (compressed): {}\n\
             P2SH-P2WPKH:        {}\n\
             Native SegWit:      {}\n\
             ======================\n\n",
            match_type,
            result.passphrase_str(),
            wif_c,
            to_base58check(0x30, h160_c),
            to_base58check(0x32, &h160_nested),
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, h160_c).unwrap_or_default(),
        )
    } else {
        // Primary: all address types available
        let wif_c = compute_wif(&priv_key, 0xB0, true);
        let wif_u = compute_wif(&priv_key, 0xB0, false);
        
        format!(
            "=== {} ===\n\
             Passphrase: {}\n\
             WIF (compressed):   {}\n\
             WIF (uncompressed): {}\n\
             \n\
             P2PKH (compressed):   {}\n\
             P2PKH (uncompressed): {}\n\
             P2SH-P2WPKH:          {}\n\
             Native SegWit:        {}\n\
             ======================\n\n",
            match_type,
            result.passphrase_str(),
            wif_c,
            wif_u,
            to_base58check(0x30, &result.h160_c),
            to_base58check(0x30, &result.h160_u),
            to_base58check(0x32, &result.h160_nested),
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &result.h160_c).unwrap_or_default(),
        )
    }
}

/// Format Ethereum match - address from GPU Keccak256
/// Uses EIP-55 checksum encoding for proper address formatting
/// If is_glv is true, computes the correct GLV private key
fn format_eth_match(result: &BrainwalletResult, is_glv: bool) -> String {
    let priv_key = if is_glv {
        compute_glv_privkey(&result.priv_key)
    } else {
        result.priv_key
    };
    
    let eth_addr = if is_glv { &result.glv_eth_addr } else { &result.eth_addr };
    let eth_addr_checksum = to_checksum_address(eth_addr);
    let priv_hex = hex::encode(&priv_key);
    
    let match_type = if is_glv { "ETHEREUM GLV MATCH" } else { "ETHEREUM MATCH" };

    format!(
        "=== {} ===\n\
         Passphrase: {}\n\
         Address: {}\n\
         Private Key: {}\n\
         ========================\n\n",
        match_type, result.passphrase_str(), eth_addr_checksum, priv_hex
    )
}

/// EIP-55 Mixed-case checksum address encoding
/// https://eips.ethereum.org/EIPS/eip-55
#[inline]
fn to_checksum_address(addr: &[u8; 20]) -> String {
    use tiny_keccak::{Hasher, Keccak};
    
    let addr_hex = hex::encode(addr);
    
    // Keccak256 of lowercase hex string (without 0x prefix)
    let mut hasher = Keccak::v256();
    hasher.update(addr_hex.as_bytes());
    let mut hash = [0u8; 32];
    hasher.finalize(&mut hash);
    
    // Apply checksum: uppercase if corresponding hash nibble >= 8
    let mut checksum = String::with_capacity(42);
    checksum.push_str("0x");
    
    for (i, c) in addr_hex.chars().enumerate() {
        if c.is_ascii_digit() {
            checksum.push(c);
        } else {
            // Get corresponding nibble from hash
            let hash_byte = hash[i / 2];
            let hash_nibble = if i % 2 == 0 { hash_byte >> 4 } else { hash_byte & 0x0F };
            
            if hash_nibble >= 8 {
                checksum.push(c.to_ascii_uppercase());
            } else {
                checksum.push(c); // already lowercase from hex::encode
            }
        }
    }
    
    checksum
}

/// Compute WIF from GPU-provided private key
#[inline]
fn compute_wif(priv_bytes: &[u8; 32], version: u8, compressed: bool) -> String {
    let mut wif_bytes = vec![version];
    wif_bytes.extend_from_slice(priv_bytes);
    if compressed {
        wif_bytes.push(0x01);
    }
    bs58::encode(&wif_bytes).with_check().into_string()
}

/// Base58Check encode with version byte
#[inline]
fn to_base58check(version: u8, hash: &[u8; 20]) -> String {
    let mut data = vec![version];
    data.extend_from_slice(hash);
    bs58::encode(&data).with_check().into_string()
}

/// Compute P2SH-P2WPKH script hash from compressed pubkey hash
/// Script: OP_0 <20-byte-hash> = 0x00 0x14 || h160
#[inline]
fn compute_p2sh_p2wpkh_hash(h160_c: &[u8; 20]) -> [u8; 20] {
    use ripemd::Ripemd160;
    use sha2::{Digest, Sha256};
    
    let mut script = [0u8; 22];
    script[0] = 0x00; // OP_0
    script[1] = 0x14; // Push 20 bytes
    script[2..22].copy_from_slice(h160_c);
    
    // HASH160 = RIPEMD160(SHA256(script))
    let sha = Sha256::digest(&script);
    let rip = Ripemd160::digest(&sha);
    
    let mut result = [0u8; 20];
    result.copy_from_slice(&rip);
    result
}
