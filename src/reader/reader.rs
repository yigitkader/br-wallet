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
use crate::metal::{BatchProcessor, BrainwalletResult, PassphraseBatcher};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

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

/// GPU-accelerated brainwallet cracking
///
/// ALL operations happen on GPU:
/// - SHA256 (passphrase → private key)
/// - secp256k1 scalar multiplication (private key → public key)
/// - RIPEMD160 (public key → Bitcoin/Litecoin addresses)
/// - Keccak256 (public key → Ethereum address)
///
/// # Panics
/// Panics if GPU initialization fails. This is intentional - GPU is required.
pub fn start_cracking(dict: &str, comparer: &Comparer) {
    // Initialize GPU - this MUST succeed
    let processor = match BatchProcessor::new() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ GPU initialization failed: {}", e);
            eprintln!("   This application requires Apple Metal GPU.");
            eprintln!("   Make sure you're running on macOS with Metal support.");
            std::process::exit(1);
        }
    };

    let batch_size = processor.max_batch_size();

    println!("🚀 GPU Mode: Metal Accelerated");
    println!("   Batch size: {} passphrases/dispatch", batch_size);

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
    pb.set_message("GPU scanning...");

    let counter = AtomicU64::new(0);
    let mut batcher = PassphraseBatcher::new(&mmap, batch_size);

    // Main GPU processing loop
    while let Some(batch) = batcher.next_batch() {
        let batch_len = batch.len();

        // Process batch on GPU - ALL crypto operations happen here
        let gpu_results = match processor.process(&batch) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("⚠️  GPU processing error: {}", e);
                continue;
            }
        };

        // Parallel check for all chains using GPU results
        let all_matches: Vec<String> = gpu_results
            .par_iter()
            .filter_map(|result| {
                if !result.is_valid() {
                    return None;
                }

                let pass = String::from_utf8_lossy(&result.passphrase);
                let mut rep = String::new();

                // Check Bitcoin (GPU-computed hashes)
                if comparer.btc_on {
                    if comparer.btc_20.contains(&result.h160_c)
                        || comparer.btc_20.contains(&result.h160_u)
                        || comparer.btc_20.contains(&result.h160_nested)
                        || comparer.btc_32.contains(&result.taproot)
                    {
                        rep.push_str(&format_btc_match(result, &pass));
                    }
                }

                // Check Litecoin (GPU-computed hashes)
                if comparer.ltc_on {
                    if comparer.ltc_20.contains(&result.h160_c)
                        || comparer.ltc_20.contains(&result.h160_u)
                        || comparer.ltc_20.contains(&result.h160_nested)
                        || comparer.ltc_32.contains(&result.taproot)
                    {
                        rep.push_str(&format_ltc_match(result, &pass));
                    }
                }

                // Check Ethereum (GPU-computed Keccak256 address)
                if comparer.eth_on {
                    if comparer.eth_20.contains(&result.eth_addr) {
                        rep.push_str(&format_eth_match(result, &pass));
                    }
                }

                if rep.is_empty() { None } else { Some(rep) }
            })
            .collect();

        // Write all matches
        if !all_matches.is_empty() {
            let mut f = log.lock().unwrap();
            for rep in &all_matches {
                let _ = f.write_all(rep.as_bytes());
                pb.println(format!("\n{}", rep));
            }
            let _ = f.flush();
        }

        // Update progress
        let current = counter.fetch_add(batch_len as u64, Ordering::Relaxed);
        if current % 10_000 < batch_len as u64 {
            pb.set_position(current + batch_len as u64);
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

/// Format Bitcoin match - all values from GPU
fn format_btc_match(result: &BrainwalletResult, pass: &str) -> String {
    let wif = compute_wif(&result.priv_key, 0x80, true);
    let hrp = bech32::Hrp::parse("bc").unwrap();

    format!(
        "=== BITCOIN MATCH ===\n\
         Passphrase: {}\n\
         WIF: {}\n\
         Legacy (1...):      {}\n\
         Legacy Uncomp:      {}\n\
         P2SH-SegWit (3...): {}\n\
         Native SegWit:      {}\n\
         Taproot:            {}\n\
         =====================\n\n",
        pass,
        wif,
        to_base58check(0x00, &result.h160_c),
        to_base58check(0x00, &result.h160_u),
        to_base58check(0x05, &result.h160_nested),
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &result.h160_c).unwrap_or_default(),
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &result.taproot).unwrap_or_default(),
    )
}

/// Format Litecoin match - all values from GPU
fn format_ltc_match(result: &BrainwalletResult, pass: &str) -> String {
    let wif = compute_wif(&result.priv_key, 0xB0, true);
    let hrp = bech32::Hrp::parse("ltc").unwrap();

    format!(
        "=== LITECOIN MATCH ===\n\
         Passphrase: {}\n\
         WIF: {}\n\
         Legacy (L...):      {}\n\
         Legacy Uncomp:      {}\n\
         P2SH-SegWit (M...): {}\n\
         Native SegWit:      {}\n\
         Taproot:            {}\n\
         ======================\n\n",
        pass,
        wif,
        to_base58check(0x30, &result.h160_c),   // LTC P2PKH
        to_base58check(0x30, &result.h160_u),
        to_base58check(0x32, &result.h160_nested), // LTC P2SH
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &result.h160_c).unwrap_or_default(),
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &result.taproot).unwrap_or_default(),
    )
}

/// Format Ethereum match - address from GPU Keccak256
fn format_eth_match(result: &BrainwalletResult, pass: &str) -> String {
    let eth_addr_hex = format!("0x{}", hex::encode(&result.eth_addr));
    let priv_hex = hex::encode(&result.priv_key);

    format!(
        "=== ETHEREUM MATCH ===\n\
         Passphrase: {}\n\
         Address: {}\n\
         Private Key: {}\n\
         ========================\n\n",
        pass, eth_addr_hex, priv_hex
    )
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
