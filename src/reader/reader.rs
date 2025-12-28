use crate::brainwallet::MultiWallet;
use crate::comparer::Comparer;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

#[cfg(feature = "gpu")]
use crate::metal::{BatchProcessor, BrainwalletResult, PassphraseBatcher};

/// Ortalama satır uzunluğunu tahmin ederek satır sayısını hesaplar (çift taramadan kaçınır)
fn estimate_line_count(file_size: u64) -> u64 {
    // Tipik rockyou.txt ortalama satır uzunluğu ~9 karakter + newline
    const AVG_LINE_LENGTH: u64 = 10;
    file_size / AVG_LINE_LENGTH
}

/// Sadece Windows satır sonu karakterini temizler (\r)
/// NOT: Boşluk ve tab karakterleri brainwallet passphrase'inin parçası olabilir!
#[inline(always)]
fn strip_cr(line: &[u8]) -> &[u8] {
    // Sadece sondaki \r karakterini temizle (Windows CRLF uyumu)
    if line.ends_with(b"\r") {
        &line[..line.len() - 1]
    } else {
        line
    }
}

/// GPU-accelerated cracking (when gpu feature is enabled)
#[cfg(feature = "gpu")]
pub fn start_cracking(dict: &str, comparer: &Comparer) {
    // Try GPU first
    match try_gpu_cracking(dict, comparer) {
        Ok(()) => return,
        Err(e) => {
            eprintln!("⚠️  GPU initialization failed: {}", e);
            eprintln!("   Falling back to CPU mode...\n");
        }
    }
    
    // Fall back to CPU
    start_cracking_cpu(dict, comparer);
}

/// GPU-accelerated processing
#[cfg(feature = "gpu")]
fn try_gpu_cracking(dict: &str, comparer: &Comparer) -> Result<(), String> {
    
    let file = std::fs::File::open(dict).map_err(|e| e.to_string())?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
    
    // Initialize GPU processor
    let processor = BatchProcessor::new()?;
    let batch_size = processor.max_batch_size();
    
    println!("🚀 GPU Mode Enabled");
    println!("   Batch size: {}", batch_size);
    
    let log = Mutex::new(BufWriter::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open("found.txt")
            .map_err(|e| e.to_string())?,
    ));
    
    let estimated_lines = estimate_line_count(file_size);
    let pb = ProgressBar::new(estimated_lines);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (~{eta} kaldı) {msg}")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message("GPU tarama...");
    
    let counter = AtomicU64::new(0);
    let mut batcher = PassphraseBatcher::new(&mmap, batch_size);
    
    while let Some(batch) = batcher.next_batch() {
        let batch_len = batch.len();
        
        // Process batch on GPU and check for matches
        let matches = processor.process_and_match(&batch, |h160_c, h160_u, h160_nested, taproot| {
            // Check Bitcoin
            if comparer.btc_on {
                if comparer.btc_20.contains(h160_c)
                    || comparer.btc_20.contains(h160_u)
                    || comparer.btc_20.contains(h160_nested)
                    || comparer.btc_32.contains(taproot)
                {
                    return true;
                }
            }
            
            // Check Litecoin (same hash format as Bitcoin)
            if comparer.ltc_on {
                if comparer.ltc_20.contains(h160_c)
                    || comparer.ltc_20.contains(h160_u)
                    || comparer.ltc_20.contains(h160_nested)
                    || comparer.ltc_32.contains(taproot)
                {
                    return true;
                }
            }
            
            // Note: Ethereum and Solana use different derivation paths
            // They are still processed by CPU for now
            
            false
        }).map_err(|e| e.to_string())?;
        
        // Process matches
        for result in matches {
            let pass = String::from_utf8_lossy(&result.passphrase);
            let rep = format_gpu_match(&result, &pass, comparer);
            
            if !rep.is_empty() {
                let mut f = log.lock().unwrap();
                let _ = f.write_all(rep.as_bytes());
                let _ = f.flush();
                pb.println(format!("\n{}", rep));
            }
        }
        
        // For Ethereum and Solana, we need CPU fallback for now
        // (they use different derivation: keccak256 for ETH, ed25519 for SOL)
        if comparer.eth_on || comparer.sol_on {
            for passphrase in &batch {
                let w = MultiWallet::generate_active(
                    passphrase,
                    false, // BTC already checked by GPU
                    false, // LTC already checked by GPU
                    comparer.eth_on,
                    comparer.sol_on,
                );
                
                let pass = String::from_utf8_lossy(passphrase);
                let mut rep = String::new();
                
                if let Some(eth) = w.eth {
                    if comparer.eth_20.contains(&eth.address) {
                        rep.push_str(&eth.get_report(&pass));
                    }
                }
                if let Some(sol) = w.sol {
                    if comparer.sol_32.contains(&sol.address) {
                        rep.push_str(&sol.get_report(&pass));
                    }
                }
                
                if !rep.is_empty() {
                    let mut f = log.lock().unwrap();
                    let _ = f.write_all(rep.as_bytes());
                    let _ = f.flush();
                    pb.println(format!("\n{}", rep));
                }
            }
        }
        
        // Update progress
        let current = counter.fetch_add(batch_len as u64, Ordering::Relaxed);
        if current % 10_000 < batch_len as u64 {
            pb.set_position(current + batch_len as u64);
        }
    }
    
    let final_count = counter.load(Ordering::Relaxed);
    pb.set_position(final_count);
    
    {
        if let Ok(mut f) = log.lock() {
            let _ = f.flush();
        }
    }
    
    let processed = processor.total_processed();
    pb.finish_with_message(format!(
        "Tamamlandı! {} satır tarandı (GPU: {} işlendi)",
        final_count, processed
    ));
    
    Ok(())
}

/// Format GPU match result
#[cfg(feature = "gpu")]
fn format_gpu_match(result: &BrainwalletResult, pass: &str, comparer: &Comparer) -> String {
    use sha2::{Digest, Sha256};
    
    // Recompute private key from passphrase
    let priv_bytes: [u8; 32] = Sha256::digest(&result.passphrase).into();
    
    let mut rep = String::new();
    
    // Check Bitcoin
    if comparer.btc_on {
        let btc_match = comparer.btc_20.contains(&result.h160_c)
            || comparer.btc_20.contains(&result.h160_u)
            || comparer.btc_20.contains(&result.h160_nested)
            || comparer.btc_32.contains(&result.taproot);
        
        if btc_match {
            if let Some(btc) = crate::brainwallet::BtcWallet::generate(priv_bytes) {
                rep.push_str(&btc.get_report(pass));
            }
        }
    }
    
    // Check Litecoin
    if comparer.ltc_on {
        let ltc_match = comparer.ltc_20.contains(&result.h160_c)
            || comparer.ltc_20.contains(&result.h160_u)
            || comparer.ltc_20.contains(&result.h160_nested)
            || comparer.ltc_32.contains(&result.taproot);
        
        if ltc_match {
            if let Some(ltc) = crate::brainwallet::LtcWallet::generate(priv_bytes) {
                rep.push_str(&ltc.get_report(pass));
            }
        }
    }
    
    rep
}

/// CPU-only cracking (fallback or when gpu feature is disabled)
#[cfg(not(feature = "gpu"))]
pub fn start_cracking(dict: &str, comparer: &Comparer) {
    start_cracking_cpu(dict, comparer);
}

/// CPU implementation of cracking
fn start_cracking_cpu(dict: &str, comparer: &Comparer) {
    let file = std::fs::File::open(dict).expect("Dict missing");
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    
    // Memory-mapped file yükle
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    
    // BufWriter ile daha verimli dosya yazımı
    let log = Mutex::new(BufWriter::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open("found.txt")
            .unwrap(),
    ));

    // Çift tarama yerine tahmini satır sayısı kullan
    let estimated_lines = estimate_line_count(file_size);
    let pb = ProgressBar::new(estimated_lines);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (~{eta} kaldı) {msg}")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message("CPU tarama...");
    pb.set_position(0);
    
    let counter = AtomicU64::new(0);

    mmap.par_split(|&b| b == b'\n').for_each(|raw_line| {
        // Sadece Windows \r karakterini temizle (boşluklar passphrase parçası!)
        let line = strip_cr(raw_line);
        
        if line.is_empty() {
            return;
        }

        let w = MultiWallet::generate_active(
            line,
            comparer.btc_on,
            comparer.ltc_on,
            comparer.eth_on,
            comparer.sol_on,
        );
        let pass = String::from_utf8_lossy(line);
        let mut rep = String::new();

        if let Some(btc) = w.btc {
            if comparer.btc_20.contains(&btc.h160_c)
                || comparer.btc_20.contains(&btc.h160_u)
                || comparer.btc_20.contains(&btc.h160_nested)
                || comparer.btc_32.contains(&btc.taproot)
            {
                rep.push_str(&btc.get_report(&pass));
            }
        }
        if let Some(ltc) = w.ltc {
            if comparer.ltc_20.contains(&ltc.h160_c)
                || comparer.ltc_20.contains(&ltc.h160_u)
                || comparer.ltc_20.contains(&ltc.h160_nested)
                || comparer.ltc_32.contains(&ltc.taproot)
            {
                rep.push_str(&ltc.get_report(&pass));
            }
        }
        if let Some(eth) = w.eth {
            if comparer.eth_20.contains(&eth.address) {
                rep.push_str(&eth.get_report(&pass));
            }
        }
        if let Some(sol) = w.sol {
            if comparer.sol_32.contains(&sol.address) {
                rep.push_str(&sol.get_report(&pass));
            }
        }

        if !rep.is_empty() {
            let mut f = log.lock().unwrap();
            let _ = f.write_all(rep.as_bytes());
            let _ = f.flush();
            pb.println(format!("\n{}", rep));
        }

        // Progress bar güncelleme
        let current = counter.fetch_add(1, Ordering::Relaxed);
        if current % 1_000 == 0 {
            pb.set_position(current);
        }
    });
    
    let final_count = counter.load(Ordering::Relaxed);
    pb.set_position(final_count);
    
    {
        if let Ok(mut f) = log.lock() {
            let _ = f.flush();
        }
    }
    
    pb.finish_with_message(format!("Tamamlandı! {} satır tarandı.", final_count));
}
