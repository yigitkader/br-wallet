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

/// Örnek taramayla satır sayısını tahmin eder (daha doğru)
/// 
/// İlk 1MB'ı tarayarak ortalama satır uzunluğunu hesaplar,
/// sonra dosya boyutuna göre toplam satır sayısını tahmin eder.
fn estimate_line_count(mmap: &Mmap) -> u64 {
    const SAMPLE_SIZE: usize = 1_048_576; // 1 MB
    
    let sample_len = SAMPLE_SIZE.min(mmap.len());
    if sample_len == 0 {
        return 0;
    }
    
    let sample = &mmap[..sample_len];
    let sample_lines = sample.iter().filter(|&&b| b == b'\n').count();
    
    if sample_lines == 0 {
        // Fallback: tek satırlık dosya veya \n yok
        return if mmap.len() > 0 { 1 } else { 0 };
    }
    
    // Ortalama satır uzunluğu = örnek boyut / örnek satır sayısı
    // Toplam tahmin = dosya boyutu / ortalama satır uzunluğu
    let total_lines = (mmap.len() as u64 * sample_lines as u64) / sample_len as u64;
    total_lines
}

/// Satır temizleme: CRLF/LF ve leading/trailing whitespace
/// 
/// Wordlist dosyalarında genellikle:
/// - Windows CRLF (\r\n) veya Unix LF (\n) satır sonları
/// - Yanlışlıkla eklenen boşluk/tab karakterleri olabilir
/// 
/// Brainwallet passphrases genellikle trim edilmiş formatta kullanılır.
#[inline(always)]
fn clean_line(line: &[u8]) -> &[u8] {
    let mut l = line;
    
    // Strip line endings (CRLF, LF, CR)
    if l.ends_with(b"\r\n") {
        l = &l[..l.len() - 2];
    } else if l.ends_with(b"\n") || l.ends_with(b"\r") {
        l = &l[..l.len() - 1];
    }
    
    // Trim leading whitespace (space, tab)
    while !l.is_empty() && (l[0] == b' ' || l[0] == b'\t') {
        l = &l[1..];
    }
    
    // Trim trailing whitespace (space, tab)
    while !l.is_empty() && (l[l.len() - 1] == b' ' || l[l.len() - 1] == b'\t') {
        l = &l[..l.len() - 1];
    }
    
    l
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
    
    let estimated_lines = estimate_line_count(&mmap);
    println!("📊 Tahmini satır sayısı: {}", estimated_lines);
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
        
        // Process batch on GPU - get ALL results for BTC/LTC/ETH checking
        let gpu_results = processor.process(&batch).map_err(|e| e.to_string())?;
        
        // Parallel check for all chains using GPU results
        let all_matches: Vec<String> = gpu_results.par_iter()
            .filter_map(|result| {
                if !result.is_valid() {
                    return None;
                }
                
                let pass = String::from_utf8_lossy(&result.passphrase);
                let mut rep = String::new();
                
                // Check Bitcoin
                if comparer.btc_on {
                    if comparer.btc_20.contains(&result.h160_c)
                        || comparer.btc_20.contains(&result.h160_u)
                        || comparer.btc_20.contains(&result.h160_nested)
                        || comparer.btc_32.contains(&result.taproot)
                    {
                        rep.push_str(&format_gpu_match(result, &pass, comparer));
                    }
                }
                
                // Check Litecoin
                if comparer.ltc_on {
                    if comparer.ltc_20.contains(&result.h160_c)
                        || comparer.ltc_20.contains(&result.h160_u)
                        || comparer.ltc_20.contains(&result.h160_nested)
                        || comparer.ltc_32.contains(&result.taproot)
                    {
                        // For LTC match, generate proper report
                        rep.push_str(&format!("[LTC MATCH] passphrase: {}\n", pass));
                    }
                }
                
                // Check Ethereum - use GPU pubkey_u with Keccak256 (NO secp256k1 re-computation!)
                if comparer.eth_on {
                    use tiny_keccak::{Hasher, Keccak};
                    let mut keccak = Keccak::v256();
                    keccak.update(&result.pubkey_u);
                    let mut hash = [0u8; 32];
                    keccak.finalize(&mut hash);
                    
                    let eth_addr: [u8; 20] = hash[12..32].try_into().unwrap();
                    
                    if comparer.eth_20.contains(&eth_addr) {
                        let eth_addr_hex = format!("0x{}", hex::encode(&eth_addr));
                        rep.push_str(&format!(
                            "=== ETHEREUM MATCH ===\n\
                             Passphrase: {}\n\
                             Address: {}\n\
                             ========================\n\n",
                            pass, eth_addr_hex
                        ));
                    }
                }
                
                // For Solana, we still need CPU (Ed25519 derivation)
                if comparer.sol_on {
                    let w = MultiWallet::generate_active(
                        &result.passphrase,
                        false, false, false, true,
                    );
                    if let Some(sol) = w.sol {
                        if comparer.sol_32.contains(&sol.address) {
                            rep.push_str(&sol.get_report(&pass));
                        }
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

    // Örnek taramayla gerçek satır sayısı tahmini
    let estimated_lines = estimate_line_count(&mmap);
    println!("📊 Tahmini satır sayısı: {}", estimated_lines);
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
        let line = clean_line(raw_line);
        
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
