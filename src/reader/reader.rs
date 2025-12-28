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
        0,                                          // Başlangıç
        mmap.len() / 2,                            // Orta
        mmap.len().saturating_sub(SAMPLE_SIZE),    // Son
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
        // Fallback: tek satırlık dosya veya \n yok
        return if mmap.len() > 0 { 1 } else { 0 };
    }
    
    // Toplam tahmin = dosya boyutu * (toplam satır / toplam örnek byte)
    (mmap.len() as u64 * total_lines) / total_bytes as u64
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
                        rep.push_str(&format_ltc_match(result, &pass));
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

/// Format GPU match result - uses GPU-computed hashes directly (NO secp256k1 re-computation!)
#[cfg(feature = "gpu")]
fn format_gpu_match(result: &BrainwalletResult, pass: &str, _comparer: &Comparer) -> String {
    use sha2::{Digest, Sha256};
    
    // Compute private key (just SHA256, no secp256k1)
    let priv_bytes: [u8; 32] = Sha256::digest(&result.passphrase).into();
    
    // Compute WIF (compressed) - only SHA256 double hash, no EC operations
    let wif = compute_wif(&priv_bytes, 0x80, true);
    
    // Build addresses from GPU-computed hashes (NO secp256k1!)
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
        to_b58_static(0x00, &result.h160_c),
        to_b58_static(0x00, &result.h160_u),
        to_b58_static(0x05, &result.h160_nested),
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &result.h160_c).unwrap_or_default(),
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &result.taproot).unwrap_or_default(),
    )
}

/// Compute WIF from private key bytes (only SHA256 double hash, no EC)
#[cfg(feature = "gpu")]
fn compute_wif(priv_bytes: &[u8; 32], version: u8, compressed: bool) -> String {
    let mut wif_bytes = vec![version];
    wif_bytes.extend_from_slice(priv_bytes);
    if compressed {
        wif_bytes.push(0x01);
    }
    bs58::encode(&wif_bytes).with_check().into_string()
}

/// Base58Check encode with version byte
#[cfg(feature = "gpu")]
fn to_b58_static(version: u8, hash: &[u8; 20]) -> String {
    let mut data = vec![version];
    data.extend_from_slice(hash);
    bs58::encode(&data).with_check().into_string()
}

/// Format LTC match result - uses GPU-computed hashes directly (NO secp256k1!)
#[cfg(feature = "gpu")]
fn format_ltc_match(result: &BrainwalletResult, pass: &str) -> String {
    use sha2::{Digest, Sha256};
    
    // Compute private key (just SHA256, no secp256k1)
    let priv_bytes: [u8; 32] = Sha256::digest(&result.passphrase).into();
    
    // Compute WIF (Litecoin version byte 0xB0)
    let wif = compute_wif(&priv_bytes, 0xB0, true);
    
    // Build addresses from GPU-computed hashes (Litecoin version bytes)
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
        to_b58_static(0x30, &result.h160_c),   // LTC P2PKH version
        to_b58_static(0x30, &result.h160_u),
        to_b58_static(0x32, &result.h160_nested), // LTC P2SH version
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &result.h160_c).unwrap_or_default(),
        bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &result.taproot).unwrap_or_default(),
    )
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

        // Progress bar güncelleme (her 10K satırda - GPU ile tutarlı)
        let current = counter.fetch_add(1, Ordering::Relaxed);
        if current % 10_000 == 0 {
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
