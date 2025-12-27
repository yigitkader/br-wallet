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

/// Ortalama satır uzunluğunu tahmin ederek satır sayısını hesaplar (çift taramadan kaçınır)
fn estimate_line_count(file_size: u64) -> u64 {
    // Tipik rockyou.txt ortalama satır uzunluğu ~9 karakter + newline
    const AVG_LINE_LENGTH: u64 = 10;
    file_size / AVG_LINE_LENGTH
}

/// Windows/Unix satır sonu karakterlerini temizler (\r\n -> \n uyumu)
#[inline(always)]
fn trim_line(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    // Sondaki \r ve boşlukları temizle
    while end > 0 && matches!(line[end - 1], b'\r' | b' ' | b'\t') {
        end -= 1;
    }
    // Baştaki boşlukları temizle
    let mut start = 0;
    while start < end && matches!(line[start], b' ' | b'\t') {
        start += 1;
    }
    &line[start..end]
}

pub fn start_cracking(dict: &str, comparer: &Comparer) {
    let file = std::fs::File::open(dict).expect("Dict missing");
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
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
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
            .unwrap()
            .progress_chars("=>-"),
    );
    
    let counter = AtomicU64::new(0);

    mmap.par_split(|&b| b == b'\n').for_each(|raw_line| {
        // Windows satır sonu temizliği (\r karakteri)
        let line = trim_line(raw_line);
        
        if line.is_empty() {
            return;
        }

        let w =
            MultiWallet::generate_active(line, comparer.btc_on, comparer.eth_on, comparer.sol_on);
        let pass = String::from_utf8_lossy(line);
        let mut rep = String::new();

        if let Some(btc) = w.btc {
            if comparer.btc_20.contains(&btc.h160_c)
                || comparer.btc_20.contains(&btc.h160_u)  // Legacy uncompressed
                || comparer.btc_20.contains(&btc.h160_nested)
                || comparer.btc_32.contains(&btc.taproot)
            {
                rep.push_str(&btc.get_report(&pass));
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
            let _ = f.flush(); // Kritik buluş için hemen disk'e yaz
            pb.println(format!("\n{}", rep));
        }

        if counter.fetch_add(1, Ordering::Relaxed) % 10_000 == 0 {
            pb.inc(10_000);
        }
    });
    
    // BufWriter'ı flush et
    if let Ok(mut f) = log.lock() {
        let _ = f.flush();
    }
    
    pb.finish_with_message("Tarama tamamlandı!");
}
