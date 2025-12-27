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

pub fn start_cracking(dict: &str, comparer: &Comparer) {
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
    pb.set_message("taranıyor...");
    pb.set_position(0); // İlk görünümü zorla
    
    let counter = AtomicU64::new(0);

    mmap.par_split(|&b| b == b'\n').for_each(|raw_line| {
        // Sadece Windows \r karakterini temizle (boşluklar passphrase parçası!)
        let line = strip_cr(raw_line);
        
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
            // flush() kaldırıldı - BufWriter kendi buffer'ını yönetsin
            // Program sonunda zaten flush ediliyor
            pb.println(format!("\n{}", rep));
        }

        // Progress bar güncelleme - her 1000 satırda bir (daha responsive)
        let current = counter.fetch_add(1, Ordering::Relaxed);
        if current % 1_000 == 0 {
            pb.set_position(current);
        }
    });
    
    // Son durumu güncelle
    let final_count = counter.load(Ordering::Relaxed);
    pb.set_position(final_count);
    
    // BufWriter'ı flush et (blok içinde drop edilsin)
    {
        if let Ok(mut f) = log.lock() {
            let _ = f.flush();
        }
    }
    
    pb.finish_with_message(format!("Tamamlandı! {} satır tarandı.", final_count));
}
