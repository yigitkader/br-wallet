use crate::brainwallet::BrainWallet;
use crate::comparer::Comparer;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

pub fn start_cracking(dict_path: &str, comparer: &Comparer) {
    let file = File::open(dict_path).expect("Dictionary not found");
    let mmap = unsafe { Mmap::map(&file).expect("Mmap failed") };

    // Log dosyasını güvenli modda aç (Yoksa oluştur, varsa sonuna ekle)
    let found_file = Mutex::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open("found.txt")
            .expect("Could not open found.txt"),
    );

    let total_lines = mmap.par_split(|&b| b == b'\n').count() as u64;
    let pb = ProgressBar::new(total_lines);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} checks/s)")
        .unwrap()
        .progress_chars("#>-"));

    let local_counter = AtomicU64::new(0);

    println!("Scan started. Successes will be logged to found.txt");

    mmap.par_split(|&b| b == b'\n').for_each(|line| {
        if line.is_empty() {
            return;
        }

        let wallet = BrainWallet::fast_generate(line);
        let pass_str = String::from_utf8_lossy(line);
        let mut match_type = "";

        // Eşleşme Kontrolleri
        if comparer.is_match_20b(&wallet.h160_c) {
            match_type = "Legacy/P2WPKH (Compressed/bc1q)";
        } else if comparer.is_match_20b(&wallet.h160_u) {
            match_type = "Legacy (Uncompressed)";
        } else if comparer.is_match_20b(&wallet.h160_nested) {
            match_type = "Nested SegWit (3...)";
        } else if comparer.is_match_32b(&wallet.taproot) {
            match_type = "Taproot (bc1p...)";
        }

        // Eğer bir eşleşme varsa dosyaya güvenli bir şekilde yaz
        if !match_type.is_empty() {
            let log_entry = format!("MATCH! Type: {} | Pass: {}\n", match_type, pass_str);

            // Mutex ile dosyayı kilitle ve yaz (Thread-safe)
            if let Ok(mut file) = found_file.lock() {
                let _ = file.write_all(log_entry.as_bytes());
                let _ = file.flush(); // Hemen diske yazılmasını sağla
            }

            pb.println(format!("\n{}", log_entry));
        }

        // Hız için batch güncelleme
        let c = local_counter.fetch_add(1, Ordering::Relaxed);
        if c % 100_000 == 0 {
            pb.inc(100_000);
        }
    });

    pb.finish_with_message("Scan complete.");
}
