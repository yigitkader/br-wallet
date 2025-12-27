use crate::brainwallet::BrainWallet;
use crate::comparer::Comparer;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn start_cracking(dict_path: &str, comparer: &Comparer) {
    let file = File::open(dict_path).expect("Dictionary not found");
    let mmap = unsafe { Mmap::map(&file).expect("Mmap failed") };

    // Satır sayısını say ve Progress Bar'ı ayarla
    let total_lines = mmap.par_split(|&b| b == b'\n').count() as u64;
    let pb = ProgressBar::new(total_lines);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} checks/s)")
        .unwrap()
        .progress_chars("#>-"));

    let local_counter = AtomicU64::new(0);

    mmap.par_split(|&b| b == b'\n').for_each(|line| {
        if line.is_empty() {
            return;
        }

        // Şifreden 4 farklı hash tipini üret
        let wallet = BrainWallet::fast_generate(line);

        // 1 & 3. Legacy (Comp) ve Native SegWit (bc1q)
        if comparer.is_match_20b(&wallet.h160_c) {
            pb.println(format!(
                "\nMATCH! (1... / bc1q...) Pass: {:?}",
                String::from_utf8_lossy(line)
            ));
        }

        // 2. Legacy (Uncompressed)
        if comparer.is_match_20b(&wallet.h160_u) {
            pb.println(format!(
                "\nMATCH! (1... Uncomp) Pass: {:?}",
                String::from_utf8_lossy(line)
            ));
        }

        // 3. Nested SegWit (3...)
        if comparer.is_match_20b(&wallet.h160_nested) {
            pb.println(format!(
                "\nMATCH! (3...) Pass: {:?}",
                String::from_utf8_lossy(line)
            ));
        }

        // 4. Taproot (bc1p)
        if comparer.is_match_32b(&wallet.taproot) {
            pb.println(format!(
                "\nMATCH! (bc1p...) Pass: {:?}",
                String::from_utf8_lossy(line)
            ));
        }

        // Progress bar'ı 10.000 satırda bir güncelle (Kritik hız ayarı!)
        let c = local_counter.fetch_add(1, Ordering::Relaxed);
        if c % 10_000 == 0 {
            pb.inc(10_000);
        }
    });

    pb.finish_with_message("Scan complete.");
}
