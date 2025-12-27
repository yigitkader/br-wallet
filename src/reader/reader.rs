use crate::brainwallet::BrainWallet;
use crate::comparer::Comparer;
use memmap2::Mmap;
use rayon::prelude::*;
use secp256k1::Secp256k1;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub fn start_cracking(dict_path: &str, comparer: &Comparer) {
    let file = File::open(dict_path).expect("Dict not found");
    let mmap = unsafe { Mmap::map(&file).expect("Mmap failed") };
    let total_checks = AtomicU64::new(0);
    let start_time = Instant::now();

    // Context her thread için ayrı oluşturulur (En hızlısı)
    let secp = Secp256k1::new();

    println!("Scanning started...");

    // Dosyayı satır sonlarına göre böl ve paralel işle
    mmap.par_split(|&b| b == b'\n').for_each(|line| {
        if line.is_empty() {
            return;
        }

        let wallet = BrainWallet::fast_generate(line, &secp);

        // Compressed Check
        if comparer.is_match(&wallet.h160_compressed) {
            println!(
                "\n!!! MATCH FOUND !!! Pass: {:?} | Addr Type: Compressed",
                String::from_utf8_lossy(line)
            );
        }

        // Uncompressed Check
        if comparer.is_match(&wallet.h160_uncompressed) {
            println!(
                "\n!!! MATCH FOUND !!! Pass: {:?} | Addr Type: Uncompressed",
                String::from_utf8_lossy(line)
            );
        }

        let current = total_checks.fetch_add(1, Ordering::Relaxed);
        if current % 1_000_000 == 0 && current > 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            println!(
                "Speed: {:.2} Million/sec | Total: {}M",
                (current as f64 / elapsed) / 1_000_000.0,
                current / 1_000_000
            );
        }
    });
}
