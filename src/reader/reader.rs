use crate::brainwallet::BrainWallet;
use crate::comparer::Comparer;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub fn start_cracking(dict_path: &str, comparer: &Comparer) {
    let file = File::open(dict_path).expect("Dictionary file not found");
    let mmap = unsafe { Mmap::map(&file).expect("Memory mapping failed") };

    let total_checks = AtomicU64::new(0);
    let start_time = Instant::now();

    println!("Scan started. Targets are active.");

    mmap.par_split(|&b| b == b'\n').for_each(|line| {
        if line.is_empty() {
            return;
        }

        let wallet = BrainWallet::fast_generate(line);

        if comparer.is_match(&wallet.h160_compressed) {
            println!(
                "\n[MATCH COMPRESSED] Pass: {:?} | Addr: 1...",
                String::from_utf8_lossy(line)
            );
        }

        if comparer.is_match(&wallet.h160_uncompressed) {
            println!(
                "\n[MATCH UNCOMPRESSED] Pass: {:?} | Addr: 1...",
                String::from_utf8_lossy(line)
            );
        }

        let count = total_checks.fetch_add(1, Ordering::Relaxed);
        if count % 1_000_000 == 0 && count > 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            println!(
                "Current Speed: {:.2} Million/sec | Total: {}M",
                (count as f64 / elapsed) / 1_000_000.0,
                count / 1_000_000
            );
        }
    });
}
