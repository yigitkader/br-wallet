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
    let file = File::open(dict_path).expect("Dict not found");
    let mmap = unsafe { Mmap::map(&file).expect("Mmap error") };

    let found_file = Mutex::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open("found.txt")
            .unwrap(),
    );

    let total_lines = mmap.par_split(|&b| b == b'\n').count() as u64;
    let pb = ProgressBar::new(total_lines);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} checks/s)")
        .unwrap());

    let local_counter = AtomicU64::new(0);

    mmap.par_split(|&b| b == b'\n').for_each(|line| {
        if line.is_empty() {
            return;
        }

        let wallet = BrainWallet::fast_generate(line);
        let found = comparer.is_match_20b(&wallet.h160_c)
            || comparer.is_match_20b(&wallet.h160_u)
            || comparer.is_match_20b(&wallet.h160_nested)
            || comparer.is_match_32b(&wallet.taproot);

        if found {
            let report = wallet.get_report(&String::from_utf8_lossy(line));
            if let Ok(mut f) = found_file.lock() {
                let _ = f.write_all(report.as_bytes());
            }
            pb.println(format!("\n{}", report));
        }

        let c = local_counter.fetch_add(1, Ordering::Relaxed);
        if c % 10_000 == 0 {
            pb.inc(10_000);
        }
    });
    pb.finish();
}
