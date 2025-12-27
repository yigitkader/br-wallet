use crate::brainwallet::BrainWallet;
use crate::comparer::Comparer;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;

pub fn start_cracking(dict_path: &str, comparer: &Comparer) {
    let file = File::open(dict_path).expect("Dictionary not found");
    let mmap = unsafe { Mmap::map(&file).expect("Mmap failed") };

    let total_lines = mmap.par_split(|&b| b == b'\n').count() as u64;

    let pb = ProgressBar::new(total_lines);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} checks/s)")
        .unwrap()
        .progress_chars("#>-"));

    println!("Scanning started...");

    mmap.par_split(|&b| b == b'\n').for_each(|line| {
        if line.is_empty() {
            pb.inc(1);
            return;
        }

        let wallet = BrainWallet::fast_generate(line);

        if comparer.is_match(&wallet.h160_compressed) {
            pb.println(format!(
                "\n!!! MATCH (Comp) !!! Pass: {:?}",
                String::from_utf8_lossy(line)
            ));
        }

        if comparer.is_match(&wallet.h160_uncompressed) {
            pb.println(format!(
                "\n!!! MATCH (Uncomp) !!! Pass: {:?}",
                String::from_utf8_lossy(line)
            ));
        }

        pb.inc(1);
    });

    pb.finish_with_message("Scan complete.");
}
