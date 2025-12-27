use crate::brainwallet::BrainWallet;
use crate::comparer::Comparer;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn start_cracking(dict_path: &str, comparer: &Comparer) {
    let file = File::open(dict_path).expect("Dictionary file not found");
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

    println!("Scanning starting...");

    lines.par_iter().for_each(|passphrase| {
        let wallet = BrainWallet::new(passphrase);
        if comparer.is_match(&wallet.generated_wallet.compressed_point_conversion.hash160) {
            println!(
                "!!! MATCH FOUND (Compressed) !!! Pass: {} Addr: {}",
                passphrase, wallet.generated_wallet.compressed_point_conversion.address
            );
        }

        if comparer.is_match(
            &wallet
                .generated_wallet
                .uncompressed_point_conversion
                .hash160,
        ) {
            println!(
                "!!! MATCH FOUND (Uncompressed) !!! Pass: {} Addr: {}",
                passphrase,
                wallet
                    .generated_wallet
                    .uncompressed_point_conversion
                    .address
            );
        }
    });
}
