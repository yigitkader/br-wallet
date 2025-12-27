use crate::brainwallet::MultiWallet;
use crate::comparer::Comparer;
use indicatif::ProgressBar;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

pub fn start_cracking(dict: &str, comparer: &Comparer) {
    let file = std::fs::File::open(dict).expect("Dict missing");
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let log = Mutex::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open("found.txt")
            .unwrap(),
    );

    let pb = ProgressBar::new(mmap.par_split(|&b| b == b'\n').count() as u64);
    let counter = AtomicU64::new(0);

    mmap.par_split(|&b| b == b'\n').for_each(|line| {
        if line.is_empty() {
            return;
        }

        let w =
            MultiWallet::generate_active(line, comparer.btc_on, comparer.eth_on, comparer.sol_on);
        let pass = String::from_utf8_lossy(line);
        let mut rep = String::new();

        if let Some(btc) = w.btc {
            if comparer.btc_20.contains(&btc.h160_c)
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
            pb.println(format!("\n{}", rep));
        }

        if counter.fetch_add(1, Ordering::Relaxed) % 10_000 == 0 {
            pb.inc(10_000);
        }
    });
    pb.finish();
}
