use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

#[derive(Deserialize, Serialize)]
struct TargetFile {
    addresses: Vec<String>,
}

pub struct Comparer {
    pub target_hashes: HashSet<[u8; 20]>,
}

impl Comparer {
    pub fn load(json_path: &str, bin_path: &str) -> Self {
        if Path::new(bin_path).exists() {
            println!("Loading from binary cache (very fast): {}...", bin_path);
            let file = File::open(bin_path).expect("Failed to open binary cache");
            let reader = BufReader::new(file);

            let target_hashes: HashSet<[u8; 20]> = bincode::deserialize_from(reader)
                .expect("Failed to deserialize binary cache. Try deleting it.");

            println!(
                "{} targets loaded instantly from cache.",
                target_hashes.len()
            );
            return Comparer { target_hashes };
        }

        // 2. Cache yoksa JSON'dan yükle
        println!(
            "Binary cache not found. Parsing JSON (this might take a while): {}...",
            json_path
        );
        let file = File::open(json_path).expect("Target JSON file not found");
        let reader = BufReader::new(file);
        let data: TargetFile = serde_json::from_reader(reader).expect("JSON parsing failed");

        let mut target_hashes = HashSet::with_capacity(data.addresses.len());

        for addr in data.addresses {
            if let Ok(decoded) = bs58::decode(addr).with_check(None).into_vec() {
                if decoded.len() >= 21 {
                    let mut hash = [0u8; 20];
                    hash.copy_from_slice(&decoded[1..21]);
                    target_hashes.insert(hash);
                }
            }
        }

        println!("Saving targets to binary cache for next run...");
        let mut bin_file = File::create(bin_path).expect("Failed to create binary cache file");
        let encoded: Vec<u8> =
            bincode::serialize(&target_hashes).expect("Failed to serialize targets");
        bin_file
            .write_all(&encoded)
            .expect("Failed to write to binary cache");

        println!("Done! {} targets loaded and cached.", target_hashes.len());
        Comparer { target_hashes }
    }

    #[inline(always)]
    pub fn is_match(&self, hash: &[u8; 20]) -> bool {
        self.target_hashes.contains(hash)
    }
}
