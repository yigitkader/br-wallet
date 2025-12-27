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
        // 1. Binary Cache Kontrolü
        if Path::new(bin_path).exists() {
            println!("Loading from binary cache: {}...", bin_path);
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

        // 2. JSON'dan Yükleme ve Hata Analizi
        println!("Parsing JSON: {}...", json_path);
        let file = File::open(json_path).expect("Target JSON file not found");
        let reader = BufReader::new(file);
        let data: TargetFile = serde_json::from_reader(reader).expect("JSON parsing failed");

        let mut target_hashes = HashSet::with_capacity(data.addresses.len());
        let mut duplicates = 0;
        let mut errors = 0;

        for addr in data.addresses {
            // Sadece Legacy (1...) adresleri Hash160'a çevirir
            match bs58::decode(&addr).with_check(None).into_vec() {
                Ok(decoded) if decoded.len() >= 21 => {
                    let mut hash = [0u8; 20];
                    hash.copy_from_slice(&decoded[1..21]);
                    if !target_hashes.insert(hash) {
                        duplicates += 1;
                    }
                }
                _ => errors += 1,
            }
        }

        println!("--- Load Report ---");
        println!("Unique Adresses: {}", target_hashes.len());
        println!("Duplicates Found: {}", duplicates);
        println!("Errors/Unsupported: {}", errors);
        println!("-------------------");

        // 3. Cache Kaydı
        let encoded = bincode::serialize(&target_hashes).expect("Serialization failed");
        let mut bin_file = File::create(bin_path).expect("Failed to create bin file");
        bin_file.write_all(&encoded).expect("Write failed");

        Comparer { target_hashes }
    }

    #[inline(always)]
    pub fn is_match(&self, hash: &[u8; 20]) -> bool {
        self.target_hashes.contains(hash)
    }
}
