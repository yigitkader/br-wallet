use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct Comparer {
    target_hashes: HashSet<[u8; 20]>, //  store 20 byte Hash as binary
}

impl Comparer {
    pub fn load_from_file(path: &str) -> Self {
        let file = File::open(path).expect("Targets address file not found");
        let reader = BufReader::new(file);
        let mut target_hashes = HashSet::with_capacity(60_000_000);

        for line in reader.lines() {
            if let Ok(addr) = line {
                // Burada base58 adresi Hash160'a çevirme mantığı olmalı
                // Hız için dosyayı doğrudan Hash160 (hex) olarak hazırlamanı öneririm
                if let Ok(decoded) = hex::decode(addr) {
                    let mut hash = [0u8; 20];
                    hash.copy_from_slice(&decoded[..20]);
                    target_hashes.insert(hash);
                }
            }
        }
        println!("{} addresses loaded.", target_hashes.len());
        Comparer { target_hashes }
    }

    pub fn is_match(&self, hash160_hex: &str) -> bool {
        if let Ok(bytes) = hex::decode(hash160_hex) {
            let mut check = [0u8; 20];
            check.copy_from_slice(&bytes);
            return self.target_hashes.contains(&check);
        }
        false
    }
}
