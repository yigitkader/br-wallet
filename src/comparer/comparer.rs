use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;

#[derive(Deserialize)]
struct TargetFile {
    addresses: Vec<String>,
}

pub struct Comparer {
    pub target_hashes: HashSet<[u8; 20]>,
}

impl Comparer {
    pub fn load_from_json(path: &str) -> Self {
        println!("Loading targets from JSON...");
        let file = File::open(path).expect("Target JSON file not found");
        let data: TargetFile = serde_json::from_reader(file).expect("JSON parsing failed");

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
        println!("{} targets loaded into memory.", target_hashes.len());
        Comparer { target_hashes }
    }

    #[inline(always)]
    pub fn is_match(&self, hash: &[u8; 20]) -> bool {
        self.target_hashes.contains(hash)
    }
}
