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
    pub targets_20b: HashSet<[u8; 20]>,
    pub targets_32b: HashSet<[u8; 32]>,
}

impl Comparer {
    pub fn load(json_path: &str, bin_path: &str) -> Self {
        if Path::new(bin_path).exists() {
            let file = File::open(bin_path).expect("Cache error");
            let (t20, t32): (HashSet<[u8; 20]>, HashSet<[u8; 32]>) =
                bincode::deserialize_from(BufReader::new(file)).expect("Cache parse error");
            println!(
                "Targets loaded from cache: 20b: {}, 32b: {}",
                t20.len(),
                t32.len()
            );
            return Comparer {
                targets_20b: t20,
                targets_32b: t32,
            };
        }

        println!("Decoding targets from JSON...");
        let file = File::open(json_path).expect("JSON not found");
        let data: TargetFile = serde_json::from_reader(BufReader::new(file)).expect("JSON error");

        let mut t20 = HashSet::with_capacity(data.addresses.len());
        let mut t32 = HashSet::with_capacity(1_000_000);

        for addr in data.addresses {
            if addr.starts_with("bc1") {
                // Native SegWit / Taproot
                if let Ok((_hrp, _ver, prog)) = bech32::segwit::decode(&addr) {
                    if prog.len() == 20 {
                        let mut h = [0u8; 20];
                        h.copy_from_slice(&prog);
                        t20.insert(h);
                    } else if prog.len() == 32 {
                        let mut h = [0u8; 32];
                        h.copy_from_slice(&prog);
                        t32.insert(h);
                    }
                }
            } else {
                // Legacy / Nested
                if let Ok(decoded) = bs58::decode(&addr).with_check(None).into_vec() {
                    if decoded.len() >= 21 {
                        let mut h = [0u8; 20];
                        h.copy_from_slice(&decoded[1..21]);
                        t20.insert(h);
                    }
                }
            }
        }

        let mut bin_file = File::create(bin_path).expect("Cache create failed");
        let encoded = bincode::serialize(&(&t20, &t32)).expect("Serialization failed");
        bin_file.write_all(&encoded).expect("Write failed");

        Comparer {
            targets_20b: t20,
            targets_32b: t32,
        }
    }

    #[inline(always)]
    pub fn is_match_20b(&self, h: &[u8; 20]) -> bool {
        self.targets_20b.contains(h)
    }

    #[inline(always)]
    pub fn is_match_32b(&self, h: &[u8; 32]) -> bool {
        self.targets_32b.contains(h)
    }
}
