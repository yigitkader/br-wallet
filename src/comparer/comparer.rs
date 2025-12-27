use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Write};

#[derive(Deserialize, Serialize)]
struct TargetFile {
    addresses: Vec<String>,
}

pub struct Comparer {
    pub btc_20: HashSet<[u8; 20]>,
    pub btc_32: HashSet<[u8; 32]>,
    pub eth_20: HashSet<[u8; 20]>,
    pub sol_32: HashSet<[u8; 32]>,
    pub btc_on: bool,
    pub eth_on: bool,
    pub sol_on: bool,
}

impl Comparer {
    pub fn load() -> Self {
        let (btc_20, btc_32) = Self::load_net("bitcoin", "btc.bin");
        let (eth_20, _) = Self::load_net("ethereum", "eth.bin");
        let (_, sol_32) = Self::load_net("solana", "sol.bin");

        Comparer {
            btc_on: !btc_20.is_empty() || !btc_32.is_empty(),
            eth_on: !eth_20.is_empty(),
            sol_on: !sol_32.is_empty(),
            btc_20,
            btc_32,
            eth_20,
            sol_32,
        }
    }

    fn load_net(name: &str, bin: &str) -> (HashSet<[u8; 20]>, HashSet<[u8; 32]>) {
        let mut h20 = HashSet::new();
        let mut h32 = HashSet::new();

        if std::path::Path::new(bin).exists() {
            let f = File::open(bin).unwrap();
            return bincode::deserialize_from(BufReader::new(f)).unwrap();
        }

        let json = format!("{}_targets.json", name);
        if let Ok(f) = File::open(&json) {
            let data: TargetFile = serde_json::from_reader(BufReader::new(f)).unwrap();
            for a in data.addresses {
                match name {
                    "bitcoin" => {
                        if a.starts_with("bc1") {
                            if let Ok((_, _, p)) = bech32::segwit::decode(&a) {
                                if p.len() == 20 {
                                    h20.insert(p.try_into().unwrap());
                                } else {
                                    h32.insert(p.try_into().unwrap());
                                }
                            }
                        } else if let Ok(d) = bs58::decode(&a).with_check(None).into_vec() {
                            if d.len() >= 21 {
                                h20.insert(d[1..21].try_into().unwrap());
                            }
                        }
                    }
                    "ethereum" => {
                        if let Ok(b) = hex::decode(a.trim_start_matches("0x")) {
                            if b.len() == 20 {
                                h20.insert(b.try_into().unwrap());
                            }
                        }
                    }
                    "solana" => {
                        if let Ok(b) = bs58::decode(&a).into_vec() {
                            if b.len() == 32 {
                                h32.insert(b.try_into().unwrap());
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !h20.is_empty() || !h32.is_empty() {
                let mut cache = File::create(bin).unwrap();
                bincode::serialize_into(cache, &(&h20, &h32)).unwrap();
            }
        }
        (h20, h32)
    }
}
