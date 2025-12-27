use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;

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

        // Yüklenen adres sayılarını göster
        let btc_count = btc_20.len() + btc_32.len();
        let eth_count = eth_20.len();
        let sol_count = sol_32.len();
        
        if btc_count > 0 {
            println!("📦 Bitcoin: {} adres (Legacy/SegWit: {}, Taproot: {})", 
                     btc_count, btc_20.len(), btc_32.len());
        }
        if eth_count > 0 {
            println!("📦 Ethereum: {} adres", eth_count);
        }
        if sol_count > 0 {
            println!("📦 Solana: {} adres", sol_count);
        }

        Comparer {
            btc_on: btc_count > 0,
            eth_on: eth_count > 0,
            sol_on: sol_count > 0,
            btc_20,
            btc_32,
            eth_20,
            sol_32,
        }
    }

    fn load_net(name: &str, bin: &str) -> (HashSet<[u8; 20]>, HashSet<[u8; 32]>) {
        let mut h20 = HashSet::new();
        let mut h32 = HashSet::new();

        // Önce binary cache'i dene
        if std::path::Path::new(bin).exists() {
            if let Ok(f) = File::open(bin) {
                match bincode::deserialize_from(BufReader::new(f)) {
                    Ok(data) => return data,
                    Err(e) => {
                        eprintln!("⚠️  Cache dosyası bozuk ({}): {}", bin, e);
                        // Bozuk cache'i sil, JSON'dan yeniden yükle
                        let _ = std::fs::remove_file(bin);
                    }
                }
            }
        }

        let json = format!("{}_targets.json", name);
        if let Ok(f) = File::open(&json) {
            let data: TargetFile = match serde_json::from_reader(BufReader::new(f)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("⚠️  JSON parse hatası ({}): {}", json, e);
                    return (h20, h32);
                }
            };
            for raw_addr in data.addresses {
                // Whitespace temizliği
                let a = raw_addr.trim();
                if a.is_empty() {
                    continue;
                }
                
                match name {
                    "bitcoin" => {
                        if a.starts_with("bc1") {
                            if let Ok((_, _, p)) = bech32::segwit::decode(a) {
                                // Native SegWit (bc1q) = 20 byte, Taproot (bc1p) = 32 byte
                                if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                                    h20.insert(arr);
                                } else if let Ok(arr) = <[u8; 32]>::try_from(p.as_slice()) {
                                    h32.insert(arr);
                                }
                                // Diğer uzunluklar sessizce atlanır
                            }
                        } else if let Ok(d) = bs58::decode(a).with_check(None).into_vec() {
                            // Legacy (1...) ve P2SH (3...) = 21 byte (1 version + 20 hash)
                            if let Ok(arr) = <[u8; 20]>::try_from(&d[1..21]) {
                                h20.insert(arr);
                            }
                        }
                    }
                    "ethereum" => {
                        if let Ok(b) = hex::decode(a.trim_start_matches("0x")) {
                            if let Ok(arr) = <[u8; 20]>::try_from(b.as_slice()) {
                                h20.insert(arr);
                            }
                        }
                    }
                    "solana" => {
                        if let Ok(b) = bs58::decode(a).into_vec() {
                            if let Ok(arr) = <[u8; 32]>::try_from(b.as_slice()) {
                                h32.insert(arr);
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Binary cache oluştur (hata olursa sessizce atla)
            if !h20.is_empty() || !h32.is_empty() {
                if let Ok(cache) = File::create(bin) {
                    let _ = bincode::serialize_into(cache, &(&h20, &h32));
                }
            }
        }
        (h20, h32)
    }
}
