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
        let (btc_20, btc_32) = Self::load_net("bitcoin");
        let (eth_20, _) = Self::load_net("ethereum");
        let (_, sol_32) = Self::load_net("solana");

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

    fn load_net(name: &str) -> (HashSet<[u8; 20]>, HashSet<[u8; 32]>) {
        let mut h20 = HashSet::new();
        let mut h32 = HashSet::new();
        
        // Tutarlı dosya isimleri: {name}_targets.json → {name}_targets.bin
        let json_path = format!("{}_targets.json", name);
        let bin_path = format!("{}_targets.bin", name);

        // Önce binary cache'i dene (daha hızlı yükleme)
        if std::path::Path::new(&bin_path).exists() {
            if let Ok(meta) = std::fs::metadata(&bin_path) {
                let size_mb = meta.len() as f64 / 1_048_576.0;
                print!("   {} cache yükleniyor ({:.1} MB)... ", name, size_mb);
                let _ = std::io::stdout().flush();
            }
            
            if let Ok(f) = File::open(&bin_path) {
                match bincode::deserialize_from(BufReader::new(f)) {
                    Ok(data) => {
                        println!("✓");
                        return data;
                    }
                    Err(e) => {
                        println!("✗");
                        eprintln!("   ⚠️  Cache dosyası bozuk: {}", e);
                        let _ = std::fs::remove_file(&bin_path);
                    }
                }
            }
        }

        // JSON'dan yükle
        let json = &json_path;
        if let Ok(meta) = std::fs::metadata(json) {
            let size_mb = meta.len() as f64 / 1_048_576.0;
            if size_mb > 10.0 {
                println!("   {} JSON yükleniyor ({:.1} MB) - bu biraz sürebilir...", name, size_mb);
            } else {
                print!("   {} JSON yükleniyor... ", name);
                let _ = std::io::stdout().flush();
            }
        }
        
        if let Ok(f) = File::open(&json) {
            let data: TargetFile = match serde_json::from_reader(BufReader::new(f)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("⚠️  JSON parse hatası ({}): {}", json, e);
                    return (h20, h32);
                }
            };
            
            let total = data.addresses.len();
            let mut processed = 0;
            let show_progress = total > 100_000;
            
            for raw_addr in data.addresses {
                processed += 1;
                if show_progress && processed % 500_000 == 0 {
                    println!("   ... {}/{} adres işlendi", processed, total);
                }
                // Whitespace temizliği
                let a = raw_addr.trim();
                if a.is_empty() {
                    continue;
                }
                
                match name {
                    "bitcoin" => {
                        if a.starts_with("bc1") {
                            // Bech32: Native SegWit (bc1q) veya Taproot (bc1p)
                            if let Ok((_, _, p)) = bech32::segwit::decode(a) {
                                if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                                    h20.insert(arr); // bc1q - Native SegWit
                                } else if let Ok(arr) = <[u8; 32]>::try_from(p.as_slice()) {
                                    h32.insert(arr); // bc1p - Taproot
                                }
                            }
                        } else if let Ok(d) = bs58::decode(a).with_check(None).into_vec() {
                            // Base58Check: Legacy (1...) veya P2SH (3...)
                            // Format: 1 byte version + 20 byte hash = 21 byte minimum
                            if d.len() >= 21 {
                                if let Ok(arr) = <[u8; 20]>::try_from(&d[1..21]) {
                                    h20.insert(arr);
                                }
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
            println!("   ✓ {} adres yüklendi", h20.len() + h32.len());
            
            // Binary cache oluştur (sonraki çalıştırmalar için)
            if !h20.is_empty() || !h32.is_empty() {
                print!("   💾 Cache oluşturuluyor... ");
                let _ = std::io::stdout().flush();
                if let Ok(cache) = File::create(&bin_path) {
                    if bincode::serialize_into(cache, &(&h20, &h32)).is_ok() {
                        if let Ok(meta) = std::fs::metadata(&bin_path) {
                            let size_mb = meta.len() as f64 / 1_048_576.0;
                            println!("✓ ({:.1} MB)", size_mb);
                        } else {
                            println!("✓");
                        }
                    }
                }
            }
        }
        (h20, h32)
    }
}
