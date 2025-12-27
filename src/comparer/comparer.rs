use indicatif::{ProgressBar, ProgressStyle};
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
        let (btc_20, btc_32) = Self::load_net("bitcoin");
        let (eth_20, _) = Self::load_net("ethereum");
        let (_, sol_32) = Self::load_net("solana");

        // Yüklenen adres sayılarını göster
        let btc_count = btc_20.len() + btc_32.len();
        let eth_count = eth_20.len();
        let sol_count = sol_32.len();
        
        println!(); // Boş satır
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
        
        let json_path = format!("{}_targets.json", name);
        let bin_path = format!("{}_targets.bin", name);

        // Önce binary cache'i dene (daha hızlı yükleme)
        if std::path::Path::new(&bin_path).exists() {
            let size_mb = std::fs::metadata(&bin_path)
                .map(|m| m.len() as f64 / 1_048_576.0)
                .unwrap_or(0.0);
            
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .unwrap()
            );
            pb.set_message(format!("{} cache yükleniyor ({:.1} MB)...", name, size_mb));
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            
            if let Ok(f) = File::open(&bin_path) {
                match bincode::deserialize_from(BufReader::new(f)) {
                    Ok(data) => {
                        pb.finish_with_message(format!("✅ {} cache yüklendi ({:.1} MB)", name, size_mb));
                        return data;
                    }
                    Err(e) => {
                        pb.finish_with_message(format!("❌ {} cache bozuk: {}", name, e));
                        let _ = std::fs::remove_file(&bin_path);
                    }
                }
            }
        }

        // JSON dosyası var mı kontrol et
        if !std::path::Path::new(&json_path).exists() {
            return (h20, h32);
        }

        let size_mb = std::fs::metadata(&json_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);

        // JSON parse için spinner
        let parse_pb = ProgressBar::new_spinner();
        parse_pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.yellow} {msg}")
                .unwrap()
        );
        parse_pb.set_message(format!("{} JSON parse ediliyor ({:.1} MB)...", name, size_mb));
        parse_pb.enable_steady_tick(std::time::Duration::from_millis(100));
        
        let f = match File::open(&json_path) {
            Ok(f) => f,
            Err(_) => {
                parse_pb.finish_with_message(format!("❌ {} dosyası açılamadı", name));
                return (h20, h32);
            }
        };
        
        let data: TargetFile = match serde_json::from_reader(BufReader::new(f)) {
            Ok(d) => {
                parse_pb.finish_and_clear();
                d
            }
            Err(e) => {
                parse_pb.finish_with_message(format!("❌ {} JSON hatası: {}", name, e));
                return (h20, h32);
            }
        };
        
        let total = data.addresses.len() as u64;
        
        // Adres işleme için progress bar
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("   {spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░")
        );
        pb.set_message(format!("{} adresleri işleniyor", name));
        
        for (i, raw_addr) in data.addresses.into_iter().enumerate() {
            if i % 10_000 == 0 {
                pb.set_position(i as u64);
            }
            
            let a = raw_addr.trim();
            if a.is_empty() {
                continue;
            }
            
            match name {
                "bitcoin" => {
                    if a.starts_with("bc1") {
                        if let Ok((_, _, p)) = bech32::segwit::decode(a) {
                            if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                                h20.insert(arr);
                            } else if let Ok(arr) = <[u8; 32]>::try_from(p.as_slice()) {
                                h32.insert(arr);
                            }
                        }
                    } else if let Ok(d) = bs58::decode(a).with_check(None).into_vec() {
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
        
        let loaded = h20.len() + h32.len();
        pb.finish_with_message(format!("✅ {} adres yüklendi", loaded));
        
        // Binary cache oluştur
        if loaded > 0 {
            let cache_pb = ProgressBar::new_spinner();
            cache_pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.magenta} {msg}")
                    .unwrap()
            );
            cache_pb.set_message(format!("{} cache oluşturuluyor...", name));
            cache_pb.enable_steady_tick(std::time::Duration::from_millis(100));
            
            if let Ok(cache) = File::create(&bin_path) {
                if bincode::serialize_into(cache, &(&h20, &h32)).is_ok() {
                    let cache_size = std::fs::metadata(&bin_path)
                        .map(|m| m.len() as f64 / 1_048_576.0)
                        .unwrap_or(0.0);
                    cache_pb.finish_with_message(format!("💾 {} cache oluşturuldu ({:.1} MB)", name, cache_size));
                }
            }
        }
        
        (h20, h32)
    }
}
