use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Deserialize)]
struct TargetFile {
    addresses: Vec<String>,
}

/// Cache file format (raw bytes, no serialization overhead):
/// [8 bytes: h20_count][8 bytes: h32_count][h20_count * 20 bytes][h32_count * 32 bytes]
const CACHE_HEADER_SIZE: usize = 16;

pub struct Comparer {
    pub btc_20: FxHashSet<[u8; 20]>,
    pub btc_32: FxHashSet<[u8; 32]>,
    pub eth_20: FxHashSet<[u8; 20]>,
    pub sol_32: FxHashSet<[u8; 32]>,
    pub btc_on: bool,
    pub eth_on: bool,
    pub sol_on: bool,
}

impl Comparer {
    pub fn load() -> Self {
        let (btc_20, btc_32) = Self::load_net("bitcoin");
        let (eth_20, _) = Self::load_net("ethereum");
        let (_, sol_32) = Self::load_net("solana");

        let btc_count = btc_20.len() + btc_32.len();
        let eth_count = eth_20.len();
        let sol_count = sol_32.len();
        
        println!();
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

    /// Mmap ile cache'den yükle - O(n) yükleme, O(1) lookup, false positive YOK
    fn load_from_cache(bin_path: &str) -> Option<(FxHashSet<[u8; 20]>, FxHashSet<[u8; 32]>)> {
        let file = File::open(bin_path).ok()?;
        let mmap = unsafe { Mmap::map(&file).ok()? };
        
        if mmap.len() < CACHE_HEADER_SIZE {
            return None;
        }
        
        // Header oku
        let h20_count = u64::from_le_bytes(mmap[0..8].try_into().ok()?) as usize;
        let h32_count = u64::from_le_bytes(mmap[8..16].try_into().ok()?) as usize;
        
        let expected_size = CACHE_HEADER_SIZE + h20_count * 20 + h32_count * 32;
        if mmap.len() != expected_size {
            return None;
        }
        
        // Raw bytes'tan hash'leri oku
        let h20_start = CACHE_HEADER_SIZE;
        let h20_end = h20_start + h20_count * 20;
        let h32_end = h20_end + h32_count * 32;
        
        // Paralel FxHashSet oluşturma (Rayon) - 3-5x daha hızlı hash
        let h20: FxHashSet<[u8; 20]> = mmap[h20_start..h20_end]
            .par_chunks_exact(20)
            .map(|chunk| {
                let mut arr = [0u8; 20];
                arr.copy_from_slice(chunk);
                arr
            })
            .collect();
        
        let h32: FxHashSet<[u8; 32]> = mmap[h20_end..h32_end]
            .par_chunks_exact(32)
            .map(|chunk| {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(chunk);
                arr
            })
            .collect();
        
        Some((h20, h32))
    }
    
    /// Cache'e raw bytes olarak yaz
    fn save_to_cache(bin_path: &str, h20: &FxHashSet<[u8; 20]>, h32: &FxHashSet<[u8; 32]>) -> std::io::Result<()> {
        let file = File::create(bin_path)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
        
        // Header yaz
        writer.write_all(&(h20.len() as u64).to_le_bytes())?;
        writer.write_all(&(h32.len() as u64).to_le_bytes())?;
        
        // h20 hash'lerini yaz
        for hash in h20.iter() {
            writer.write_all(hash)?;
        }
        
        // h32 hash'lerini yaz
        for hash in h32.iter() {
            writer.write_all(hash)?;
        }
        
        writer.flush()?;
        Ok(())
    }

    fn load_net(name: &str) -> (FxHashSet<[u8; 20]>, FxHashSet<[u8; 32]>) {
        let json_path = format!("{}_targets.json", name);
        let bin_path = format!("{}_targets.bin", name);

        // Önce binary cache'i dene (mmap ile çok hızlı)
        if std::path::Path::new(&bin_path).exists() {
            let size_mb = std::fs::metadata(&bin_path)
                .map(|m| m.len() as f64 / 1_048_576.0)
                .unwrap_or(0.0);
            
            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}").unwrap());
            pb.set_message(format!("{} cache yükleniyor ({:.1} MB)...", name, size_mb));
            pb.enable_steady_tick(std::time::Duration::from_millis(80));
            
            if let Some((h20, h32)) = Self::load_from_cache(&bin_path) {
                let total = h20.len() + h32.len();
                pb.finish_with_message(format!("✅ {} cache yüklendi: {} adres ({:.1} MB)", name, total, size_mb));
                return (h20, h32);
            } else {
                pb.finish_with_message(format!("⚠️  {} cache bozuk, yeniden oluşturuluyor...", name));
                let _ = std::fs::remove_file(&bin_path);
            }
        }

        // JSON dosyası var mı?
        if !std::path::Path::new(&json_path).exists() {
            return (FxHashSet::default(), FxHashSet::default());
        }

        let size_mb = std::fs::metadata(&json_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);

        // JSON parse
        let parse_pb = ProgressBar::new_spinner();
        parse_pb.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.yellow} {msg}").unwrap());
        parse_pb.set_message(format!("{} JSON parse ediliyor ({:.1} MB)...", name, size_mb));
        parse_pb.enable_steady_tick(std::time::Duration::from_millis(80));
        
        let file = match File::open(&json_path) {
            Ok(f) => f,
            Err(_) => {
                parse_pb.finish_with_message(format!("❌ {} dosyası açılamadı", name));
                return (FxHashSet::default(), FxHashSet::default());
            }
        };
        
        let data: TargetFile = match serde_json::from_reader(BufReader::new(file)) {
            Ok(d) => {
                parse_pb.finish_and_clear();
                d
            }
            Err(e) => {
                parse_pb.finish_with_message(format!("❌ {} JSON hatası: {}", name, e));
                return (FxHashSet::default(), FxHashSet::default());
            }
        };
        
        let total = data.addresses.len();
        let mut h20: FxHashSet<[u8; 20]> = FxHashSet::default();
        h20.reserve(total);
        let mut h32: FxHashSet<[u8; 32]> = FxHashSet::default();
        
        // Adres işleme
        let pb = ProgressBar::new(total as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("   {spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▓░"));
        pb.set_message(format!("{} adresleri işleniyor", name));
        
        for (i, raw_addr) in data.addresses.into_iter().enumerate() {
            if i % 50_000 == 0 {
                pb.set_position(i as u64);
            }
            
            let a = raw_addr.trim();
            if a.is_empty() { continue; }
            
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
        
        // Cache oluştur (raw bytes, çok hızlı)
        if loaded > 0 {
            let cache_pb = ProgressBar::new_spinner();
            cache_pb.set_style(ProgressStyle::default_spinner()
                .template("{spinner:.magenta} {msg}").unwrap());
            cache_pb.set_message(format!("{} cache oluşturuluyor...", name));
            cache_pb.enable_steady_tick(std::time::Duration::from_millis(80));
            
            match Self::save_to_cache(&bin_path, &h20, &h32) {
                Ok(_) => {
                    let cache_size = std::fs::metadata(&bin_path)
                        .map(|m| m.len() as f64 / 1_048_576.0)
                        .unwrap_or(0.0);
                    cache_pb.finish_with_message(format!("💾 {} cache oluşturuldu ({:.1} MB)", name, cache_size));
                }
                Err(e) => {
                    cache_pb.finish_with_message(format!("⚠️  {} cache yazılamadı: {}", name, e));
                }
            }
        }
        
        (h20, h32)
    }
}
