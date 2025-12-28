use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::sync::Arc;

#[derive(Deserialize)]
struct TargetFile {
    addresses: Vec<String>,
}

const CACHE_HEADER_SIZE: usize = 8;
const LARGE_DATASET_THRESHOLD: usize = 200_000_000;

pub struct SortedMmapLookup {
    mmap: Arc<Mmap>,
    h20_start: usize,
    h20_count: usize,
}

impl SortedMmapLookup {
    #[inline]
    pub fn contains_h20(&self, hash: &[u8; 20]) -> bool {
        if self.h20_count == 0 {
            return false;
        }
        
        let data = &self.mmap[self.h20_start..self.h20_start + self.h20_count * 20];
        let mut left = 0;
        let mut right = self.h20_count;
        
        while left < right {
            let mid = left + (right - left) / 2;
            let offset = mid * 20;
            let entry = &data[offset..offset + 20];
            
            match entry.cmp(hash.as_slice()) {
                std::cmp::Ordering::Equal => return true,
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
            }
        }
        false
    }
    
    pub fn h20_count(&self) -> usize {
        self.h20_count
    }
}

pub enum HashStore20 {
    HashSet(FxHashSet<[u8; 20]>),
    SortedMmap(Arc<SortedMmapLookup>),
}

impl HashStore20 {
    #[inline]
    pub fn contains(&self, hash: &[u8; 20]) -> bool {
        match self {
            HashStore20::HashSet(set) => set.contains(hash),
            HashStore20::SortedMmap(lookup) => lookup.contains_h20(hash),
        }
    }
    
    pub fn len(&self) -> usize {
        match self {
            HashStore20::HashSet(set) => set.len(),
            HashStore20::SortedMmap(lookup) => lookup.h20_count(),
        }
    }
}

pub struct Comparer {
    pub btc_20: HashStore20,
    pub ltc_20: HashStore20,
    pub eth_20: HashStore20,
    pub btc_on: bool,
    pub ltc_on: bool,
    pub eth_on: bool,
}

impl Comparer {
    pub fn load() -> Self {
        let btc_20 = Self::load_net("bitcoin");
        let ltc_20 = Self::load_net("litecoin");
        let eth_20 = Self::load_net("ethereum");

        let btc_count = btc_20.len();
        let ltc_count = ltc_20.len();
        let eth_count = eth_20.len();
        
        println!();
        if btc_count > 0 {
            let mode = match &btc_20 {
                HashStore20::HashSet(_) => "HashSet",
                HashStore20::SortedMmap(_) => "mmap",
            };
            println!("📦 Bitcoin: {} addresses [{}]", btc_count, mode);
        }
        if ltc_count > 0 {
            let mode = match &ltc_20 {
                HashStore20::HashSet(_) => "HashSet",
                HashStore20::SortedMmap(_) => "mmap",
            };
            println!("📦 Litecoin: {} addresses [{}]", ltc_count, mode);
        }
        if eth_count > 0 {
            let mode = match &eth_20 {
                HashStore20::HashSet(_) => "HashSet",
                HashStore20::SortedMmap(_) => "mmap",
            };
            println!("📦 Ethereum: {} addresses [{}]", eth_count, mode);
        }

        Comparer {
            btc_on: btc_count > 0,
            ltc_on: ltc_count > 0,
            eth_on: eth_count > 0,
            btc_20,
            ltc_20,
            eth_20,
        }
    }

    fn load_from_cache(bin_path: &str) -> Option<HashStore20> {
        let file = File::open(bin_path).ok()?;
        let mmap = unsafe { Mmap::map(&file).ok()? };
        
        if mmap.len() < CACHE_HEADER_SIZE {
            return None;
        }
        
        let h20_count = u64::from_le_bytes(mmap[0..8].try_into().ok()?) as usize;
        let expected_size = CACHE_HEADER_SIZE + h20_count * 20;
        
        if mmap.len() != expected_size {
            return None;
        }
        
        if h20_count >= LARGE_DATASET_THRESHOLD {
            let lookup = Arc::new(SortedMmapLookup {
                mmap: Arc::new(mmap),
                h20_start: CACHE_HEADER_SIZE,
                h20_count,
            });
            return Some(HashStore20::SortedMmap(lookup));
        }
        
        let h20: FxHashSet<[u8; 20]> = mmap[CACHE_HEADER_SIZE..CACHE_HEADER_SIZE + h20_count * 20]
            .par_chunks_exact(20)
            .map(|chunk| {
                let mut arr = [0u8; 20];
                arr.copy_from_slice(chunk);
                arr
            })
            .collect();
        
        Some(HashStore20::HashSet(h20))
    }
    
    fn save_to_cache(bin_path: &str, h20: &FxHashSet<[u8; 20]>) -> std::io::Result<()> {
        let file = File::create(bin_path)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        
        writer.write_all(&(h20.len() as u64).to_le_bytes())?;
        
        let mut h20_sorted: Vec<[u8; 20]> = h20.iter().copied().collect();
        h20_sorted.par_sort_unstable();
        
        for hash in &h20_sorted {
            writer.write_all(hash)?;
        }
        
        writer.flush()?;
        Ok(())
    }

    fn load_net(name: &str) -> HashStore20 {
        let json_path = format!("{}_targets.json", name);
        let bin_path = format!("{}_targets.bin", name);

        if std::path::Path::new(&bin_path).exists() {
            let size_mb = std::fs::metadata(&bin_path)
                .map(|m| m.len() as f64 / 1_048_576.0)
                .unwrap_or(0.0);
            
            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}").unwrap());
            pb.set_message(format!("{} loading ({:.1} MB)...", name, size_mb));
            pb.enable_steady_tick(std::time::Duration::from_millis(80));
            
            if let Some(store) = Self::load_from_cache(&bin_path) {
                let mode = match &store {
                    HashStore20::HashSet(_) => "HashSet",
                    HashStore20::SortedMmap(_) => "zero-copy mmap",
                };
                pb.finish_with_message(format!("✅ {} loaded: {} addresses ({:.1} MB) [{}]", 
                    name, store.len(), size_mb, mode));
                return store;
            } else {
                pb.finish_with_message(format!("⚠️ {} cache invalid, rebuilding...", name));
                let _ = std::fs::remove_file(&bin_path);
            }
        }

        if !std::path::Path::new(&json_path).exists() {
            return HashStore20::HashSet(FxHashSet::default());
        }

        let size_mb = std::fs::metadata(&json_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);

        let parse_pb = ProgressBar::new_spinner();
        parse_pb.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.yellow} {msg}").unwrap());
        parse_pb.set_message(format!("{} parsing ({:.1} MB)...", name, size_mb));
        parse_pb.enable_steady_tick(std::time::Duration::from_millis(80));
        
        let file = match File::open(&json_path) {
            Ok(f) => f,
            Err(_) => {
                parse_pb.finish_with_message(format!("❌ {} file not found", name));
                return HashStore20::HashSet(FxHashSet::default());
            }
        };
        
        let data: TargetFile = match serde_json::from_reader(BufReader::new(file)) {
            Ok(d) => {
                parse_pb.finish_and_clear();
                d
            }
            Err(e) => {
                parse_pb.finish_with_message(format!("❌ {} JSON error: {}", name, e));
                return HashStore20::HashSet(FxHashSet::default());
            }
        };
        
        let total = data.addresses.len();
        let mut h20: FxHashSet<[u8; 20]> = FxHashSet::default();
        h20.reserve(total);
        
        let pb = ProgressBar::new(total as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("   {spinner:.green} [{bar:30.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("█▓░"));
        
        for (i, raw_addr) in data.addresses.into_iter().enumerate() {
            if i % 50_000 == 0 {
                pb.set_position(i as u64);
            }
            
            let a = raw_addr.trim();
            if a.is_empty() { continue; }
            
            match name {
                "bitcoin" => {
                    if a.starts_with("bc1q") {
                        if let Ok((_, _, p)) = bech32::segwit::decode(a) {
                            if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                                h20.insert(arr);
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
                "litecoin" => {
                    if a.starts_with("ltc1q") {
                        if let Ok((_, _, p)) = bech32::segwit::decode(a) {
                            if let Ok(arr) = <[u8; 20]>::try_from(p.as_slice()) {
                                h20.insert(arr);
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
                _ => {}
            }
        }
        
        let loaded = h20.len();
        pb.finish_with_message(format!("✅ {} addresses loaded", loaded));
        
        if loaded > 0 {
            let cache_pb = ProgressBar::new_spinner();
            cache_pb.set_style(ProgressStyle::default_spinner()
                .template("{spinner:.magenta} {msg}").unwrap());
            cache_pb.set_message(format!("{} caching...", name));
            cache_pb.enable_steady_tick(std::time::Duration::from_millis(80));
            
            match Self::save_to_cache(&bin_path, &h20) {
                Ok(_) => {
                    let cache_size = std::fs::metadata(&bin_path)
                        .map(|m| m.len() as f64 / 1_048_576.0)
                        .unwrap_or(0.0);
                    cache_pb.finish_with_message(format!("💾 {} cached ({:.1} MB)", name, cache_size));
                    
                    if loaded >= LARGE_DATASET_THRESHOLD {
                        if let Some(store) = Self::load_from_cache(&bin_path) {
                            return store;
                        }
                    }
                }
                Err(e) => {
                    cache_pb.finish_with_message(format!("⚠️ {} cache error: {}", name, e));
                }
            }
        }
        
        HashStore20::HashSet(h20)
    }
}
