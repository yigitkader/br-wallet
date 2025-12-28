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

/// Cache file format (raw bytes, no serialization overhead):
/// [8 bytes: h20_count][8 bytes: h32_count][h20_count * 20 bytes][h32_count * 32 bytes]
/// Data is stored SORTED for binary search compatibility
const CACHE_HEADER_SIZE: usize = 16;

/// Threshold for switching from HashSet to sorted binary search
/// Above this count, binary search saves significant RAM
/// 
/// RAM comparison at 1M addresses:
/// - HashSet: 1M * (20 bytes data + ~40 bytes overhead) = ~60MB
/// - Sorted mmap: 1M * 20 bytes = 20MB (zero-copy from disk)
/// 
/// Trade-off: O(1) lookup (HashSet) vs O(log n) lookup (binary search)
/// At 1M entries, binary search is ~20 comparisons which is still fast
const LARGE_DATASET_THRESHOLD: usize = 1_000_000;

/// Memory-efficient sorted lookup for large datasets
/// Uses mmap directly with binary search - zero additional RAM allocation
pub struct SortedMmapLookup {
    mmap: Arc<Mmap>,
    h20_start: usize,
    h20_count: usize,
    h32_start: usize,
    h32_count: usize,
}

impl SortedMmapLookup {
    /// Binary search on sorted 20-byte hashes in mmap
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
    
    /// Binary search on sorted 32-byte hashes in mmap
    #[inline]
    pub fn contains_h32(&self, hash: &[u8; 32]) -> bool {
        if self.h32_count == 0 {
            return false;
        }
        
        let data = &self.mmap[self.h32_start..self.h32_start + self.h32_count * 32];
        
        let mut left = 0;
        let mut right = self.h32_count;
        
        while left < right {
            let mid = left + (right - left) / 2;
            let offset = mid * 32;
            let entry = &data[offset..offset + 32];
            
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
    
    pub fn h32_count(&self) -> usize {
        self.h32_count
    }
}

/// Hash storage that automatically chooses efficient backing store
pub enum HashStore20 {
    /// O(1) lookup, higher RAM usage - for small datasets
    HashSet(FxHashSet<[u8; 20]>),
    /// O(log n) lookup, zero RAM overhead - for large datasets
    SortedMmap(Arc<SortedMmapLookup>),
}

pub enum HashStore32 {
    HashSet(FxHashSet<[u8; 32]>),
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

impl HashStore32 {
    #[inline]
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        match self {
            HashStore32::HashSet(set) => set.contains(hash),
            HashStore32::SortedMmap(lookup) => lookup.contains_h32(hash),
        }
    }
    
    pub fn len(&self) -> usize {
        match self {
            HashStore32::HashSet(set) => set.len(),
            HashStore32::SortedMmap(lookup) => lookup.h32_count(),
        }
    }
}

pub struct Comparer {
    pub btc_20: HashStore20,
    pub btc_32: HashStore32,
    pub ltc_20: HashStore20,
    pub ltc_32: HashStore32,
    pub eth_20: HashStore20,
    pub btc_on: bool,
    pub ltc_on: bool,
    pub eth_on: bool,
}

impl Comparer {
    pub fn load() -> Self {
        let (btc_20, btc_32) = Self::load_net("bitcoin");
        let (ltc_20, ltc_32) = Self::load_net("litecoin");
        let (eth_20, eth_32) = Self::load_net("ethereum");

        let btc_count = btc_20.len() + btc_32.len();
        let ltc_count = ltc_20.len() + ltc_32.len();
        let eth_count = eth_20.len();
        
        println!();
        if btc_count > 0 {
            let mode = match &btc_20 {
                HashStore20::HashSet(_) => "HashSet",
                HashStore20::SortedMmap(_) => "mmap-binary-search",
            };
            println!("📦 Bitcoin: {} adres (Legacy/SegWit: {}, Taproot: {}) [{}]", 
                     btc_count, btc_20.len(), btc_32.len(), mode);
        }
        if ltc_count > 0 {
            let mode = match &ltc_20 {
                HashStore20::HashSet(_) => "HashSet",
                HashStore20::SortedMmap(_) => "mmap-binary-search",
            };
            println!("📦 Litecoin: {} adres (Legacy/SegWit: {}, Taproot: {}) [{}]", 
                     ltc_count, ltc_20.len(), ltc_32.len(), mode);
        }
        if eth_count > 0 {
            let mode = match &eth_20 {
                HashStore20::HashSet(_) => "HashSet",
                HashStore20::SortedMmap(_) => "mmap-binary-search",
            };
            println!("📦 Ethereum: {} adres [{}]", eth_count, mode);
        }

        // Ethereum doesn't use h32 (no Taproot equivalent)
        let _ = eth_32;

        Comparer {
            btc_on: btc_count > 0,
            ltc_on: ltc_count > 0,
            eth_on: eth_count > 0,
            btc_20,
            btc_32,
            ltc_20,
            ltc_32,
            eth_20,
        }
    }

    /// Load from cache with automatic memory optimization
    /// 
    /// For small datasets (< 5M entries): Uses HashSet for O(1) lookups
    /// For large datasets (>= 5M entries): Uses mmap binary search for ~0 RAM overhead
    /// 
    /// Binary cache format: data is stored SORTED for binary search compatibility
    fn load_from_cache(bin_path: &str) -> Option<(HashStore20, HashStore32)> {
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
        
        let h20_start = CACHE_HEADER_SIZE;
        let h20_end = h20_start + h20_count * 20;
        
        let total_count = h20_count + h32_count;
        
        // For large datasets, use zero-copy mmap binary search
        if total_count >= LARGE_DATASET_THRESHOLD {
            let lookup = Arc::new(SortedMmapLookup {
                mmap: Arc::new(mmap),
                h20_start,
                h20_count,
                h32_start: h20_end,
                h32_count,
            });
            
            return Some((
                HashStore20::SortedMmap(lookup.clone()),
                HashStore32::SortedMmap(lookup),
            ));
        }
        
        // For small datasets, use HashSet for O(1) lookups
        let h20: FxHashSet<[u8; 20]> = mmap[h20_start..h20_end]
            .par_chunks_exact(20)
            .map(|chunk| {
                let mut arr = [0u8; 20];
                arr.copy_from_slice(chunk);
                arr
            })
            .collect();
        
        let h32: FxHashSet<[u8; 32]> = mmap[h20_end..h20_end + h32_count * 32]
            .par_chunks_exact(32)
            .map(|chunk| {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(chunk);
                arr
            })
            .collect();
        
        Some((HashStore20::HashSet(h20), HashStore32::HashSet(h32)))
    }
    
    /// Cache'e raw bytes olarak yaz - SORTED for binary search compatibility
    fn save_to_cache(bin_path: &str, h20: &FxHashSet<[u8; 20]>, h32: &FxHashSet<[u8; 32]>) -> std::io::Result<()> {
        let file = File::create(bin_path)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
        
        // Header yaz
        writer.write_all(&(h20.len() as u64).to_le_bytes())?;
        writer.write_all(&(h32.len() as u64).to_le_bytes())?;
        
        // Sort h20 hashes for binary search compatibility
        let mut h20_sorted: Vec<[u8; 20]> = h20.iter().copied().collect();
        h20_sorted.par_sort_unstable();
        
        for hash in &h20_sorted {
            writer.write_all(hash)?;
        }
        
        // Sort h32 hashes for binary search compatibility
        let mut h32_sorted: Vec<[u8; 32]> = h32.iter().copied().collect();
        h32_sorted.par_sort_unstable();
        
        for hash in &h32_sorted {
            writer.write_all(hash)?;
        }
        
        writer.flush()?;
        Ok(())
    }

    fn load_net(name: &str) -> (HashStore20, HashStore32) {
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
                let mode = match &h20 {
                    HashStore20::HashSet(_) => "HashSet",
                    HashStore20::SortedMmap(_) => "zero-copy mmap",
                };
                pb.finish_with_message(format!("✅ {} cache yüklendi: {} adres ({:.1} MB) [{}]", 
                    name, total, size_mb, mode));
                return (h20, h32);
            } else {
                pb.finish_with_message(format!("⚠️  {} cache bozuk, yeniden oluşturuluyor...", name));
                let _ = std::fs::remove_file(&bin_path);
            }
        }

        // JSON dosyası var mı?
        if !std::path::Path::new(&json_path).exists() {
            return (
                HashStore20::HashSet(FxHashSet::default()), 
                HashStore32::HashSet(FxHashSet::default())
            );
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
                return (
                    HashStore20::HashSet(FxHashSet::default()), 
                    HashStore32::HashSet(FxHashSet::default())
                );
            }
        };
        
        let data: TargetFile = match serde_json::from_reader(BufReader::new(file)) {
            Ok(d) => {
                parse_pb.finish_and_clear();
                d
            }
            Err(e) => {
                parse_pb.finish_with_message(format!("❌ {} JSON hatası: {}", name, e));
                return (
                    HashStore20::HashSet(FxHashSet::default()), 
                    HashStore32::HashSet(FxHashSet::default())
                );
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
                "litecoin" => {
                    // Litecoin: ltc1q... (SegWit), ltc1p... (Taproot), L.../M... (Legacy/P2SH)
                    if a.starts_with("ltc1") {
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
                _ => {}
            }
        }
        
        let loaded = h20.len() + h32.len();
        pb.finish_with_message(format!("✅ {} adres yüklendi", loaded));
        
        // Cache oluştur (sorted for binary search)
        if loaded > 0 {
            let cache_pb = ProgressBar::new_spinner();
            cache_pb.set_style(ProgressStyle::default_spinner()
                .template("{spinner:.magenta} {msg}").unwrap());
            cache_pb.set_message(format!("{} cache oluşturuluyor (sorted)...", name));
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
        
        (HashStore20::HashSet(h20), HashStore32::HashSet(h32))
    }
}
