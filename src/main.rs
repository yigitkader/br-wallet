mod brainwallet;
mod comparer;
mod reader;

#[cfg(feature = "gpu")]
mod metal;

fn main() {
    println!("--- Universal Blockchain Brainwallet Cracker v2.3 ---");
    println!("Ağlar: Bitcoin, Litecoin, Ethereum (GPU: secp256k1)\n");

    // Dictionary dosyası
    let dict_path = std::env::args().nth(1).unwrap_or_else(|| "weakpass_4.merged.txt".to_string());
    
    // Dosya varlığını kontrol et
    if !std::path::Path::new(&dict_path).exists() {
        eprintln!("❌ Hata: Dictionary dosyası bulunamadı: {}", dict_path);
        eprintln!("   Kullanım: brwallet <wordlist.txt>");
        return;
    }

    println!("🔍 Hedef adresler yükleniyor...");
    let comparer = comparer::Comparer::load();
    
    if !comparer.btc_on && !comparer.ltc_on && !comparer.eth_on {
        eprintln!("\n⚠️  Uyarı: Hiçbir hedef adres yüklenmedi!");
        eprintln!("   Aşağıdaki dosyaları oluşturun:");
        eprintln!("   - bitcoin_targets.json");
        eprintln!("   - litecoin_targets.json");
        eprintln!("   - ethereum_targets.json");
        eprintln!("\n   Format: {{\"addresses\": [\"addr1\", \"addr2\", ...]}}");
        return;
    }
    
    println!(
        "\n✅ Aktif ağlar: {}{}{}",
        if comparer.btc_on { "BTC " } else { "" },
        if comparer.ltc_on { "LTC " } else { "" },
        if comparer.eth_on { "ETH " } else { "" }
    );
    
    // Dictionary boyutunu göster
    if let Ok(meta) = std::fs::metadata(&dict_path) {
        let size_mb = meta.len() as f64 / 1_048_576.0;
        println!("📖 Dictionary: {} ({:.2} MB)", dict_path, size_mb);
    }
    
    println!("\n🚀 Tarama başlatılıyor...\n");
    
    reader::start_cracking(&dict_path, &comparer);
}
