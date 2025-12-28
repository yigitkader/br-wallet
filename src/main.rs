//! Universal Blockchain Brainwallet Cracker - GPU Accelerated
//!
//! This tool scans wordlists for brainwallet private keys that match target addresses.
//! ALL cryptographic operations run on Apple Metal GPU for maximum performance.
//!
//! Supported chains:
//! - Bitcoin (BTC): All address types (Legacy, SegWit, Taproot)
//! - Litecoin (LTC): All address types
//! - Ethereum (ETH): Keccak256-based addresses
//!
//! ⚠️ REQUIRES: macOS with Apple Metal GPU support

mod comparer;
mod metal;
mod reader;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   Universal Brainwallet Cracker v3.0 - GPU Accelerated     ║");
    println!("║   Chains: Bitcoin, Litecoin, Ethereum                      ║");
    println!("║   Engine: Apple Metal (ALL crypto on GPU)                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Verify GPU availability at startup
    if !metal::is_gpu_available() {
        eprintln!("❌ FATAL: No Metal GPU detected!");
        eprintln!("   This application requires Apple Metal GPU.");
        eprintln!("   Supported: macOS 10.14+ with Metal-capable GPU");
        std::process::exit(1);
    }

    // Dictionary dosyası
    let dict_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weakpass_4.merged.txt".to_string());

    // Dosya varlığını kontrol et
    if !std::path::Path::new(&dict_path).exists() {
        eprintln!("❌ Error: Dictionary file not found: {}", dict_path);
        eprintln!("   Usage: brwallet <wordlist.txt>");
        std::process::exit(1);
    }

    println!("🔍 Loading target addresses...");
    let comparer = comparer::Comparer::load();

    if !comparer.btc_on && !comparer.ltc_on && !comparer.eth_on {
        eprintln!("\n⚠️  Warning: No target addresses loaded!");
        eprintln!("   Create one or more of these files:");
        eprintln!("   - bitcoin_targets.json");
        eprintln!("   - litecoin_targets.json");
        eprintln!("   - ethereum_targets.json");
        eprintln!("\n   Format: {{\"addresses\": [\"addr1\", \"addr2\", ...]}}");
        std::process::exit(1);
    }

    println!(
        "\n✅ Active chains: {}{}{}",
        if comparer.btc_on { "BTC " } else { "" },
        if comparer.ltc_on { "LTC " } else { "" },
        if comparer.eth_on { "ETH " } else { "" }
    );

    // Dictionary boyutunu göster
    if let Ok(meta) = std::fs::metadata(&dict_path) {
        let size_mb = meta.len() as f64 / 1_048_576.0;
        println!("📖 Dictionary: {} ({:.2} MB)", dict_path, size_mb);
    }

    println!("\n🚀 Starting GPU-accelerated scan...\n");

    reader::start_cracking(&dict_path, &comparer);
}
