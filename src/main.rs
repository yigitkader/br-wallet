mod brainwallet;
mod comparer;
mod reader;

fn main() {
    println!("--- Universal Blockchain Brainwallet Cracker v2.1 ---");
    println!("Ağlar: Bitcoin (Legacy, SegWit, Taproot), Ethereum, Solana\n");

    let comparer = comparer::Comparer::load();
    
    if !comparer.btc_on && !comparer.eth_on && !comparer.sol_on {
        eprintln!("⚠️  Uyarı: Hiçbir hedef adres yüklenmedi!");
        eprintln!("   bitcoin_targets.json, ethereum_targets.json veya solana_targets.json dosyalarını kontrol edin.");
        return;
    }
    
    println!(
        "Aktif ağlar: {}{}{}",
        if comparer.btc_on { "BTC " } else { "" },
        if comparer.eth_on { "ETH " } else { "" },
        if comparer.sol_on { "SOL " } else { "" }
    );
    
    reader::start_cracking("rockyou.txt", &comparer);
}
