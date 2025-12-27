mod brainwallet;
mod comparer;
mod reader;

fn main() {
    println!("--- Universal Universal Blockchain Cracker v2.0 ---");
    println!("Ağlar: Bitcoin (Legacy, SegWit, Taproot), Ethereum, Solana");

    let comparer = comparer::Comparer::load();
    reader::start_cracking("rockyou.txt", &comparer);
}
