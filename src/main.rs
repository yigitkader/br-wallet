mod brainwallet;
mod comparer;
mod reader;

fn main() {
    let comparer = comparer::Comparer::load_from_json("targets.json");
    println!("Loaded targets. Starting scan...");

    reader::start_cracking("rockyou.txt", &comparer);
}
