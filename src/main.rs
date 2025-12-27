mod brainwallet;
mod comparer;
mod reader;

fn main() {
    let json_file = "targets.json";
    let bin_cache = "targets.bin";
    let comparer = comparer::Comparer::load(json_file, bin_cache);
    println!("Starting the high-speed dictionary scan...");
    reader::start_cracking("weakpass_4.txt", &comparer);
}
