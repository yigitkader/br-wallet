use crate::comparer::Comparer;

mod brainwallet;
mod comparer;
mod reader;

fn main() {
    println!("Target addresses loading...");
    let comparer = Comparer::load_from_file("targets.txt");

    println!("Dictionary file getting read...");
    reader::start_cracking("rockyou.txt", &comparer);
}
