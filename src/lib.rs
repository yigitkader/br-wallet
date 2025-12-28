//! Brainwallet Cracker Library
//!
//! High-performance brainwallet cracking for Bitcoin, Litecoin, Ethereum, and Solana.

pub mod brainwallet;
pub mod comparer;

#[cfg(feature = "gpu")]
pub mod metal;

