//! Brainwallet Cracker Library - GPU Accelerated
//!
//! High-performance brainwallet cracking for Bitcoin, Litecoin, and Ethereum.
//! ALL cryptographic operations run on Apple Metal GPU.
//!
//! ## Supported Chains
//!
//! - **Bitcoin**: P2PKH (1...), P2SH-P2WPKH (3...), Native SegWit (bc1q...), Taproot (bc1p...)
//! - **Litecoin**: Same address types with LTC prefixes
//! - **Ethereum**: Keccak256-based addresses (0x...)
//!
//! ## GPU Operations
//!
//! All operations computed on Metal GPU:
//! - SHA256 (passphrase → private key)
//! - secp256k1 scalar multiplication
//! - RIPEMD160 (HASH160)
//! - Keccak256 (Ethereum)
//!
//! ## Performance
//!
//! - 30-300x faster than CPU-only implementations
//! - Processes 100K+ passphrases per second on Apple Silicon

pub mod comparer;
pub mod metal;

// CPU brainwallet module kept for reference/testing only
// Not used in main application path
#[doc(hidden)]
pub mod brainwallet;
