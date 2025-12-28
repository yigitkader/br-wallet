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
//! All cryptographic operations computed on Metal GPU:
//! - SHA256 (passphrase → private key)
//! - secp256k1 scalar multiplication (private key → public key)
//! - RIPEMD160 (public key → HASH160)
//! - Keccak256 (public key → Ethereum address)
//!
//! ## Performance
//!
//! - 30-300x faster than CPU-only implementations
//! - Processes 100K+ passphrases per second on Apple Silicon
//!
//! ## Requirements
//!
//! - macOS with Metal GPU support (Apple Silicon or discrete GPU)

pub mod comparer;
pub mod metal;
