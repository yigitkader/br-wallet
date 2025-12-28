//! GPU-Accelerated Brainwallet Processing - Apple Metal
//!
//! This module provides 30-300x speedup over CPU by running ALL cryptographic
//! operations on Apple Metal GPU:
//!
//! - SHA256 (passphrase → private key)
//! - secp256k1 scalar multiplication (private key → public key)
//! - RIPEMD160 (public key → Bitcoin/Litecoin HASH160)
//! - Keccak256 (public key → Ethereum address)
//!
//! ## Architecture
//!
//! ```text
//! CPU (minimal)                 GPU (all crypto)
//! ─────────────────            ─────────────────────────────
//! Passphrase batch   ───────>  SHA256 → secp256k1 → HASH160
//!                               ↓
//!                              Keccak256 (ETH)
//!                               ↓
//! HashSet.contains() <─────── [h160_c, h160_u, h160_nested,
//!                               taproot, eth_addr, priv_key]
//! ```
//!
//! ## Output per passphrase: 144 bytes
//!
//! | Field        | Size | Description                    |
//! |--------------|------|--------------------------------|
//! | h160_c       | 20   | HASH160(compressed pubkey)     |
//! | h160_u       | 20   | HASH160(uncompressed pubkey)   |
//! | h160_nested  | 20   | HASH160(P2SH-P2WPKH script)    |
//! | taproot      | 32   | Taproot x-only pubkey          |
//! | eth_addr     | 20   | Keccak256(pubkey)[12:32]       |
//! | priv_key     | 32   | SHA256(passphrase)             |

mod gpu;
pub mod batch;

// Re-export commonly used types for library consumers
pub use batch::{BatchProcessor, BrainwalletResult, PassphraseBatcher};
pub use batch::{RawBatchOutput, RawGpuResult};

// Re-export GPU types for advanced users who need direct GPU access
pub use gpu::GpuBrainwallet;

// Constants for buffer sizing
pub use gpu::{OUTPUT_SIZE, MAX_PASSPHRASE_LEN, PASSPHRASE_STRIDE};

/// Check if Metal GPU is available on this system
pub fn is_gpu_available() -> bool {
    metal::Device::system_default().is_some()
}
