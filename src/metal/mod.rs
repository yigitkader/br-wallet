//! GPU-accelerated brainwallet cracking using Metal
//!
//! This module provides 10-100x speedup over CPU-only processing by:
//! - Batch processing 65K+ passphrases per GPU dispatch
//! - Parallel SHA256 + secp256k1 + RIPEMD160 on GPU
//! - Zero-copy data transfer via unified memory (Apple Silicon)
//!
//! ## Architecture
//!
//! ```text
//! CPU                          GPU
//! ──────────────────          ──────────────────
//! Passphrase batch   ───────> SHA256 (parallel)
//!                              ↓
//!                             secp256k1 point mul
//!                              ↓
//!                             HASH160
//!                              ↓
//! HashSet.contains() <─────── [h160_c: 20 bytes] × 65K
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use brwallet::metal::GpuBrainwallet;
//!
//! let gpu = GpuBrainwallet::new()?;
//! let passphrases: Vec<&[u8]> = vec![b"password1", b"password2"];
//! let results = gpu.process_batch(&passphrases)?;
//! // results: Vec<BrainwalletResult> with h160_c, h160_u, h160_nested, taproot
//! ```

#[cfg(feature = "gpu")]
mod gpu;

#[cfg(feature = "gpu")]
pub mod batch;

#[cfg(feature = "gpu")]
pub use gpu::GpuBrainwallet;

#[cfg(feature = "gpu")]
pub use batch::{BatchProcessor, BrainwalletResult, PassphraseBatcher};

/// Check if GPU is available on this system
#[cfg(feature = "gpu")]
pub fn is_gpu_available() -> bool {
    metal::Device::system_default().is_some()
}

#[cfg(not(feature = "gpu"))]
pub fn is_gpu_available() -> bool {
    false
}

