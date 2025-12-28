//! CPU Reference Implementation for Brainwallet Generation
//!
//! ⚠️ THIS MODULE IS NOT USED IN PRODUCTION!
//! The main application uses GPU (Metal) for ALL cryptographic operations.
//!
//! This module is kept for:
//! - Reference/documentation of the algorithm
//! - Unit testing and verification
//! - Debugging GPU output against known-correct CPU results

pub mod bitcoin;
pub mod ethereum;
pub mod litecoin;

// Re-export wallet types for testing
pub use bitcoin::BtcWallet;
pub use ethereum::EthWallet;
pub use litecoin::LtcWallet;

use sha2::{Digest, Sha256};

/// CPU-based multi-chain wallet generator (for testing/reference only)
///
/// ⚠️ NOT USED IN PRODUCTION - GPU implementation is used instead
#[allow(dead_code)]
pub struct MultiWallet {
    pub btc: Option<bitcoin::BtcWallet>,
    pub ltc: Option<litecoin::LtcWallet>,
    pub eth: Option<ethereum::EthWallet>,
}

#[allow(dead_code)]
impl MultiWallet {
    /// Generate wallets for active chains (CPU reference implementation)
    #[inline(always)]
    pub fn generate_active(
        passphrase: &[u8],
        btc_on: bool,
        ltc_on: bool,
        eth_on: bool,
    ) -> Self {
        // SHA256 - only computed once
        let priv_hash = Sha256::digest(passphrase);
        let mut priv_bytes = [0u8; 32];
        priv_bytes.copy_from_slice(&priv_hash);

        Self {
            btc: if btc_on {
                bitcoin::BtcWallet::generate(priv_bytes)
            } else {
                None
            },
            ltc: if ltc_on {
                litecoin::LtcWallet::generate(priv_bytes)
            } else {
                None
            },
            eth: if eth_on {
                ethereum::EthWallet::generate(priv_bytes)
            } else {
                None
            },
        }
    }
}
