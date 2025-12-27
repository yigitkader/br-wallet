pub mod bitcoin;
pub mod ethereum;
pub mod solana;

use sha2::{Digest, Sha256};

pub struct MultiWallet {
    pub btc: Option<bitcoin::BtcWallet>,
    pub eth: Option<ethereum::EthWallet>,
    pub sol: Option<solana::SolWallet>,
}

impl MultiWallet {
    #[inline(always)]
    pub fn generate_active(passphrase: &[u8], btc_on: bool, eth_on: bool, sol_on: bool) -> Self {
        // SHA256 işlemini sadece bir kez yapıyoruz (Optimizasyon)
        let priv_hash = Sha256::digest(passphrase);
        let mut priv_bytes = [0u8; 32];
        priv_bytes.copy_from_slice(&priv_hash);

        Self {
            btc: if btc_on {
                Some(bitcoin::BtcWallet::generate(priv_bytes))
            } else {
                None
            },
            eth: if eth_on {
                Some(ethereum::EthWallet::generate(priv_bytes))
            } else {
                None
            },
            sol: if sol_on {
                Some(solana::SolWallet::generate(priv_bytes))
            } else {
                None
            },
        }
    }
}
