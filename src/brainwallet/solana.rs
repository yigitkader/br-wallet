use ed25519_dalek::SigningKey;

pub struct SolWallet {
    pub priv_bytes: [u8; 32],
    pub address: [u8; 32],
}

impl SolWallet {
    #[inline(always)]
    pub fn generate(priv_bytes: [u8; 32]) -> Self {
        let sk = SigningKey::from_bytes(&priv_bytes);
        let address = sk.verifying_key().to_bytes(); // Pubkey = Adres
        SolWallet {
            priv_bytes,
            address,
        }
    }

    pub fn get_report(&self, pass: &str) -> String {
        // Solana keypair = seed (32) + pubkey (32) = 64 bytes
        let mut keypair = [0u8; 64];
        keypair[..32].copy_from_slice(&self.priv_bytes);
        keypair[32..].copy_from_slice(&self.address);
        
        format!(
            "[SOL MATCH] Pass: {}\nAddr: {}\nKeypair: {}\n",
            pass,
            bs58::encode(self.address).into_string(),
            bs58::encode(keypair).into_string()
        )
    }
}
