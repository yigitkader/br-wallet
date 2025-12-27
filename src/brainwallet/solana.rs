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
        format!(
            "[SOL MATCH] Pass: {}\nAddr: {}\nPriv: {}\n",
            pass,
            bs58::encode(self.address).into_string(),
            bs58::encode(self.priv_bytes).into_string()
        )
    }
}
