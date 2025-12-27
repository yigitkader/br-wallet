use secp256k1::{PublicKey, SecretKey, SECP256K1};
use tiny_keccak::{Hasher, Keccak};

pub struct EthWallet {
    pub priv_bytes: [u8; 32],
    pub address: [u8; 20],
}

impl EthWallet {
    #[inline(always)]
    pub fn generate(priv_bytes: [u8; 32]) -> Option<Self> {
        // secp256k1 private key: 1 <= key < curve_order
        let sk = SecretKey::from_slice(&priv_bytes).ok()?;
        let pk = PublicKey::from_secret_key(SECP256K1, &sk);

        let mut k = Keccak::v256();
        k.update(&pk.serialize_uncompressed()[1..65]); // 0x04 prefix atılır
        let mut h = [0u8; 32];
        k.finalize(&mut h);

        let mut address = [0u8; 20];
        address.copy_from_slice(&h[12..32]); // Son 20 byte
        Some(EthWallet {
            priv_bytes,
            address,
        })
    }

    pub fn get_report(&self, pass: &str) -> String {
        format!(
            "[ETH MATCH] Pass: {}\nPriv: 0x{}\nAddr: 0x{}\n",
            pass,
            hex::encode(self.priv_bytes),
            hex::encode(self.address)
        )
    }
}
