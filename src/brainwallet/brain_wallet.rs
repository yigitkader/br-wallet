use ripemd::Ripemd160;
use secp256k1::{PublicKey, SecretKey, SECP256K1};
use sha2::{Digest, Sha256};

pub struct BrainWallet {
    pub h160_compressed: [u8; 20],
    pub h160_uncompressed: [u8; 20],
}

impl BrainWallet {
    #[inline(always)]
    pub fn fast_generate(passphrase: &[u8]) -> Self {
        let priv_bytes = Sha256::digest(passphrase);

        // 2. Secret Key & Public Key (Global Context)
        let sk = SecretKey::from_slice(&priv_bytes).expect("32 bytes");
        let pk = PublicKey::from_secret_key(SECP256K1, &sk);

        // 3. Hash160 Türetme
        let h160_c = Self::derive_h160(&pk.serialize());
        let h160_u = Self::derive_h160(&pk.serialize_uncompressed());

        BrainWallet {
            h160_compressed: h160_c,
            h160_uncompressed: h160_u,
        }
    }

    #[inline(always)]
    fn derive_h160(pub_bytes: &[u8]) -> [u8; 20] {
        let sha = Sha256::digest(pub_bytes);
        let rip = Ripemd160::digest(sha);
        let mut out = [0u8; 20];
        out.copy_from_slice(&rip);
        out
    }
}
