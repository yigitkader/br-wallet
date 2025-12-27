use ripemd::Ripemd160;
use secp256k1::{PublicKey, SecretKey, SECP256K1};
use sha2::{Digest, Sha256};

pub struct BrainWallet {
    pub h160_c: [u8; 20],      // Legacy (1...) ve Native SegWit (bc1q...) için
    pub h160_u: [u8; 20],      // Legacy Uncompressed (1...) için
    pub h160_nested: [u8; 20], // Nested SegWit (3...) için <--- YENİ
    pub taproot: [u8; 32],     // Taproot (bc1p...) için
}

impl BrainWallet {
    #[inline(always)]
    pub fn fast_generate(passphrase: &[u8]) -> Self {
        let priv_bytes = Sha256::digest(passphrase);
        let sk = SecretKey::from_slice(&priv_bytes).expect("32 bytes expected");
        let pk = PublicKey::from_secret_key(SECP256K1, &sk);

        let pk_comp = pk.serialize();
        let h160_c = Self::derive_h160(&pk_comp);
        let h160_u = Self::derive_h160(&pk.serialize_uncompressed());

        let mut script_sig = [0u8; 22];
        script_sig[0] = 0x00;
        script_sig[1] = 0x14;
        script_sig[2..22].copy_from_slice(&h160_c);
        let h160_nested = Self::derive_h160(&script_sig);

        let mut taproot = [0u8; 32];
        taproot.copy_from_slice(&pk_comp[1..33]);

        BrainWallet {
            h160_c,
            h160_u,
            h160_nested,
            taproot,
        }
    }

    #[inline(always)]
    fn derive_h160(data: &[u8]) -> [u8; 20] {
        let sha = Sha256::digest(data);
        let rip = Ripemd160::digest(sha);
        let mut out = [0u8; 20];
        out.copy_from_slice(&rip);
        out
    }
}
