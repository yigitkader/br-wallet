use ripemd::Ripemd160;
use secp256k1::{PublicKey, SecretKey, SECP256K1};
use sha2::{Digest, Sha256};

pub struct BtcWallet {
    pub priv_bytes: [u8; 32],
    pub h160_c: [u8; 20],      // Legacy & Native SegWit
    pub h160_u: [u8; 20],      // Legacy Uncomp
    pub h160_nested: [u8; 20], // Nested SegWit
    pub taproot: [u8; 32],     // Taproot
}

impl BtcWallet {
    #[inline(always)]
    pub fn generate(priv_bytes: [u8; 32]) -> Self {
        let sk = SecretKey::from_slice(&priv_bytes).unwrap();
        let pk = PublicKey::from_secret_key(SECP256K1, &sk);

        let pk_comp = pk.serialize();
        let h160_c = Self::hash160(&pk_comp);
        let h160_u = Self::hash160(&pk.serialize_uncompressed());

        // P2SH-P2WPKH (3...) türetme
        let mut script = [0u8; 22];
        script[0] = 0x00;
        script[1] = 0x14;
        script[2..22].copy_from_slice(&h160_c);
        let h160_nested = Self::hash160(&script);

        let mut taproot = [0u8; 32];
        taproot.copy_from_slice(&pk_comp[1..33]);

        BtcWallet {
            priv_bytes,
            h160_c,
            h160_u,
            h160_nested,
            taproot,
        }
    }

    pub fn get_report(&self, pass: &str) -> String {
        let mut wif_b = vec![0x80];
        wif_b.extend_from_slice(&self.priv_bytes);
        wif_b.push(0x01);
        let wif = bs58::encode(&wif_b).with_check().into_string();
        let hrp = bech32::Hrp::parse("bc").unwrap();

        format!(
            "[BTC MATCH] Pass: {}\nWIF: {}\nAddr: {}, {}, {}\n",
            pass,
            wif,
            self.to_b58(0x00, &self.h160_c),
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &self.h160_c).unwrap(),
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &self.taproot).unwrap()
        )
    }

    fn to_b58(&self, v: u8, h: &[u8; 20]) -> String {
        let mut p = vec![v];
        p.extend_from_slice(h);
        bs58::encode(&p).with_check().into_string()
    }

    fn hash160(data: &[u8]) -> [u8; 20] {
        Ripemd160::digest(Sha256::digest(data)).into()
    }
}
