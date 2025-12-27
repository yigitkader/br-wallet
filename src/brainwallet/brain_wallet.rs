use sha2::{Sha256, Digest};
use ripemd::Ripemd160;
use secp256k1::{PublicKey, SecretKey, SECP256K1};

pub struct BrainWallet {
    pub priv_bytes: [u8; 32],
    pub h160_c: [u8; 20],
    pub h160_u: [u8; 20],
    pub h160_nested: [u8; 20],
    pub taproot: [u8; 32],
}

impl BrainWallet {
    #[inline(always)]
    pub fn fast_generate(passphrase: &[u8]) -> Self {
        let priv_bytes_hash = Sha256::digest(passphrase);
        let mut priv_bytes = [0u8; 32];
        priv_bytes.copy_from_slice(&priv_bytes_hash);

        let sk = SecretKey::from_slice(&priv_bytes).expect("32 bytes");
        let pk = PublicKey::from_secret_key(SECP256K1, &sk);

        let pk_comp = pk.serialize();
        let h160_c = Self::derive_h160(&pk_comp);
        let h160_u = Self::derive_h160(&pk.serialize_uncompressed());

        // Nested SegWit (3...)
        let mut script_sig = [0u8; 22];
        script_sig[0] = 0x00; script_sig[1] = 0x14;
        script_sig[2..22].copy_from_slice(&h160_c);
        let h160_nested = Self::derive_h160(&script_sig);

        // Taproot (bc1p) - X-only
        let mut taproot = [0u8; 32];
        taproot.copy_from_slice(&pk_comp[1..33]);

        BrainWallet { priv_bytes, h160_c, h160_u, h160_nested, taproot }
    }

    pub fn get_report(&self, pass: &str) -> String {
        let priv_hex = hex::encode(self.priv_bytes);
        let mut wif_bytes = vec![0x80];
        wif_bytes.extend_from_slice(&self.priv_bytes);
        wif_bytes.push(0x01);
        let wif_comp = bs58::encode(&wif_bytes).with_check().into_string();

        let hrp = bech32::Hrp::parse("bc").unwrap(); // Hrp tipini oluştur

        // bech32 v0.11 encode standartları
        let addr_segwit = bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &self.h160_c).unwrap();
        let addr_taproot = bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &self.taproot).unwrap();

        format!(
            "------------------------------------------\n\
             !!! MATCH FOUND !!!\n\
             Passphrase: {}\n\
             Private Key (Hex): {}\n\
             Private Key (WIF): {}\n\
             Legacy (Comp):    {}\n\
             Legacy (Uncomp):  {}\n\
             Nested SegWit:    {}\n\
             Native SegWit:    {}\n\
             Taproot (bc1p):   {}\n\
             ------------------------------------------\n",
            pass, priv_hex, wif_comp,
            self.to_base58(0x00, &self.h160_c),
            self.to_base58(0x00, &self.h160_u),
            self.to_base58(0x05, &self.h160_nested),
            addr_segwit, addr_taproot
        )
    }

    fn to_base58(&self, version: u8, hash: &[u8; 20]) -> String {
        let mut payload = vec![version];
        payload.extend_from_slice(hash);
        bs58::encode(&payload).with_check().into_string()
    }

    fn derive_h160(data: &[u8]) -> [u8; 20] {
        let sha = Sha256::digest(data);
        let rip = Ripemd160::digest(sha);
        let mut out = [0u8; 20];
        out.copy_from_slice(&rip);
        out
    }
}