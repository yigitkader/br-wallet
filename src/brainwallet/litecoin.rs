use ripemd::Ripemd160;
use secp256k1::{Keypair, PublicKey, SecretKey, XOnlyPublicKey, SECP256K1};
use sha2::{Digest, Sha256};

/// Litecoin Wallet
/// Version bytes: P2PKH=0x30 (L...), P2SH=0x32 (M...), WIF=0xB0
/// Bech32 HRP: "ltc" (ltc1q... SegWit, ltc1p... Taproot)
pub struct LtcWallet {
    pub priv_bytes: [u8; 32],
    pub h160_c: [u8; 20],       // Legacy P2PKH (compressed) & Native SegWit
    pub h160_u: [u8; 20],       // Legacy P2PKH (uncompressed)
    pub h160_nested: [u8; 20],  // Nested SegWit (P2SH-P2WPKH)
    pub taproot: [u8; 32],      // Taproot (MWEB sonrası)
}

impl LtcWallet {
    /// BIP340/341 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || data)
    #[inline(always)]
    fn tagged_hash(tag: &str, data: &[u8]) -> [u8; 32] {
        let tag_hash = Sha256::digest(tag.as_bytes());
        let mut engine = Sha256::new();
        engine.update(&tag_hash);
        engine.update(&tag_hash);
        engine.update(data);
        engine.finalize().into()
    }

    /// Taproot output key türetmesi
    #[inline(always)]
    fn compute_taproot_output_key(internal_key: &XOnlyPublicKey) -> Option<[u8; 32]> {
        let tweak_hash = Self::tagged_hash("TapTweak", &internal_key.serialize());
        let tweak = secp256k1::Scalar::from_be_bytes(tweak_hash).ok()?;
        let (tweaked_key, _parity) = internal_key.add_tweak(SECP256K1, &tweak).ok()?;
        Some(tweaked_key.serialize())
    }

    #[inline(always)]
    pub fn generate(priv_bytes: [u8; 32]) -> Option<Self> {
        let sk = SecretKey::from_slice(&priv_bytes).ok()?;
        let pk = PublicKey::from_secret_key(SECP256K1, &sk);

        let pk_comp = pk.serialize();
        let h160_c = Self::hash160(&pk_comp);
        let h160_u = Self::hash160(&pk.serialize_uncompressed());

        // P2SH-P2WPKH (M...) türetme
        let mut script = [0u8; 22];
        script[0] = 0x00;
        script[1] = 0x14;
        script[2..22].copy_from_slice(&h160_c);
        let h160_nested = Self::hash160(&script);

        // Taproot
        let keypair = Keypair::from_secret_key(SECP256K1, &sk);
        let (internal_key, _parity) = XOnlyPublicKey::from_keypair(&keypair);
        let taproot = Self::compute_taproot_output_key(&internal_key)?;

        Some(LtcWallet {
            priv_bytes,
            h160_c,
            h160_u,
            h160_nested,
            taproot,
        })
    }

    pub fn get_report(&self, pass: &str) -> String {
        // WIF - Litecoin version: 0xB0
        let mut wif_b = vec![0xB0];
        wif_b.extend_from_slice(&self.priv_bytes);
        wif_b.push(0x01); // compressed flag
        let wif = bs58::encode(&wif_b).with_check().into_string();
        
        let hrp = bech32::Hrp::parse("ltc").unwrap();

        format!(
            "[LTC MATCH] Pass: {}\n\
             WIF: {}\n\
             Legacy (L...):     {}\n\
             Legacy Uncomp:     {}\n\
             P2SH-SegWit (M...): {}\n\
             Native SegWit:     {}\n\
             Taproot:           {}\n",
            pass,
            wif,
            Self::to_b58_static(0x30, &self.h160_c),           // P2PKH (compressed)
            Self::to_b58_static(0x30, &self.h160_u),           // P2PKH (uncompressed)
            Self::to_b58_static(0x32, &self.h160_nested),      // P2SH-P2WPKH
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &self.h160_c).unwrap(),
            bech32::segwit::encode(hrp, bech32::segwit::VERSION_1, &self.taproot).unwrap()
        )
    }

    fn to_b58_static(version: u8, hash: &[u8; 20]) -> String {
        let mut payload = vec![version];
        payload.extend_from_slice(hash);
        bs58::encode(&payload).with_check().into_string()
    }

    fn hash160(data: &[u8]) -> [u8; 20] {
        Ripemd160::digest(Sha256::digest(data)).into()
    }
}

