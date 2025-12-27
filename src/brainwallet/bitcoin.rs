use ripemd::Ripemd160;
use secp256k1::{Keypair, PublicKey, SecretKey, XOnlyPublicKey, SECP256K1};
use sha2::{Digest, Sha256};

pub struct BtcWallet {
    pub priv_bytes: [u8; 32],
    pub h160_c: [u8; 20],       // Legacy P2PKH (compressed) & Native SegWit
    pub h160_u: [u8; 20],       // Legacy P2PKH (uncompressed)
    pub h160_nested: [u8; 20],  // Nested SegWit (P2SH-P2WPKH)
    pub taproot: [u8; 32],      // Taproot (BIP341 tweaked)
}

impl BtcWallet {
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

    /// BIP341 Taproot output key türetmesi (keypath-only, merkle root yok)
    #[inline(always)]
    fn compute_taproot_output_key(internal_key: &XOnlyPublicKey) -> Option<[u8; 32]> {
        // t = tagged_hash("TapTweak", P) - merkle root olmadan
        let tweak_hash = Self::tagged_hash("TapTweak", &internal_key.serialize());
        
        // Q = P + t*G (tweaked public key)
        let tweak = secp256k1::Scalar::from_be_bytes(tweak_hash).ok()?;
        let (tweaked_key, _parity) = internal_key.add_tweak(SECP256K1, &tweak).ok()?;
        
        Some(tweaked_key.serialize())
    }

    #[inline(always)]
    pub fn generate(priv_bytes: [u8; 32]) -> Option<Self> {
        // secp256k1 private key: 1 <= key < curve_order
        // SHA256 çıktısı nadiren geçersiz olabilir
        let sk = SecretKey::from_slice(&priv_bytes).ok()?;
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

        // BIP341 Taproot: Internal key'den tweaked output key türet
        let keypair = Keypair::from_secret_key(SECP256K1, &sk);
        let (internal_key, _parity) = XOnlyPublicKey::from_keypair(&keypair);
        let taproot = Self::compute_taproot_output_key(&internal_key)?;

        Some(BtcWallet {
            priv_bytes,
            h160_c,
            h160_u,
            h160_nested,
            taproot,
        })
    }

    pub fn get_report(&self, pass: &str) -> String {
        // WIF (Wallet Import Format) - compressed
        let mut wif_b = vec![0x80];
        wif_b.extend_from_slice(&self.priv_bytes);
        wif_b.push(0x01); // compressed flag
        let wif = bs58::encode(&wif_b).with_check().into_string();
        
        let hrp = bech32::Hrp::parse("bc").unwrap();

        format!(
            "[BTC MATCH] Pass: {}\n\
             WIF: {}\n\
             Legacy (1...):     {}\n\
             Legacy Uncomp:     {}\n\
             P2SH-SegWit (3...): {}\n\
             Native SegWit:     {}\n\
             Taproot:           {}\n",
            pass,
            wif,
            Self::to_b58_static(0x00, &self.h160_c),           // P2PKH (compressed)
            Self::to_b58_static(0x00, &self.h160_u),           // P2PKH (uncompressed)
            Self::to_b58_static(0x05, &self.h160_nested),      // P2SH-P2WPKH (version 0x05!)
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
