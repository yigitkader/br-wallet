use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{PublicKey, SecretKey};
use ripemd::Ripemd160;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BrainWallet {
    pub passphrase: String,
    pub generated_wallet: GeneratedWallet,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeneratedWallet {
    pub secret_exponent: String,
    pub compressed_point_conversion: CompressedPointConversion,
    pub uncompressed_point_conversion: UncompressedPointConversion,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompressedPointConversion {
    pub address: String,
    pub asn1key: String,
    pub public_key: String,
    pub hash160: String,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UncompressedPointConversion {
    pub address: String,
    pub asn1key: String,
    pub public_key: String,
    pub hash160: String,
}

impl BrainWallet {
    pub fn new(passphrase: &str) -> BrainWallet {
        let priv_bytes = Sha256::digest(passphrase.as_bytes());
        let secret_exponent = hex::encode(priv_bytes);

        let secret_key = SecretKey::from_slice(&priv_bytes).expect("32 bytes expected");
        let public_key = secret_key.public_key();

        BrainWallet {
            passphrase: passphrase.to_string(),
            generated_wallet: GeneratedWallet {
                secret_exponent,
                compressed_point_conversion: Self::derive_compressed(&priv_bytes, &public_key),
                uncompressed_point_conversion: Self::derive_uncompressed(&priv_bytes, &public_key),
            },
        }
    }

    fn derive_compressed(priv_bytes: &[u8], pub_key: &PublicKey) -> CompressedPointConversion {
        let encoded = pub_key.to_encoded_point(true);
        let (addr, h160) = Self::derive_crypto(encoded.as_bytes());
        CompressedPointConversion {
            address: addr,
            asn1key: format!(
                "302e0201010420{}a00706052b8104000a",
                hex::encode(priv_bytes)
            ),
            public_key: hex::encode(encoded.as_bytes()),
            hash160: h160,
        }
    }

    fn derive_uncompressed(priv_bytes: &[u8], pub_key: &PublicKey) -> UncompressedPointConversion {
        let encoded = pub_key.to_encoded_point(false);
        let (addr, h160) = Self::derive_crypto(encoded.as_bytes());
        UncompressedPointConversion {
            address: addr,
            asn1key: format!(
                "304e0201010420{}a00706052b8104000a",
                hex::encode(priv_bytes)
            ),
            public_key: hex::encode(encoded.as_bytes()),
            hash160: h160,
        }
    }

    fn derive_crypto(pub_bytes: &[u8]) -> (String, String) {
        let sha = Sha256::digest(pub_bytes);
        let rip = Ripemd160::digest(sha);
        let hash160_hex = hex::encode(rip);

        let mut payload = vec![0x00];
        payload.extend_from_slice(&rip);
        let address = bs58::encode(&payload).with_check().into_string();

        (address, hash160_hex)
    }
}
