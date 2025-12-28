┌─────────────────────────────────────────────────────────────────┐
│                       BRWALLET v3.0.0                          │
│                     GPU-ONLY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Wordlist ──> PassphraseBatcher ──> GPU (Metal) ──> Results    │
│                                        │                        │
│                    ┌───────────────────┴───────────────────┐    │
│                    │                                       │    │
│                    │  SHA256 → secp256k1 → HASH160         │    │
│                    │                  ↓                     │    │
│                    │            Keccak256 (ETH)            │    │
│                    │                                       │    │
│                    └───────────────────────────────────────┘    │
│                                        │                        │
│                                        ↓                        │
│  found.txt <── Match Check <── [h160_c, h160_u, h160_nested,   │
│                                 taproot, eth_addr, priv_key]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Performans:
GPU Output: 144 byte/passphrase
Input Stride: 144 byte (16-byte aligned)
Tüm kripto: GPU'da
CPU işi: Sadece eşleşme kontrolü (HashSet lookup)