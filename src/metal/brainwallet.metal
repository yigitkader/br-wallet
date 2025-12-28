// ============================================================================
// BRAINWALLET GPU SHADER - Metal Implementation for Apple Silicon
//
// Pipeline: Passphrase → SHA256 → secp256k1 → HASH160/Keccak256 → Output
//
// Supported Chains (ALL computed on GPU):
// - Bitcoin: P2PKH (1...), P2SH-P2WPKH (3...), Native SegWit (bc1q...), Taproot (bc1p...)
// - Litecoin: Same address types with LTC prefixes
// - Ethereum: Keccak256-based addresses (0x...) - FULL GPU COMPUTATION!
//
// Current Optimizations:
// 1. One passphrase per GPU thread (massively parallel)
// 2. Extended Jacobian coordinates for EC operations
// 3. Fermat's Little Theorem modular inversion (a^(p-2) mod p)
// 4. Double-and-add scalar multiplication
// 5. Keccak256 on GPU for Ethereum (eliminates CPU bottleneck)
//
// TODO - Future optimizations (not yet implemented):
// - wNAF (Windowed Non-Adjacent Form) for faster scalar multiplication
// - GLV endomorphism for ~2x EC speedup
// - Montgomery batch inversion across thread groups
//
// Output per passphrase: h160_c(20) + h160_u(20) + h160_nested(20) + 
//                        taproot(32) + eth_addr(20) + priv_key(32) = 144 bytes
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// SECP256K1 CONSTANTS
// ============================================================================

constant ulong4 SECP256K1_P = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

constant ulong4 SECP256K1_N = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

constant ulong4 SECP256K1_GX = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
};

constant ulong4 SECP256K1_GY = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
};

constant ulong SECP256K1_K = 4294968273ULL;

// ============================================================================
// HASH CONSTANTS
// ============================================================================

constant uint SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

constant uint RIPEMD_KL[5] = {0x00000000,0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xa953fd4e};
constant uint RIPEMD_KR[5] = {0x50a28be6,0x5c4dd124,0x6d703ef3,0x7a6d76e9,0x00000000};
constant uchar RIPEMD_RL[80] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
constant uchar RIPEMD_RR[80] = {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
constant uchar RIPEMD_SL[80] = {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
constant uchar RIPEMD_SR[80] = {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};

// Output size per passphrase: h160_c(20) + h160_u(20) + h160_nested(20) + taproot(32) + eth_addr(20) + priv_key(32) = 144 bytes
// eth_addr is computed on GPU using Keccak256 - no CPU post-processing needed!
// priv_key is the SHA256(passphrase) - avoids recomputation on CPU
#define OUTPUT_SIZE 144
#define MAX_PASSPHRASE_LEN 128

// Input stride: 16-byte aligned header (1 byte length + 15 padding) + 128 bytes data = 144 bytes
// This alignment improves GPU memory coalescing performance
#define PASSPHRASE_HEADER_SIZE 16
#define PASSPHRASE_STRIDE (PASSPHRASE_HEADER_SIZE + MAX_PASSPHRASE_LEN)

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

inline bool IsZero(ulong4 a) {
    return (a.x | a.y | a.z | a.w) == 0;
}

inline ulong4 load_be(thread const uchar* d) {
    ulong4 r;
    r.w = ((ulong)d[0]<<56)|((ulong)d[1]<<48)|((ulong)d[2]<<40)|((ulong)d[3]<<32)|
          ((ulong)d[4]<<24)|((ulong)d[5]<<16)|((ulong)d[6]<<8)|(ulong)d[7];
    r.z = ((ulong)d[8]<<56)|((ulong)d[9]<<48)|((ulong)d[10]<<40)|((ulong)d[11]<<32)|
          ((ulong)d[12]<<24)|((ulong)d[13]<<16)|((ulong)d[14]<<8)|(ulong)d[15];
    r.y = ((ulong)d[16]<<56)|((ulong)d[17]<<48)|((ulong)d[18]<<40)|((ulong)d[19]<<32)|
          ((ulong)d[20]<<24)|((ulong)d[21]<<16)|((ulong)d[22]<<8)|(ulong)d[23];
    r.x = ((ulong)d[24]<<56)|((ulong)d[25]<<48)|((ulong)d[26]<<40)|((ulong)d[27]<<32)|
          ((ulong)d[28]<<24)|((ulong)d[29]<<16)|((ulong)d[30]<<8)|(ulong)d[31];
    return r;
}

inline void store_be(ulong4 v, thread uchar* o) {
    o[0]=(v.w>>56);o[1]=(v.w>>48);o[2]=(v.w>>40);o[3]=(v.w>>32);
    o[4]=(v.w>>24);o[5]=(v.w>>16);o[6]=(v.w>>8);o[7]=v.w;
    o[8]=(v.z>>56);o[9]=(v.z>>48);o[10]=(v.z>>40);o[11]=(v.z>>32);
    o[12]=(v.z>>24);o[13]=(v.z>>16);o[14]=(v.z>>8);o[15]=v.z;
    o[16]=(v.y>>56);o[17]=(v.y>>48);o[18]=(v.y>>40);o[19]=(v.y>>32);
    o[20]=(v.y>>24);o[21]=(v.y>>16);o[22]=(v.y>>8);o[23]=v.y;
    o[24]=(v.x>>56);o[25]=(v.x>>48);o[26]=(v.x>>40);o[27]=(v.x>>32);
    o[28]=(v.x>>24);o[29]=(v.x>>16);o[30]=(v.x>>8);o[31]=v.x;
}

inline uint rotr32(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
inline uint rotl32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

// ============================================================================
// SHA256 - Variable Length (for passphrase hashing)
// ============================================================================

void sha256_var(thread const uchar* data, uint data_len, thread uchar* hash) {
    uint state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    
    // Pad and process in 64-byte blocks
    uint total_bits = data_len * 8;
    uint padded_len = ((data_len + 9 + 63) / 64) * 64;
    
    // Process each 64-byte block
    uint processed = 0;
    while (processed < padded_len) {
        uint w[64];
        
        // Build message schedule
        for (int i = 0; i < 16; i++) {
            uint idx = processed + i * 4;
            uint val = 0;
            
            for (int j = 0; j < 4; j++) {
                uint byte_idx = idx + j;
                uchar byte_val = 0;
                
                if (byte_idx < data_len) {
                    byte_val = data[byte_idx];
                } else if (byte_idx == data_len) {
                    byte_val = 0x80;  // Padding start
                } else if (byte_idx >= padded_len - 8) {
                    // Length in bits (big-endian, last 8 bytes)
                    uint len_offset = byte_idx - (padded_len - 8);
                    if (len_offset < 4) {
                        byte_val = 0;  // High 32 bits (we only support < 4GB)
                    } else {
                        byte_val = (total_bits >> (24 - (len_offset - 4) * 8)) & 0xFF;
                    }
                }
                
                val = (val << 8) | byte_val;
            }
            w[i] = val;
        }
        
        // Extend message schedule
        for (int i = 16; i < 64; i++) {
            uint s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15]>>3);
            uint s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        
        // Compression
        uint a=state[0],b=state[1],c=state[2],d=state[3],
             e=state[4],f=state[5],g=state[6],h=state[7];
        
        for (int i = 0; i < 64; i++) {
            uint S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            uint ch = (e & f) ^ (~e & g);
            uint t1 = h + S1 + ch + SHA256_K[i] + w[i];
            uint S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            uint maj = (a & b) ^ (a & c) ^ (b & c);
            uint t2 = S0 + maj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
        
        processed += 64;
    }
    
    // Output hash
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (state[i]>>24); hash[i*4+1] = (state[i]>>16);
        hash[i*4+2] = (state[i]>>8); hash[i*4+3] = state[i];
    }
}

// SHA256 for 33-byte input (compressed pubkey)
void sha256_33(thread const uchar* data, thread uchar* hash) {
    uint state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint w[64];
    
    for (int i = 0; i < 8; i++) 
        w[i] = ((uint)data[i*4]<<24)|((uint)data[i*4+1]<<16)|
               ((uint)data[i*4+2]<<8)|(uint)data[i*4+3];
    w[8] = ((uint)data[32] << 24) | 0x800000;
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 264;  // 33 * 8 = 264 bits
    
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15]>>3);
        uint s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    uint a=state[0],b=state[1],c=state[2],d=state[3],
         e=state[4],f=state[5],g=state[6],h=state[7];
    
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (state[i]>>24); hash[i*4+1] = (state[i]>>16);
        hash[i*4+2] = (state[i]>>8); hash[i*4+3] = state[i];
    }
}

// SHA256 for 65-byte input (uncompressed pubkey)
void sha256_65(thread const uchar* data, thread uchar* hash) {
    uint state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint w[64];
    
    // First block: bytes 0-63
    for (int i = 0; i < 16; i++) 
        w[i] = ((uint)data[i*4]<<24)|((uint)data[i*4+1]<<16)|
               ((uint)data[i*4+2]<<8)|(uint)data[i*4+3];
    
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15]>>3);
        uint s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    uint a=state[0],b=state[1],c=state[2],d=state[3],
         e=state[4],f=state[5],g=state[6],h=state[7];
    
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    
    // Second block: byte 64 + padding + length
    w[0] = ((uint)data[64] << 24) | 0x800000;  // Last byte + 0x80
    for (int i = 1; i < 15; i++) w[i] = 0;
    w[15] = 520;  // 65 * 8 = 520 bits
    
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15]>>3);
        uint s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    a=state[0]; b=state[1]; c=state[2]; d=state[3];
    e=state[4]; f=state[5]; g=state[6]; h=state[7];
    
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (state[i]>>24); hash[i*4+1] = (state[i]>>16);
        hash[i*4+2] = (state[i]>>8); hash[i*4+3] = state[i];
    }
}

// SHA256 for 22-byte input (witness script)
void sha256_22(thread const uchar* input, thread uchar* out) {
    uint state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint w[64];
    
    w[0] = ((uint)input[0]<<24)|((uint)input[1]<<16)|((uint)input[2]<<8)|(uint)input[3];
    w[1] = ((uint)input[4]<<24)|((uint)input[5]<<16)|((uint)input[6]<<8)|(uint)input[7];
    w[2] = ((uint)input[8]<<24)|((uint)input[9]<<16)|((uint)input[10]<<8)|(uint)input[11];
    w[3] = ((uint)input[12]<<24)|((uint)input[13]<<16)|((uint)input[14]<<8)|(uint)input[15];
    w[4] = ((uint)input[16]<<24)|((uint)input[17]<<16)|((uint)input[18]<<8)|(uint)input[19];
    w[5] = ((uint)input[20]<<24)|((uint)input[21]<<16)|0x8000;
    for (int i = 6; i < 15; i++) w[i] = 0;
    w[15] = 176;  // 22 * 8 = 176 bits
    
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15]>>3);
        uint s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    uint a=state[0],b=state[1],c=state[2],d=state[3],
         e=state[4],f=state[5],g=state[6],h=state[7];
    
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    
    for (int i = 0; i < 8; i++) {
        out[i*4] = (state[i]>>24); out[i*4+1] = (state[i]>>16);
        out[i*4+2] = (state[i]>>8); out[i*4+3] = state[i];
    }
}

// ============================================================================
// RIPEMD-160
// ============================================================================

void ripemd160_32(thread const uchar* data, thread uchar* hash) {
    uint h0=0x67452301,h1=0xefcdab89,h2=0x98badcfe,h3=0x10325476,h4=0xc3d2e1f0;
    uint x[16];
    
    for (int i = 0; i < 8; i++) 
        x[i] = ((uint)data[i*4]) | ((uint)data[i*4+1]<<8) | 
               ((uint)data[i*4+2]<<16) | ((uint)data[i*4+3]<<24);
    x[8] = 0x80;
    for (int i = 9; i < 14; i++) x[i] = 0;
    x[14] = 256; x[15] = 0;  // 32 * 8 = 256 bits
    
    uint al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    uint ar=h0,br=h1,cr=h2,dr=h3,er=h4;
    
    for (int j = 0; j < 80; j++) {
        int r = j / 16;
        uint fl = (r==0) ? (bl^cl^dl) : (r==1) ? ((bl&cl)|(~bl&dl)) : 
                 (r==2) ? ((bl|~cl)^dl) : (r==3) ? ((bl&dl)|(cl&~dl)) : (bl^(cl|~dl));
        uint fr = (r==0) ? (br^(cr|~dr)) : (r==1) ? ((br&dr)|(cr&~dr)) : 
                 (r==2) ? ((br|~cr)^dr) : (r==3) ? ((br&cr)|(~br&dr)) : (br^cr^dr);
        uint tl = rotl32(al+fl+x[RIPEMD_RL[j]]+RIPEMD_KL[r], RIPEMD_SL[j]) + el;
        al=el; el=dl; dl=rotl32(cl,10); cl=bl; bl=tl;
        uint tr = rotl32(ar+fr+x[RIPEMD_RR[j]]+RIPEMD_KR[r], RIPEMD_SR[j]) + er;
        ar=er; er=dr; dr=rotl32(cr,10); cr=br; br=tr;
    }
    
    uint t = h1+cl+dr;
    h1 = h2+dl+er; h2 = h3+el+ar; h3 = h4+al+br; h4 = h0+bl+cr; h0 = t;
    
    hash[0]=h0; hash[1]=h0>>8; hash[2]=h0>>16; hash[3]=h0>>24;
    hash[4]=h1; hash[5]=h1>>8; hash[6]=h1>>16; hash[7]=h1>>24;
    hash[8]=h2; hash[9]=h2>>8; hash[10]=h2>>16; hash[11]=h2>>24;
    hash[12]=h3; hash[13]=h3>>8; hash[14]=h3>>16; hash[15]=h3>>24;
    hash[16]=h4; hash[17]=h4>>8; hash[18]=h4>>16; hash[19]=h4>>24;
}

// HASH160 = RIPEMD160(SHA256(data))
void hash160_33(thread const uchar* pubkey, thread uchar* out) {
    uchar sha[32];
    sha256_33(pubkey, sha);
    ripemd160_32(sha, out);
}

void hash160_65(thread const uchar* pubkey, thread uchar* out) {
    uchar sha[32];
    sha256_65(pubkey, sha);
    ripemd160_32(sha, out);
}

void hash160_22(thread const uchar* script, thread uchar* out) {
    uchar sha[32];
    sha256_22(script, sha);
    ripemd160_32(sha, out);
}

// ============================================================================
// KECCAK-256 (Ethereum Address Computation)
// ============================================================================
//
// Keccak-256 is used to derive Ethereum addresses from uncompressed public keys:
// eth_address = Keccak256(pubkey_x || pubkey_y)[12:32]  (last 20 bytes)
//
// Implementation: Keccak-f[1600] with r=1088, c=512 (Keccak-256 parameters)
//

// Keccak round constants
constant ulong KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Keccak rotation offsets
constant int KECCAK_ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

// Keccak pi lane indices
constant int KECCAK_PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

inline ulong rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation
void keccakf(thread ulong* st) {
    for (int round = 0; round < 24; round++) {
        // Theta
        ulong bc[5];
        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        
        for (int i = 0; i < 5; i++) {
            ulong t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                st[j + i] ^= t;
            }
        }
        
        // Rho and Pi
        ulong t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = KECCAK_PILN[i];
            ulong temp = st[j];
            st[j] = rotl64(t, KECCAK_ROTC[i]);
            t = temp;
        }
        
        // Chi
        for (int j = 0; j < 25; j += 5) {
            ulong temp[5];
            for (int i = 0; i < 5; i++) {
                temp[i] = st[j + i];
            }
            for (int i = 0; i < 5; i++) {
                st[j + i] ^= (~temp[(i + 1) % 5]) & temp[(i + 2) % 5];
            }
        }
        
        // Iota
        st[0] ^= KECCAK_RC[round];
    }
}

// Keccak-256 for 64-byte input (uncompressed pubkey without 0x04 prefix)
// Returns 32-byte hash, but we only need last 20 bytes for Ethereum address
void keccak256_64(thread const uchar* data, thread uchar* hash) {
    // State: 25 x 64-bit words = 200 bytes
    ulong st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    
    // Absorb: rate = 1088 bits = 136 bytes, so 64 bytes fits in one block
    // Load 64 bytes of data (8 words) in little-endian
    for (int i = 0; i < 8; i++) {
        st[i] = ((ulong)data[i*8]) | ((ulong)data[i*8+1] << 8) |
                ((ulong)data[i*8+2] << 16) | ((ulong)data[i*8+3] << 24) |
                ((ulong)data[i*8+4] << 32) | ((ulong)data[i*8+5] << 40) |
                ((ulong)data[i*8+6] << 48) | ((ulong)data[i*8+7] << 56);
    }
    
    // Padding: 0x01 after data, 0x80 at end of rate block
    // For 64-byte input with 136-byte rate:
    // Byte 64 = 0x01, Byte 135 = 0x80
    st[8] ^= 0x01ULL;           // Byte 64: padding start (in word 8)
    st[16] ^= 0x8000000000000000ULL;  // Byte 135: padding end (MSB of word 16, since 135 = 16*8 + 7)
    
    // Apply Keccak-f[1600]
    keccakf(st);
    
    // Squeeze: extract 32 bytes (256 bits) in little-endian
    for (int i = 0; i < 4; i++) {
        hash[i*8+0] = st[i] & 0xFF;
        hash[i*8+1] = (st[i] >> 8) & 0xFF;
        hash[i*8+2] = (st[i] >> 16) & 0xFF;
        hash[i*8+3] = (st[i] >> 24) & 0xFF;
        hash[i*8+4] = (st[i] >> 32) & 0xFF;
        hash[i*8+5] = (st[i] >> 40) & 0xFF;
        hash[i*8+6] = (st[i] >> 48) & 0xFF;
        hash[i*8+7] = (st[i] >> 56) & 0xFF;
    }
}

// Compute Ethereum address from uncompressed pubkey (64 bytes, no 0x04 prefix)
// Returns 20-byte address = Keccak256(pubkey)[12:32]
void eth_address_from_pubkey(thread const uchar* pubkey_xy, thread uchar* addr) {
    uchar hash[32];
    keccak256_64(pubkey_xy, hash);
    
    // Ethereum address = last 20 bytes of Keccak256 hash
    for (int i = 0; i < 20; i++) {
        addr[i] = hash[12 + i];
    }
}

// ============================================================================
// MODULAR ARITHMETIC (256-bit over secp256k1 prime)
// ============================================================================

inline void mul64(ulong a, ulong b, thread ulong& hi, thread ulong& lo) {
    ulong a_lo = a & 0xFFFFFFFF, a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF, b_hi = b >> 32;
    ulong p0 = a_lo * b_lo;
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;
    ulong mid = p1 + (p0 >> 32);
    mid += p2;
    if (mid < p2) p3 += 0x100000000UL;
    lo = (p0 & 0xFFFFFFFF) | (mid << 32);
    hi = p3 + (mid >> 32);
}

inline ulong4 add_p(ulong4 a) {
    ulong4 r; ulong c = 0;
    r.x = a.x + SECP256K1_P.x; c = (r.x < a.x) ? 1 : 0;
    r.y = a.y + SECP256K1_P.y + c; c = (r.y < a.y) || (c && r.y == a.y) ? 1 : 0;
    r.z = a.z + SECP256K1_P.z + c; c = (r.z < a.z) || (c && r.z == a.z) ? 1 : 0;
    r.w = a.w + SECP256K1_P.w + c;
    return r;
}

inline ulong4 mod_add(ulong4 a, ulong4 b) {
    ulong4 r; ulong c = 0;
    r.x = a.x + b.x; c = (r.x < a.x) ? 1 : 0;
    r.y = a.y + b.y + c; c = (r.y < a.y) || (c && r.y == a.y) ? 1 : 0;
    r.z = a.z + b.z + c; c = (r.z < a.z) || (c && r.z == a.z) ? 1 : 0;
    r.w = a.w + b.w + c;
    ulong fc = (r.w < a.w) || (c && r.w == a.w) ? 1 : 0;
    
    if (fc || r.w > SECP256K1_P.w || (r.w == SECP256K1_P.w && (r.z > SECP256K1_P.z ||
        (r.z == SECP256K1_P.z && (r.y > SECP256K1_P.y || (r.y == SECP256K1_P.y && r.x >= SECP256K1_P.x)))))) {
        ulong4 s; ulong bw = 0;
        s.x = r.x - SECP256K1_P.x; bw = (r.x < SECP256K1_P.x) ? 1 : 0;
        s.y = r.y - SECP256K1_P.y - bw; bw = (r.y < SECP256K1_P.y) || (bw && r.y == SECP256K1_P.y) ? 1 : 0;
        s.z = r.z - SECP256K1_P.z - bw; bw = (r.z < SECP256K1_P.z) || (bw && r.z == SECP256K1_P.z) ? 1 : 0;
        s.w = r.w - SECP256K1_P.w - bw;
        return s;
    }
    return r;
}

inline ulong4 mod_sub(ulong4 a, ulong4 b) {
    bool need = a.w < b.w || (a.w == b.w && (a.z < b.z ||
        (a.z == b.z && (a.y < b.y || (a.y == b.y && a.x < b.x)))));
    if (need) a = add_p(a);
    ulong4 r; ulong bw = 0;
    r.x = a.x - b.x; bw = (a.x < b.x) ? 1 : 0;
    r.y = a.y - b.y - bw; bw = (a.y < b.y) || (bw && a.y == b.y) ? 1 : 0;
    r.z = a.z - b.z - bw; bw = (a.z < b.z) || (bw && a.z == b.z) ? 1 : 0;
    r.w = a.w - b.w - bw;
    return r;
}

inline ulong4 secp256k1_reduce(ulong r0, ulong r1, ulong r2, ulong r3,
                               ulong r4, ulong r5, ulong r6, ulong r7) {
    ulong s0 = r0, s1 = r1, s2 = r2, s3 = r3, c = 0, hi, lo, old;

    mul64(r4, SECP256K1_K, hi, lo);
    old = s0; s0 += lo; c = (s0 < old) ? 1 : 0;
    old = s1; s1 += hi + c; c = (s1 < old || (c && s1 == old + hi)) ? 1 : 0;
    s2 += c; c = (s2 < c) ? 1 : 0; s3 += c;

    mul64(r5, SECP256K1_K, hi, lo);
    old = s1; s1 += lo; c = (s1 < old) ? 1 : 0;
    old = s2; s2 += hi + c; c = (s2 < old || (c && s2 == old + hi)) ? 1 : 0;
    s3 += c;

    mul64(r6, SECP256K1_K, hi, lo);
    old = s2; s2 += lo; c = (s2 < old) ? 1 : 0;
    s3 += hi + c;

    mul64(r7, SECP256K1_K, hi, lo);
    old = s3; s3 += lo;
    ulong overflow = hi + ((s3 < old) ? 1 : 0);

    if (overflow > 0) {
        mul64(overflow, SECP256K1_K, hi, lo);
        old = s0; s0 += lo; c = (s0 < old) ? 1 : 0;
        old = s1; s1 += hi + c; c = (s1 < old) ? 1 : 0;
        s2 += c; c = (s2 < c) ? 1 : 0; s3 += c;
    }

    ulong4 res = {s0, s1, s2, s3};
    bool need = s3 > SECP256K1_P.w || (s3 == SECP256K1_P.w && (s2 > SECP256K1_P.z ||
        (s2 == SECP256K1_P.z && (s1 > SECP256K1_P.y || (s1 == SECP256K1_P.y && s0 >= SECP256K1_P.x)))));
    if (need) res = mod_sub(res, SECP256K1_P);
    return res;
}

inline ulong add_with_carry(ulong a, ulong b, ulong carry_in, thread ulong* carry_out) {
    ulong sum = a + b;
    ulong c1 = (sum < a) ? 1ULL : 0ULL;
    sum += carry_in;
    ulong c2 = (sum < carry_in) ? 1ULL : 0ULL;
    *carry_out = c1 + c2;
    return sum;
}

ulong4 mod_mul(ulong4 a, ulong4 b) {
    ulong r[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        ulong ai = (i == 0) ? a.x : ((i == 1) ? a.y : ((i == 2) ? a.z : a.w));
        ulong c = 0;
        for (int j = 0; j < 4; j++) {
            ulong bj = (j == 0) ? b.x : ((j == 1) ? b.y : ((j == 2) ? b.z : b.w));
            ulong hi, lo;
            mul64(ai, bj, hi, lo);
            
            ulong c1;
            r[i + j] = add_with_carry(r[i + j], lo, 0, &c1);
            ulong c2, c3;
            r[i + j + 1] = add_with_carry(r[i + j + 1], hi, c1, &c2);
            r[i + j + 1] = add_with_carry(r[i + j + 1], c, 0, &c3);
            c = c2 + c3;
        }
        for (int k = i + 4; k < 8 && c; k++) {
            ulong ck;
            r[k] = add_with_carry(r[k], c, 0, &ck);
            c = ck;
        }
    }
    return secp256k1_reduce(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

inline ulong4 mod_sqr(ulong4 a) { return mod_mul(a, a); }

ulong4 mod_inv(ulong4 a) {
    if (IsZero(a)) return ulong4{0, 0, 0, 0};
    
    ulong4 res = {1, 0, 0, 0};
    ulong4 base = a;
    ulong exp[4] = {0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 
                    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    
    for (int w = 0; w < 4; w++) {
        ulong word = exp[w];
        for (int i = 0; i < 64; i++) {
            ulong4 mul_res = mod_mul(res, base);
            ulong mask = -(word & 1ULL);
            res.x = (res.x & ~mask) | (mul_res.x & mask);
            res.y = (res.y & ~mask) | (mul_res.y & mask);
            res.z = (res.z & ~mask) | (mul_res.z & mask);
            res.w = (res.w & ~mask) | (mul_res.w & mask);
            base = mod_sqr(base);
            word >>= 1;
        }
    }
    return res;
}

// ============================================================================
// ELLIPTIC CURVE OPERATIONS (Extended Jacobian)
// ============================================================================

inline void ext_jac_dbl(ulong4 X1, ulong4 Y1, ulong4 Z1, ulong4 ZZ1,
                        thread ulong4& X3, thread ulong4& Y3, 
                        thread ulong4& Z3, thread ulong4& ZZ3) {
    if (IsZero(Z1) || IsZero(Y1)) {
        X3 = X1; Y3 = Y1; Z3 = Z1; ZZ3 = ZZ1;
        return;
    }
    
    ulong4 Y2 = mod_sqr(Y1);
    ulong4 S = mod_mul(X1, Y2);
    S = mod_add(S, S);
    S = mod_add(S, S);
    
    ulong4 X2 = mod_sqr(X1);
    ulong4 M = mod_add(X2, mod_add(X2, X2));
    
    X3 = mod_sub(mod_sqr(M), mod_add(S, S));
    
    Z3 = mod_mul(Y1, Z1);
    Z3 = mod_add(Z3, Z3);
    
    ZZ3 = mod_sqr(Z3);
    
    ulong4 Y4 = mod_sqr(Y2);
    Y4 = mod_add(Y4, Y4);
    Y4 = mod_add(Y4, Y4);
    Y4 = mod_add(Y4, Y4);
    Y3 = mod_sub(mod_mul(M, mod_sub(S, X3)), Y4);
}

inline void ext_jac_add_affine(ulong4 X1, ulong4 Y1, ulong4 Z1, ulong4 ZZ1,
                               ulong4 ax, ulong4 ay,
                               thread ulong4& X3, thread ulong4& Y3, 
                               thread ulong4& Z3, thread ulong4& ZZ3) {
    if (IsZero(Z1)) {
        X3 = ax; Y3 = ay; Z3 = {1,0,0,0}; ZZ3 = {1,0,0,0};
        return;
    }
    
    ulong4 U2 = mod_mul(ax, ZZ1);
    ulong4 S2 = mod_mul(ay, mod_mul(Z1, ZZ1));
    
    ulong4 H = mod_sub(U2, X1);
    ulong4 R = mod_sub(S2, Y1);
    
    if (IsZero(H)) {
        if (IsZero(R)) {
            ext_jac_dbl(X1, Y1, Z1, ZZ1, X3, Y3, Z3, ZZ3);
        } else {
            X3 = {0,0,0,0}; Y3 = {1,0,0,0}; Z3 = {0,0,0,0}; ZZ3 = {0,0,0,0};
        }
        return;
    }
    
    ulong4 H2 = mod_sqr(H);
    ulong4 H3 = mod_mul(H2, H);
    ulong4 V = mod_mul(X1, H2);
    
    X3 = mod_sub(mod_sub(mod_sqr(R), H3), mod_add(V, V));
    Y3 = mod_sub(mod_mul(R, mod_sub(V, X3)), mod_mul(Y1, H3));
    Z3 = mod_mul(Z1, H);
    ZZ3 = mod_sqr(Z3);
}

// ============================================================================
// SCALAR MULTIPLICATION (Double-and-Add) - FIXED
// ============================================================================
//
// FIX: Corrected word processing order from LSB→MSB to MSB→LSB
// This matches the standard double-and-add algorithm for scalar multiplication
//

void scalar_mul(ulong4 k, 
                thread ulong4& out_X, thread ulong4& out_Y,
                thread ulong4& out_Z, thread ulong4& out_ZZ) {
    // Start with point at infinity
    out_X = {0, 0, 0, 0};
    out_Y = {1, 0, 0, 0};
    out_Z = {0, 0, 0, 0};
    out_ZZ = {0, 0, 0, 0};
    
    // Generator point G
    ulong4 Gx = SECP256K1_GX;
    ulong4 Gy = SECP256K1_GY;
    
    // FIXED: Process words from MSB to LSB (word 3 → 2 → 1 → 0)
    for (int word = 3; word >= 0; word--) {
        ulong bits = (word == 3) ? k.w : 
                     ((word == 2) ? k.z : 
                     ((word == 1) ? k.y : k.x));
        
        for (int bit = 0; bit < 64; bit++) {
            // Double
            ext_jac_dbl(out_X, out_Y, out_Z, out_ZZ, out_X, out_Y, out_Z, out_ZZ);
            
            // Add G if bit is set
            if ((bits >> (63 - bit)) & 1) {
                ext_jac_add_affine(out_X, out_Y, out_Z, out_ZZ, Gx, Gy, 
                                   out_X, out_Y, out_Z, out_ZZ);
            }
        }
    }
}

// ============================================================================
// TAPROOT (BIP341) - Tagged Hash for "TapTweak" (hardcoded)
// ============================================================================

// Pre-computed SHA256("TapTweak") = 0xe80fe1639c9ca050e3af1b39c143c63e429cbceb15d940fbb5c5a1f4af57c5e9
constant uchar TAPTWEAKREAK_HASH[32] = {
    0xe8, 0x0f, 0xe1, 0x63, 0x9c, 0x9c, 0xa0, 0x50,
    0xe3, 0xaf, 0x1b, 0x39, 0xc1, 0x43, 0xc6, 0x3e,
    0x42, 0x9c, 0xbc, 0xeb, 0x15, 0xd9, 0x40, 0xfb,
    0xb5, 0xc5, 0xa1, 0xf4, 0xaf, 0x57, 0xc5, 0xe9
};

void taptweak_hash(thread const uchar* pubkey_x, thread uchar* out) {
    // SHA256(SHA256("TapTweak") || SHA256("TapTweak") || pubkey_x)
    // = SHA256(TAPTWEAKREAK_HASH || TAPTWEAKREAK_HASH || pubkey_x)
    // Total: 64 + 32 = 96 bytes
    uchar combined[96];
    for (int i = 0; i < 32; i++) {
        combined[i] = TAPTWEAKREAK_HASH[i];
        combined[32 + i] = TAPTWEAKREAK_HASH[i];
        combined[64 + i] = pubkey_x[i];
    }
    sha256_var(combined, 96, out);
}

// ============================================================================
// MAIN KERNEL: process_brainwallet_batch
// ============================================================================

kernel void process_brainwallet_batch(
    device const uchar* passphrases [[buffer(0)]],      // Flat buffer: [len:1][data:MAX_PASSPHRASE_LEN] per entry
    device const uint* passphrase_count [[buffer(1)]],  // Number of passphrases
    device uchar* output [[buffer(2)]],                 // Output: 92 bytes per passphrase
    uint gid [[thread_position_in_grid]]
) {
    uint count = *passphrase_count;
    if (gid >= count) return;
    
    // Read passphrase from 16-byte aligned buffer
    // Format: [length:1][padding:15][data:128] = 144 bytes per entry
    uint pp_offset = gid * PASSPHRASE_STRIDE;
    uint pp_len = passphrases[pp_offset];  // Length at byte 0
    
    uchar passphrase[MAX_PASSPHRASE_LEN];
    // Data starts at byte 16 (after 16-byte aligned header)
    for (uint i = 0; i < pp_len && i < MAX_PASSPHRASE_LEN; i++) {
        passphrase[i] = passphrases[pp_offset + PASSPHRASE_HEADER_SIZE + i];
    }
    
    // Step 1: SHA256(passphrase) → private_key (32 bytes)
    uchar priv_key[32];
    sha256_var(passphrase, pp_len, priv_key);
    
    // Step 2: Validate private key (must be < curve order and non-zero)
    ulong4 k = load_be(priv_key);
    bool is_zero = IsZero(k);
    bool ge_order = k.w > SECP256K1_N.w || 
        (k.w == SECP256K1_N.w && k.z > SECP256K1_N.z) ||
        (k.w == SECP256K1_N.w && k.z == SECP256K1_N.z && k.y > SECP256K1_N.y) ||
        (k.w == SECP256K1_N.w && k.z == SECP256K1_N.z && k.y == SECP256K1_N.y && k.x >= SECP256K1_N.x);
    
    uint out_offset = gid * OUTPUT_SIZE;
    
    if (is_zero || ge_order) {
        // Invalid key - zero output
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[out_offset + i] = 0;
        }
        return;
    }
    
    // Step 3: k*G → Public Key (Extended Jacobian)
    ulong4 X, Y, Z, ZZ;
    scalar_mul(k, X, Y, Z, ZZ);
    
    // Convert to affine: x = X/Z², y = Y/Z³
    ulong4 Z_inv = mod_inv(Z);
    ulong4 Z_inv2 = mod_sqr(Z_inv);
    ulong4 Z_inv3 = mod_mul(Z_inv2, Z_inv);
    ulong4 ax = mod_mul(X, Z_inv2);
    ulong4 ay = mod_mul(Y, Z_inv3);
    
    // Step 4: Build compressed pubkey (33 bytes)
    uchar pubkey_c[33];
    pubkey_c[0] = (ay.x & 1) ? 0x03 : 0x02;
    store_be(ax, pubkey_c + 1);
    
    // Step 5: Build uncompressed pubkey (65 bytes)
    uchar pubkey_u[65];
    pubkey_u[0] = 0x04;
    store_be(ax, pubkey_u + 1);
    store_be(ay, pubkey_u + 33);
    
    // Step 6: HASH160(compressed pubkey) → h160_c (20 bytes)
    uchar h160_c[20];
    hash160_33(pubkey_c, h160_c);
    
    // Step 7: HASH160(uncompressed pubkey) → h160_u (20 bytes)
    uchar h160_u[20];
    hash160_65(pubkey_u, h160_u);
    
    // Step 8: P2SH-P2WPKH nested script hash → h160_nested (20 bytes)
    uchar witness_script[22];
    witness_script[0] = 0x00;  // OP_0
    witness_script[1] = 0x14;  // PUSH20
    for (int i = 0; i < 20; i++) {
        witness_script[2 + i] = h160_c[i];
    }
    uchar h160_nested[20];
    hash160_22(witness_script, h160_nested);
    
    // Step 9: Taproot output key (BIP341) - FULL IMPLEMENTATION
    // Output key Q = P + t*G where:
    //   P = internal key (x-only pubkey with even y per BIP340)
    //   t = tagged_hash("TapTweak", P)
    //   G = generator point
    
    // Internal key = x-only pubkey (32 bytes)
    uchar x_only[32];
    store_be(ax, x_only);
    
    // t = TapTweak hash
    uchar tweak_bytes[32];
    taptweak_hash(x_only, tweak_bytes);
    
    // Convert tweak to scalar
    ulong4 tweak_scalar = load_be(tweak_bytes);
    
    // BIP340: For x-only pubkey, we need y to be even
    // If y is odd (LSB = 1), negate it: y = p - y
    ulong4 Py = ay;
    if (ay.x & 1) {
        // y is odd, negate it: Py = p - ay
        Py = mod_sub(SECP256K1_P, ay);
    }
    
    // Compute t*G (tweak point)
    ulong4 tGx, tGy, tGz, tGzz;
    scalar_mul(tweak_scalar, tGx, tGy, tGz, tGzz);
    
    // Convert t*G to affine
    ulong4 tGz_inv = mod_inv(tGz);
    ulong4 tGz_inv2 = mod_sqr(tGz_inv);
    ulong4 tGz_inv3 = mod_mul(tGz_inv2, tGz_inv);
    ulong4 tGx_aff = mod_mul(tGx, tGz_inv2);
    ulong4 tGy_aff = mod_mul(tGy, tGz_inv3);
    
    // P = (ax, Py) - internal key with even y
    // Q = P + t*G using affine point addition
    
    // Check if P == t*G (edge case - would need point doubling)
    bool same_x = (ax.x == tGx_aff.x && ax.y == tGx_aff.y && 
                   ax.z == tGx_aff.z && ax.w == tGx_aff.w);
    
    ulong4 Qx, Qy;
    bool use_fallback = false;
    
    if (same_x) {
        // Points have same x - either equal or inverse
        bool same_y = (Py.x == tGy_aff.x && Py.y == tGy_aff.y &&
                      Py.z == tGy_aff.z && Py.w == tGy_aff.w);
        if (same_y) {
            // Point doubling: Q = 2P
            ulong4 s_num = mod_mul(ax, ax);
            s_num = mod_add(s_num, mod_add(s_num, s_num)); // 3x^2
            ulong4 s_den = mod_add(Py, Py);                 // 2y
            ulong4 s = mod_mul(s_num, mod_inv(s_den));
            
            Qx = mod_sub(mod_sqr(s), mod_add(ax, ax));
            Qy = mod_sub(mod_mul(s, mod_sub(ax, Qx)), Py);
        } else {
            // Inverse points - result is infinity (shouldn't happen)
            use_fallback = true;
        }
    } else {
        // Standard affine point addition: Q = P + t*G
        // slope = (tGy - Py) / (tGx - Px)
        ulong4 dy = mod_sub(tGy_aff, Py);
        ulong4 dx = mod_sub(tGx_aff, ax);
        ulong4 dx_inv = mod_inv(dx);
        ulong4 slope = mod_mul(dy, dx_inv);
        
        // Qx = slope^2 - Px - tGx
        ulong4 slope2 = mod_sqr(slope);
        Qx = mod_sub(mod_sub(slope2, ax), tGx_aff);
        
        // Qy = slope * (Px - Qx) - Py
        Qy = mod_sub(mod_mul(slope, mod_sub(ax, Qx)), Py);
    }
    
    // Output taproot key
    uchar taproot[32];
    if (use_fallback) {
        // Fallback to internal key (edge case)
        for (int i = 0; i < 32; i++) taproot[i] = x_only[i];
    } else {
        // Output = Qx (x-only, 32 bytes)
        store_be(Qx, taproot);
    }
    
    // Step 10: Ethereum address = Keccak256(pubkey_u without 0x04)[12:32]
    // Computed entirely on GPU - no CPU post-processing needed!
    uchar eth_addr[20];
    eth_address_from_pubkey(pubkey_u + 1, eth_addr);  // Skip 0x04 prefix
    
    // Write output: h160_c(20) + h160_u(20) + h160_nested(20) + taproot(32) + eth_addr(20) + priv_key(32) = 144 bytes
    for (int i = 0; i < 20; i++) {
        output[out_offset + i] = h160_c[i];
        output[out_offset + 20 + i] = h160_u[i];
        output[out_offset + 40 + i] = h160_nested[i];
    }
    for (int i = 0; i < 32; i++) {
        output[out_offset + 60 + i] = taproot[i];
    }
    // eth_addr (20 bytes) - computed on GPU via Keccak256
    for (int i = 0; i < 20; i++) {
        output[out_offset + 92 + i] = eth_addr[i];
    }
    // priv_key (32 bytes) - avoids SHA256 recomputation on CPU
    for (int i = 0; i < 32; i++) {
        output[out_offset + 112 + i] = priv_key[i];
    }
}

