# ============================================================
# natural_aes_256_core_sym.py
# Core components for Natural (Sum-of-Corners) AES-256 sym attack
# - Bit-level Natural AES layers (SB/SR/MC/XOR)
# - AES-256 key schedule
# - NaturalAES256 blackbox + synthetic ARK-start oracle
# - Generic sym key-recovery (bit-wise)
# ============================================================

import numpy as np

# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------
def relu(x):
    return np.maximum(x, 0.0)

def nn_xor_abs(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Natural XOR: |x - k| implemented via ReLU."""
    return relu(k - x) + relu(x - k)

def bytes_to_bits(b: bytes) -> np.ndarray:
    """Bytes -> {0,1}^n as float64 (MSB-first per byte)."""
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float64)
    return bits.reshape(-1)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """{0,1}^n (float) -> bytes (threshold at 0.5)."""
    bits = np.asarray(bits).reshape(-1, 8)
    v = (bits > 0.5).astype(np.uint8)
    out = (v * (1 << np.arange(7, -1, -1))).sum(axis=1)
    return out.astype(np.uint8).tobytes()

# ------------------------------------------------------------
# Sum-of-Corners Boolean Network
# ------------------------------------------------------------
class SumOfCornersBoolean:
    def __init__(self, k: int, m: int, mapping: np.ndarray):
        """
        k: input bits
        m: output bits
        mapping: size 2^k truth table (uint8/uint32 etc)
        """
        self.k = k
        self.m = m
        self.mapping = mapping.astype(np.uint32)

        self.active_B = []
        for o in range(m):
            active = np.where(((self.mapping >> (m - 1 - o)) & 1) == 1)[0]
            B = ((active[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.float64)
            self.active_B.append(B)

    def forward(self, x: np.ndarray, c: float = 1.0) -> np.ndarray:
        xb = x.reshape(-1, self.k)
        out = np.zeros((xb.shape[0], self.m), dtype=np.float64)

        for o in range(self.m):
            B = self.active_B[o]
            match = (xb @ B.T) + ((1 - xb) @ (1 - B).T)
            out[:, o] = relu((match - self.k + c) / c).sum(axis=1)

        return out.reshape(x.shape[:-1] + (self.m,))

# ------------------------------------------------------------
# AES tables
# ------------------------------------------------------------
AES_SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

def gf_mul2(a: int) -> int:
    return ((a << 1) & 0xFF) ^ (0x1B if (a & 0x80) else 0)

def gf_mul3(a: int) -> int:
    return gf_mul2(a) ^ a

MUL2_MAP = np.array([gf_mul2(i) for i in range(256)], dtype=np.uint8)
MUL3_MAP = np.array([gf_mul3(i) for i in range(256)], dtype=np.uint8)

# ------------------------------------------------------------
# Neural nets for AES layers
# ------------------------------------------------------------
SBOX_NET = SumOfCornersBoolean(8, 8, AES_SBOX)
MUL2_NET = SumOfCornersBoolean(8, 8, MUL2_MAP)
MUL3_NET = SumOfCornersBoolean(8, 8, MUL3_MAP)

# ------------------------------------------------------------
# AES layer implementations (bit-level)
# ------------------------------------------------------------
def _shiftrows_perm() -> np.ndarray:
    P = np.zeros(16, dtype=np.int32)
    for r in range(4):
        for c in range(4):
            P[r + 4*c] = r + 4*((c + r) % 4)
    return P

SHIFTROWS_PERM = _shiftrows_perm()

def nn_subbytes_state(state_bits: np.ndarray) -> np.ndarray:
    s = state_bits.reshape(16, 8)
    out = np.zeros_like(s)
    for i in range(16):
        out[i] = SBOX_NET.forward(s[i].reshape(1, 8))[0]
    return out.reshape(128)

def nn_shiftrows_state(state_bits: np.ndarray) -> np.ndarray:
    s = state_bits.reshape(16, 8)
    return s[SHIFTROWS_PERM].reshape(128)

def nn_mixcolumns_state(state_bits: np.ndarray) -> np.ndarray:
    s = state_bits.reshape(16, 8)
    out = np.zeros_like(s)

    for col in range(4):
        a, b, c, d = s[4*col:4*col+4]

        a2, a3 = MUL2_NET.forward(a[None])[0], MUL3_NET.forward(a[None])[0]
        b2, b3 = MUL2_NET.forward(b[None])[0], MUL3_NET.forward(b[None])[0]
        c2, c3 = MUL2_NET.forward(c[None])[0], MUL3_NET.forward(c[None])[0]
        d2, d3 = MUL2_NET.forward(d[None])[0], MUL3_NET.forward(d[None])[0]

        out[4*col+0] = nn_xor_abs(nn_xor_abs(a2, b3), nn_xor_abs(c, d))
        out[4*col+1] = nn_xor_abs(nn_xor_abs(a, b2), nn_xor_abs(c3, d))
        out[4*col+2] = nn_xor_abs(nn_xor_abs(a, b), nn_xor_abs(c2, d3))
        out[4*col+3] = nn_xor_abs(nn_xor_abs(a3, b), nn_xor_abs(c, d2))

    return out.reshape(128)

# ------------------------------------------------------------
# Generic sym key-recovery (bit-wise)
# ------------------------------------------------------------
def genpairs_sym(p: np.ndarray, i: int, eps: float):
    x0  = p.copy(); x0[i]  = -eps
    x0p = p.copy(); x0p[i] = +eps
    x1  = p.copy(); x1[i]  = 1.0 - eps
    x1p = p.copy(); x1p[i] = 1.0 + eps
    return {0: (x0, x0p), 1: (x1, x1p)}

def keyrecovery_bits(
    oracle,
    nbits: int,
    eps: float = 0.01,
    max_outer_loops: int = 300,
    rng_seed: int = 0
):
    rng = np.random.default_rng(rng_seed)
    remaining = list(range(nbits))
    k_hat = np.full(nbits, -1, dtype=np.int32)
    loops = 0

    print("== Using genpairs_sym !!==")

    while remaining and loops < max_outer_loops:
        loops += 1
        p = rng.integers(0, 2, size=nbits).astype(np.float64)
        new_remaining = []

        for i in remaining:
            cand = []
            for v, (x, xp) in genpairs_sym(p, i, eps).items():
                if np.allclose(oracle(x), oracle(xp), atol=1e-12, rtol=0.0):
                    cand.append(v)
            if len(cand) == 1:
                k_hat[i] = cand[0]
            else:
                new_remaining.append(i)

        remaining = new_remaining

    return k_hat, loops

# ------------------------------------------------------------
# AES-256 key schedule (FIPS 197)
# ------------------------------------------------------------
RCON = np.array([0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36], dtype=np.uint8)

def aes256_roundkeys(key32: bytes):
    """Return list of 15 round keys (rk0..rk14), each 16 bytes, for AES-256."""
    Nk, Nr = 8, 14
    key = np.frombuffer(key32, dtype=np.uint8)
    if key.size != 32:
        raise ValueError("AES-256 master key must be 32 bytes")

    w = np.zeros((4*(Nr+1), 4), dtype=np.uint8)  # 60 words
    w[:Nk] = key.reshape(Nk, 4)

    for i in range(Nk, 4*(Nr+1)):
        temp = w[i-1].copy()

        if i % Nk == 0:
            # RotWord + SubWord + Rcon
            temp = AES_SBOX[np.roll(temp, -1)]
            temp[0] ^= RCON[(i//Nk) - 1]
        elif i % Nk == 4:
            # SubWord only (AES-256 special)
            temp = AES_SBOX[temp]

        w[i] = w[i-Nk] ^ temp

    return [w[4*r:4*r+4].reshape(16).tobytes() for r in range(Nr+1)]

# ------------------------------------------------------------
# Natural AES-256 Blackbox (bit-level)
# ------------------------------------------------------------
class NaturalAES256:
    def __init__(self, key32: bytes):
        rks = aes256_roundkeys(key32)
        self.rk_bits = [bytes_to_bits(rk) for rk in rks]  # 15 (0..14)

    def encrypt_bits(self, pt_bits_real: np.ndarray) -> np.ndarray:
        x = np.asarray(pt_bits_real, dtype=np.float64).reshape(128)

        # round 0
        x = nn_xor_abs(x, self.rk_bits[0])

        # rounds 1..13
        for r in range(1, 14):
            x = nn_subbytes_state(x)
            x = nn_shiftrows_state(x)
            x = nn_mixcolumns_state(x)
            x = nn_xor_abs(x, self.rk_bits[r])

        # final round 14
        x = nn_subbytes_state(x)
        x = nn_shiftrows_state(x)
        x = nn_xor_abs(x, self.rk_bits[14])
        return x

    def oracle_from_addroundkey(self, r: int):
        """Synthetic oracle with first op z <- z XOR rk[r], then continue."""
        assert 0 <= r <= 14

        def f(z_bits: np.ndarray) -> np.ndarray:
            x = np.asarray(z_bits, dtype=np.float64).reshape(128)

            # first: AddRoundKey of round r
            x = nn_xor_abs(x, self.rk_bits[r])

            if r == 0:
                for rr in range(1, 14):
                    x = nn_subbytes_state(x)
                    x = nn_shiftrows_state(x)
                    x = nn_mixcolumns_state(x)
                    x = nn_xor_abs(x, self.rk_bits[rr])
                x = nn_subbytes_state(x)
                x = nn_shiftrows_state(x)
                x = nn_xor_abs(x, self.rk_bits[14])
                return x

            if 1 <= r <= 13:
                for rr in range(r+1, 14):
                    x = nn_subbytes_state(x)
                    x = nn_shiftrows_state(x)
                    x = nn_mixcolumns_state(x)
                    x = nn_xor_abs(x, self.rk_bits[rr])
                x = nn_subbytes_state(x)
                x = nn_shiftrows_state(x)
                x = nn_xor_abs(x, self.rk_bits[14])
                return x

            # r == 14: final ARK only
            return x

        return f

# ------------------------------------------------------------
# Master-key reconstruction from recovered round keys
# ------------------------------------------------------------
def aes256_master_from_rk01(rk0: bytes, rk1: bytes, rk2=None) -> bytes:
    """
    AES-256 master key consists of w0..w7.
    rk0 = w0..w3, rk1 = w4..w7, so master key = rk0 || rk1.
    rk2 is optional (consistency check only).
    """
    if len(rk0) != 16 or len(rk1) != 16:
        raise ValueError("rk0 and rk1 must be 16 bytes each")

    master = rk0 + rk1  # 32 bytes

    if rk2 is not None:
        rks = aes256_roundkeys(master)
        if rks[0] != rk0 or rks[1] != rk1:
            print("[warning] rk0/rk1 mismatch in reconstructed AES-256 key")
        if rks[2] != rk2:
            print("[warning] rk2 mismatch (attack may be partial)")

    return master
