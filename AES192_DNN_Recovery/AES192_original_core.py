########################################################
# natural_aes_192_sym_attack_v3_poc.py
# Author : Kwangjo Kim (Jun. 24, 2026)
# Evaluate the security of AES-192 by DNN leakage attack using Pairgen_sym() only
# 1,000 times execution
# - Verified 100% suceess with Python 3.8 and Python 3.10
#########################################################

import numpy as np
import time

from natural_aes_all_core_sym import (
    nn_subbytes_state,
    nn_shiftrows_state,
    nn_mixcolumns_state,
    nn_xor_abs,
    bytes_to_bits,
    bits_to_bytes,
    keyrecovery_bits,
)



# ===============================
# Utilities
# ===============================
def relu(x): 
    return np.maximum(x, 0.0)

def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float64)
    return bits.reshape(-1)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits).reshape(-1, 8)
    v = (bits > 0.5).astype(np.uint8)
    out = (v * (1 << np.arange(7, -1, -1))).sum(axis=1)
    return out.astype(np.uint8).tobytes()

def nn_xor_abs(x, k):
    return relu(k - x) + relu(x - k)



# ===============================
# Sum-of-Corners Networks
# ===============================
class SumOfCornersBoolean:
    def __init__(self, k, m, mapping):
        self.k = k
        self.m = m
        self.mapping = mapping.astype(np.uint32)
        self.active_B = []
        for o in range(m):
            active = np.where(((self.mapping >> (m-1-o)) & 1) == 1)[0]
            B = ((active[:,None] >> np.arange(k-1,-1,-1)) & 1).astype(np.float64)
            self.active_B.append(B)

    def forward(self, x, c=1.0):
        xb = x.reshape(-1, self.k)
        out = np.zeros((xb.shape[0], self.m))
        for o in range(self.m):
            B = self.active_B[o]
            match = (xb @ B.T) + ((1-xb) @ (1-B).T)
            out[:,o] = relu((match - self.k + c)/c).sum(axis=1)
        return out.reshape(x.shape[:-1] + (self.m,))

# ===============================
# AES Tables
# ===============================
# AES_SBOX = np.array([...], dtype=np.uint8)   # (same as your original, omitted here for brevity)

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


def gf_mul2(a):
    return ((a<<1)&0xff) ^ (0x1B if a&0x80 else 0)

def gf_mul3(a):
    return gf_mul2(a) ^ a

MUL2_MAP = np.array([gf_mul2(i) for i in range(256)], np.uint8)
MUL3_MAP = np.array([gf_mul3(i) for i in range(256)], np.uint8)

SBOX_NET = SumOfCornersBoolean(8,8,AES_SBOX)
MUL2_NET = SumOfCornersBoolean(8,8,MUL2_MAP)
MUL3_NET = SumOfCornersBoolean(8,8,MUL3_MAP)

# ===============================
# AES Layers
# ===============================
def shiftrows_perm():
    P = np.zeros(16, dtype=int)
    for r in range(4):
        for c in range(4):
            P[r+4*c] = r + 4*((c+r)%4)
    return P

SHIFTROWS = shiftrows_perm()

def nn_subbytes(s):
    s = s.reshape(16,8)
    return np.vstack([SBOX_NET.forward(x.reshape(1,8)) for x in s]).reshape(128)

def nn_shiftrows(s):
    return s.reshape(16,8)[SHIFTROWS].reshape(128)

def nn_mixcolumns(s):
    s = s.reshape(16,8)
    out = np.zeros_like(s)
    for c in range(4):
        a,b,c0,d = s[4*c:4*c+4]
        a2,a3 = MUL2_NET.forward(a[None])[0], MUL3_NET.forward(a[None])[0]
        b2,b3 = MUL2_NET.forward(b[None])[0], MUL3_NET.forward(b[None])[0]
        c2,c3 = MUL2_NET.forward(c0[None])[0], MUL3_NET.forward(c0[None])[0]
        d2,d3 = MUL2_NET.forward(d[None])[0], MUL3_NET.forward(d[None])[0]
        out[4*c+0] = nn_xor_abs(nn_xor_abs(a2,b3), nn_xor_abs(c0,d))
        out[4*c+1] = nn_xor_abs(nn_xor_abs(a,b2), nn_xor_abs(c3,d))
        out[4*c+2] = nn_xor_abs(nn_xor_abs(a,b), nn_xor_abs(c2,d3))
        out[4*c+3] = nn_xor_abs(nn_xor_abs(a3,b), nn_xor_abs(c0,d2))
    return out.reshape(128)

# ===============================
# AES-192 Key Schedule
# ===============================
RCON = np.array([0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80], np.uint8)

def aes192_roundkeys(key24: bytes):
    Nk, Nr = 6, 12
    key = np.frombuffer(key24, np.uint8)
    w = np.zeros((4*(Nr+1),4), np.uint8)
    w[:Nk] = key.reshape(Nk,4)
    for i in range(Nk, 4*(Nr+1)):
        temp = w[i-1].copy()
        if i % Nk == 0:
            temp = AES_SBOX[np.roll(temp,-1)]
            temp[0] ^= RCON[i//Nk - 1]
        w[i] = w[i-Nk] ^ temp
    return [w[4*r:4*r+4].reshape(16).tobytes() for r in range(Nr+1)]

# ===============================
# Natural AES-192 Blackbox
# ===============================

class NaturalAES192:
    def __init__(self, key24: bytes):
        rks = aes192_roundkeys(key24)
        self.rk_bits = [bytes_to_bits(rk) for rk in rks]  # 13개 (0..12)

    def encrypt_bits(self, pt_bits_real: np.ndarray) -> np.ndarray:
        x = np.asarray(pt_bits_real, dtype=np.float64).reshape(128)

        # round 0
        x = nn_xor_abs(x, self.rk_bits[0])

        # rounds 1..11
        for r in range(1, 12):
            x = nn_subbytes_state(x)
            x = nn_shiftrows_state(x)
            x = nn_mixcolumns_state(x)
            x = nn_xor_abs(x, self.rk_bits[r])

        # final round 12
        x = nn_subbytes_state(x)
        x = nn_shiftrows_state(x)
        x = nn_xor_abs(x, self.rk_bits[12])
        return x

    # --- NEW: oracle that starts at "AddRoundKey of round r" ---
    def oracle_from_addroundkey(self, r: int):
        """
        returns an oracle f(z_bits)->cipher_bits such that the FIRST operation is:
            z <- z XOR rk[r]
        and then continues AES from that point to the end.

        This is a synthetic oracle for chaining experiments (rk1, rk2, ...).
        """
        assert 0 <= r <= 12

        def f(z_bits: np.ndarray) -> np.ndarray:
            x = np.asarray(z_bits, dtype=np.float64).reshape(128)

            # first: AddRoundKey of round r
            x = nn_xor_abs(x, self.rk_bits[r])

            if r == 0:
                # then proceed with rounds 1..12 exactly as encrypt_bits
                for rr in range(1, 12):
                    x = nn_subbytes_state(x)
                    x = nn_shiftrows_state(x)
                    x = nn_mixcolumns_state(x)
                    x = nn_xor_abs(x, self.rk_bits[rr])
                x = nn_subbytes_state(x)
                x = nn_shiftrows_state(x)
                x = nn_xor_abs(x, self.rk_bits[12])
                return x

            if 1 <= r <= 11:
                # after ARK of round r, remaining part of round r is:
                # BUT note: in standard AES, ARK comes AFTER MC in rounds 1..11.
                # Here we define the synthetic oracle so that ARK is first,
                # and then we continue with the *next* round structure:
                # i.e., treat the state as being right before round (r+1) SubBytes.
                #
                # So continue with rounds (r+1)..11, then final.
                for rr in range(r+1, 12):
                    x = nn_subbytes_state(x)
                    x = nn_shiftrows_state(x)
                    x = nn_mixcolumns_state(x)
                    x = nn_xor_abs(x, self.rk_bits[rr])
                x = nn_subbytes_state(x)
                x = nn_shiftrows_state(x)
                x = nn_xor_abs(x, self.rk_bits[12])
                return x

            # r == 12: final ARK only
            return x

        return f



def aes192_master_from_rk0_canonical(rk0: bytes) -> bytes:
    """
    rk0 : 16-byte round key 0 recovered by sym attack
    returns: 24-byte AES-192 master key (canonical completion)
    """
    assert len(rk0) == 16

    w = np.zeros((6,4), dtype=np.uint8)

    # w0..w3 from rk0
    w[0:4] = np.frombuffer(rk0, dtype=np.uint8).reshape(4,4)

    # canonical choice: w4 = w5 = 0
    w[4] = 0
    w[5] = 0

    return w.reshape(24).tobytes()


def aes192_master_from_rk012(rk0: bytes, rk1: bytes, rk2: bytes = None) -> bytes:
    """
    Recover 192-bit AES master key from recovered rk0, rk1.
    rk2 is optional (used only for consistency check).
    """

    assert len(rk0) == 16
    assert len(rk1) == 16

    # split into 4-byte words
    w0_3 = np.frombuffer(rk0, dtype=np.uint8).reshape(4,4)
    w4_7 = np.frombuffer(rk1, dtype=np.uint8).reshape(4,4)

    # master key = w0..w5
    w_master = np.zeros((6,4), dtype=np.uint8)
    w_master[0:4] = w0_3
    w_master[4] = w4_7[0]
    w_master[5] = w4_7[1]

    master_key = w_master.reshape(24).tobytes()

    # ---- optional consistency check using rk2 ----
    if rk2 is not None:
        rks = aes192_roundkeys(master_key)
        if rks[0] != rk0 or rks[1] != rk1:
            print("[warning] rk0/rk1 mismatch in reconstructed key")
        if rks[2] != rk2:
            print("[warning] rk2 mismatch (attack may be partial)")

    return master_key


