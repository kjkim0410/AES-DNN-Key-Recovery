import numpy as np
import time

# =========================
# Core ops
# =========================
def relu(x): return np.maximum(x, 0.0)

def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float64)
    return bits.reshape(-1)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits).reshape(-1, 8)
    v = (bits > 0.5).astype(np.uint8)
    out = (v * (1 << np.arange(7, -1, -1, dtype=np.uint8))).sum(axis=1).astype(np.uint8)
    return out.tobytes()

def bits_to_state(bits):
    # bits: (128,) → state[4][4][8]
    b = bits.reshape(16,8)
    state = np.zeros((4,4,8))
    for r in range(4):
        for c in range(4):
            state[r,c] = b[4*c + r]
    return state

def state_to_bits(state):
    b = np.zeros((16,8))
    for r in range(4):
        for c in range(4):
            b[4*c + r] = state[r,c]
    return b.reshape(128)


# =========================
# Definition 5/6: corner-sum network for Boolean maps
# =========================
class SumOfCornersBoolean:
    def __init__(self, k: int, m: int, mapping: np.ndarray):
        self.k = k
        self.m = m
        mapping = np.asarray(mapping)
        assert mapping.shape[0] == (1 << k)
        self.mapping = mapping.astype(np.uint32)
        self.active_B = []
        for o in range(m):
            active = np.where(((self.mapping >> (m - 1 - o)) & 1) == 1)[0]
            B = ((active[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.float64)
            self.active_B.append(B)

    def forward(self, x: np.ndarray, c: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[-1] == self.k
        xb = x.reshape(-1, self.k)
        out = np.zeros((xb.shape[0], self.m), dtype=np.float64)
        for o in range(self.m):
            B = self.active_B[o]
            term1 = xb @ B.T
            term2 = (1.0 - xb) @ (1.0 - B).T
            match = term1 + term2
            corner_vals = relu((match - self.k + c) / c)
            out[:, o] = corner_vals.sum(axis=1)
        return out.reshape(x.shape[:-1] + (self.m,))

# XOR truth table mapping for k=2 -> m=1, input bits (x1,x2) as index
# index = (x1<<1)|x2 ; XOR outputs: 0,1,1,0
XOR_MAP = np.array([0,1,1,0], dtype=np.uint8)
XOR_NET = SumOfCornersBoolean(k=2, m=1, mapping=XOR_MAP)

def nn_xor_cornersum_bits(x_bits: np.ndarray, k_bits: np.ndarray, c: float) -> np.ndarray:
    """
    Natural XOR via sum-of-corners (k=2,m=1) applied bitwise.
    x_bits,k_bits shape (128,), returns shape (128,) real.
    """
    x_bits = np.asarray(x_bits, dtype=np.float64).reshape(-1)
    k_bits = np.asarray(k_bits, dtype=np.float64).reshape(-1)
    assert x_bits.shape[0] == k_bits.shape[0]
    pairs = np.stack([x_bits, k_bits], axis=1)  # (n,2)
    y = XOR_NET.forward(pairs, c=c).reshape(-1)  # (n,)
    return y

# =========================
# AES tables + natural implementations
# =========================
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
    a &= 0xFF
    return ((a << 1) & 0xFF) ^ (0x1B if (a & 0x80) else 0)

def gf_mul3(a: int) -> int:
    return gf_mul2(a) ^ (a & 0xFF)

MUL2_MAP = np.array([gf_mul2(i) for i in range(256)], dtype=np.uint8)
MUL3_MAP = np.array([gf_mul3(i) for i in range(256)], dtype=np.uint8)

SBOX_NET = SumOfCornersBoolean(8, 8, AES_SBOX.astype(np.uint32))
MUL2_NET = SumOfCornersBoolean(8, 8, MUL2_MAP.astype(np.uint32))
MUL3_NET = SumOfCornersBoolean(8, 8, MUL3_MAP.astype(np.uint32))

def aes_shiftrows_perm():
    P = np.zeros(16, dtype=np.int32)
    for r in range(4):
        for c in range(4):
            src_c = (c + r) % 4
            dst = r + 4*c
            src = r + 4*src_c
            P[dst] = src
    return P

SHIFTROWS_PERM = aes_shiftrows_perm()

def permute_bytes(state_bits: np.ndarray, byte_perm: np.ndarray) -> np.ndarray:
    s = np.asarray(state_bits, dtype=np.float64).reshape(16, 8)
    return s[byte_perm].reshape(128)

def nn_apply_8to8(net, byte_bits, c):
    x = np.asarray(byte_bits, dtype=np.float64).reshape(1, 8)
    return net.forward(x, c=c).reshape(8)

def nn_subbytes_state(state_bits, c):
    s = np.asarray(state_bits, dtype=np.float64).reshape(16, 8)
    out = np.zeros_like(s)
    for i in range(16):
        out[i] = nn_apply_8to8(SBOX_NET, s[i], c)
    return out.reshape(128)

# Old 
# def nn_shiftrows_state(state_bits):
#     return permute_bytes(state_bits, SHIFTROWS_PERM)

#################
## New (26.2.10)
#################
def nn_shiftrows_state(bits):
    s = bits_to_state(bits)
    for r in range(4):
        s[r] = np.roll(s[r], -r, axis=0)
    return state_to_bits(s)


def xor4_chain(x1, x2, x3, x4, c):
    t = nn_xor_cornersum_bits(x1, x2, c)
    t = nn_xor_cornersum_bits(t,  x3, c)
    t = nn_xor_cornersum_bits(t,  x4, c)
    return t

######################
# nn_mixcolumn function
#######################

def nn_mixcolumns_state(bits, c):
    s = bits_to_state(bits)
    out = np.zeros_like(s)

    for col in range(4):
        a,b,c0,d = s[:,col]

        a2,a3 = nn_apply_8to8(MUL2_NET,a,c), nn_apply_8to8(MUL3_NET,a,c)
        b2,b3 = nn_apply_8to8(MUL2_NET,b,c), nn_apply_8to8(MUL3_NET,b,c)
        c2,c3 = nn_apply_8to8(MUL2_NET,c0,c),nn_apply_8to8(MUL3_NET,c0,c)
        d2,d3 = nn_apply_8to8(MUL2_NET,d,c), nn_apply_8to8(MUL3_NET,d,c)

        out[0,col] = xor4_chain(a2,b3,c0,d,c)
        out[1,col] = xor4_chain(a,b2,c3,d,c)
        out[2,col] = xor4_chain(a,b,c2,d3,c)
        out[3,col] = xor4_chain(a3,b,c0,d2,c)

    return state_to_bits(out)


# =========================
# AES-128 key schedule (embed key)
# =========================
RCON = np.array([0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36], dtype=np.uint8)
def rot_word(w4): return np.roll(w4, -1)
def sub_word(w4): return AES_SBOX[w4]

def aes128_roundkeys(key16: bytes):
    key = np.frombuffer(key16, dtype=np.uint8)
    w = np.zeros((44,4), dtype=np.uint8)
    w[0:4] = key.reshape(4,4)
    for i in range(4, 44):
        temp = w[i-1].copy()
        if i % 4 == 0:
            temp = sub_word(rot_word(temp))
            temp[0] ^= RCON[(i//4)-1]
        w[i] = w[i-4] ^ temp
    rks = []
    for r in range(11):
        rks.append(w[4*r:4*r+4].reshape(16).tobytes())
    return rks

# =========================
# Blackbox AES natural DNN oracle (ciphertext only)
# XOR uses corner-sum with parameter c (<1 case)
# =========================
class NaturalAESBlackbox:
    def __init__(self, key16: bytes, c: float):
        assert len(key16) == 16
        assert 0 < c <= 1.0
        self.c = float(c)
        rks = aes128_roundkeys(key16)
        self.rk_bits = [bytes_to_bits(rk) for rk in rks]  # embedded

    def encrypt_bits(self, pt_bits_real: np.ndarray) -> np.ndarray:
        x = np.asarray(pt_bits_real, dtype=np.float64).reshape(128)

        # ARK0: XOR(p, k^1) bitwise
        x = nn_xor_cornersum_bits(x, self.rk_bits[0], c=self.c)

        # rounds 1..9
        for r in range(1, 10):
            x = nn_subbytes_state(x, c=self.c)
            x = nn_shiftrows_state(x)
            x = nn_mixcolumns_state(x, c=self.c)
            x = nn_xor_cornersum_bits(x, self.rk_bits[r], c=self.c)

        # final round
        x = nn_subbytes_state(x, c=self.c)
        x = nn_shiftrows_state(x)
        x = nn_xor_cornersum_bits(x, self.rk_bits[10], c=self.c)

        return x

# =========================
# Algorithm 6: GenPairs^sym(p,i,ε)
# =========================
def genpairs_sym(p: np.ndarray, i: int, eps: float):
    x0 = p.copy(); x0[i]  = -eps
    x0p = p.copy(); x0p[i] = +eps
    x1 = p.copy(); x1[i]  = 1.0 - eps
    x1p = p.copy(); x1p[i] = 1.0 + eps
    return {0: (x0, x0p), 1: (x1, x1p)}


# =========================
# Algorithm 4: KeyRecovery (bits)
# =========================
def keyrecovery_bits(oracle, eps=0.01, max_outer_loops=200, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    remaining = list(range(128))
    k_hat = np.full((128,), -1, dtype=np.int32)
    np_used = 0

    while remaining and np_used < max_outer_loops:
        # choose random base plaintext p in {0,1}^128
        p = rng.integers(0, 2, size=128).astype(np.float64)
        np_used += 1

        new_remaining = []
        for i in remaining:
            cand = []
            pairs = genpairs_sym(p, i, eps)
            # pairs = genpairs_change(p, i, eps)
            for v in (0,1):
                x, xp = pairs[v]
                c1 = oracle(x)
                c2 = oracle(xp)
                if np.allclose(c1, c2, atol=1e-12, rtol=0.0):
                    cand.append(v)
            if len(cand) == 1:
                k_hat[i] = cand[0]
            else:
                new_remaining.append(i)
        remaining = new_remaining

    return k_hat, np_used


def generate_random_key():
        rng = np.random.default_rng()
        key_bytes = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        return np.frombuffer(key_bytes, dtype=np.uint8)

## random key for Known Answer test in FIPS 197 
def select_fixed_key():
    print("== random key for KAT in FIPS 197 ==")
    key_bytes = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    return np.frombuffer(key_bytes, dtype=np.uint8)


def select_manual_key():
    while True:
        user_input = input("Enter 32 hex chars (16 bytes): ").strip()
        try:
            if len(user_input) != 32:
                raise ValueError("Must be exactly 32 hex characters.")
            key_bytes = bytes.fromhex(user_input)
            return np.frombuffer(key_bytes, dtype=np.uint8)
        except Exception as e:
            print("Invalid input:", e)

########################
# ##Flag for printing for kkj
DEBUG_kkj = True
#######################

def run_attack_once(key):

# 여기에 실제 공격 코드 삽입

    print("== Started !!==")

    start_time = time.perf_counter()   # start timer

    oracle = NaturalAESBlackbox(key, c=0.5).encrypt_bits
    k_hat, loops = keyrecovery_bits(oracle)

    true_bits = bytes_to_bits(key)

    end_time = time.perf_counter()     # end timer
    elapsed = end_time - start_time

    print("== Ended   !!==")
    print(f"Elapsed time: {elapsed:.6f} seconds")
    
    if DEBUG_kkj:
        print(f"Recovered key(hex)=", bits_to_bytes(k_hat).hex())
 
    return True  # 예시용


if __name__ == "__main__":

# ===============================
# Example usage
# ===============================
# option:
#         0 -> random 16-byte key (mode = 1 for single, or 2 for multiple execution)
#         1 -> fixed test vector key (00..0f)
#         2 -> user input 16-byte hex key 

    print("+++++++++++++++++++++++++++++++++++")
    print("AES-128 Key Recovery by DNN leakage")
    print("+++++++++++++++++++++++++++++++++++")
    print("Select option using Genpairs_sym:")
    print("0 -> Random key")
    print("1 -> Fixed key")
    print("2 -> Manual input key")

    option = int(input("Enter option: "))

    # --------------------------
    # OPTION 0 (Random key)
    # --------------------------
    if option == 0:

        print("\nRandom key mode:")
        print("1 -> Single execution")
        print("2 -> Multiple execution")

        mode = int(input("Select mode (1 or 2): "))

        if mode == 1:
            # Single run
            key = generate_random_key()
            print("Selected key (hex)=", key.tobytes().hex())
            run_attack_once(key)

        elif mode == 2:
            # Multiple runs
            repeat = int(input("Enter repeat count: "))
            success = 0

            for r in range(repeat):
                print(f"\n=== Run {r+1}/{repeat} ===")
                key = generate_random_key()

                print("Selected key (hex)=", key.tobytes().hex())

                ok = run_attack_once(key)

                if ok:
                    success += 1

            print("\n===== Summary =====")
            print("Total runs:", repeat)
            print("Success:", success)
            print("Success rate:", success / repeat)

        else:
            print("Invalid mode.")

    # --------------------------
    # OPTION 1 (Fixed key)
    # --------------------------
    elif option == 1:
        key = select_fixed_key()
        if DEBUG_kkj:
            print("Selected key (hex)=", key.tobytes().hex())
        run_attack_once(key)

    # --------------------------
    # OPTION 2 (Manual key)
    # --------------------------
    elif option == 2:
        key = select_manual_key()
        if DEBUG_kkj:
            print("Selected key (hex)=", key.tobytes().hex())
        run_attack_once(key)

    else:
        print("Invalid option.")
