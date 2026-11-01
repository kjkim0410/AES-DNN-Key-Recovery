"""AES-128 Boolean vs original sum-of-corners ReLU equivalence, plus optional key recovery.
Based on AES-128_key_recovery_with_3_options.py. No interactive input.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from AES128_original_core import (AES_SBOX, bytes_to_bits, bits_to_bytes,
    NaturalAESBlackbox, aes128_roundkeys, keyrecovery_bits)


def xtime(x):
    return ((x << 1) & 255) ^ (0x1b if x & 128 else 0)


def boolean_aes128_encrypt(pt: bytes, key: bytes) -> bytes:
    """Independent byte-oriented AES-128 reference (FIPS 197 column-major state)."""
    if len(pt) != 16 or len(key) != 16:
        raise ValueError('AES-128 requires 16-byte plaintext and key')
    s = list(pt)
    rks = aes128_roundkeys(key)
    def ark(r):
        for j, b in enumerate(rks[r]): s[j] ^= b
    def shiftrows():
        old = s[:]
        for row in range(4):
            for col in range(4):
                s[4*col+row] = old[4*((col+row)%4)+row]
    def mixcolumns():
        for col in range(4):
            a = s[4*col:4*col+4]
            s[4*col+0] = xtime(a[0]) ^ (xtime(a[1])^a[1]) ^ a[2] ^ a[3]
            s[4*col+1] = a[0] ^ xtime(a[1]) ^ (xtime(a[2])^a[2]) ^ a[3]
            s[4*col+2] = a[0] ^ a[1] ^ xtime(a[2]) ^ (xtime(a[3])^a[3])
            s[4*col+3] = (xtime(a[0])^a[0]) ^ a[1] ^ a[2] ^ xtime(a[3])
    ark(0)
    for r in range(1, 11):
        s[:] = [int(AES_SBOX[b]) for b in s]
        shiftrows()
        if r != 10: mixcolumns()
        ark(r)
    return bytes(s)


def test_kat(c):
    pt = bytes.fromhex('00112233445566778899aabbccddeeff')
    key = bytes.fromhex('000102030405060708090a0b0c0d0e0f')
    expected = bytes.fromhex('69c4e0d86a7b0430d8cdb78070b4c55a')
    boolean = boolean_aes128_encrypt(pt, key)
    neural = NaturalAESBlackbox(key, c).encrypt_bits(bytes_to_bits(pt))
    exact = np.array_equal(neural, bytes_to_bits(expected))
    ok = boolean == expected and exact
    print('FIPS-197 KAT: Boolean =', boolean.hex(), ' ReLU =', bits_to_bytes(neural).hex(), 'PASS =', ok)
    if not ok: raise AssertionError('FIPS-197 known-answer test failed')


def run(args):
    rng = np.random.default_rng(args.seed)
    test_kat(args.c)
    output = Path("results") / Path(args.output).name
    output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    recovery_passed = 0
    started = time.perf_counter()
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['trial','plaintext_hex','key_hex','boolean_ct_hex',
            'relu_ct_hex','equivalence_pass','max_abs_error','recovery_pass','unresolved_bits',
            'recovery_loops','elapsed_seconds'])
        writer.writeheader()
        for i in range(1, args.trials+1):
            t0 = time.perf_counter()
            pt = rng.bytes(16)
            key = rng.bytes(16)
            expected = boolean_aes128_encrypt(pt, key)
            oracle = NaturalAESBlackbox(key, args.c)
            neural = oracle.encrypt_bits(bytes_to_bits(pt))
            expected_bits = bytes_to_bits(expected)
            max_error = float(np.max(np.abs(neural - expected_bits)))
            # Compare raw neural output, not just thresholded bits.
            ok = bool(np.array_equal(neural, expected_bits))
            passed += int(ok)
            recovery = ''
            unresolved = ''
            loops = ''
            if args.recover:
                recovered, loops = keyrecovery_bits(oracle.encrypt_bits,
                    eps=args.eps, max_outer_loops=args.max_loops,
                    rng_seed=args.seed+i)
                unresolved = int(np.sum(recovered < 0))
                recovery = bool(unresolved == 0 and np.array_equal(recovered, bytes_to_bits(key)))
                recovery_passed += int(recovery)
            writer.writerow({'trial':i,'plaintext_hex':pt.hex(),'key_hex':key.hex(),
                'boolean_ct_hex':expected.hex(),'relu_ct_hex':bits_to_bytes(neural).hex(),
                'equivalence_pass':ok,'max_abs_error':max_error,
                'recovery_pass':recovery,'unresolved_bits':unresolved,
                'recovery_loops':loops,'elapsed_seconds':round(time.perf_counter()-t0,6)})
            f.flush()
            if i <= 3 or i % args.progress == 0 or not ok:
                print(f'Trial {i}/{args.trials}: equivalence={ok}, max_error={max_error:.3g}' +
                      (f', recovery={recovery}, unresolved={unresolved}' if args.recover else ''), flush=True)
    print(f'Equivalence: {passed}/{args.trials} ({passed/args.trials:.2%})')
    if args.recover: print(f'Key recovery: {recovery_passed}/{args.trials} ({recovery_passed/args.trials:.2%})')
    print(f'Elapsed: {time.perf_counter()-started:.1f} s; CSV: {output}')
    if passed != args.trials or (args.recover and recovery_passed != args.trials):
        raise SystemExit(1)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--c', type=float, default=0.5)
    p.add_argument('--output', default='aes128_equivalence_results.csv')
    p.add_argument('--progress', type=int, default=10)
    p.add_argument('--recover', action='store_true', help='Also run expensive 128-bit key-recovery attack per trial')
    p.add_argument('--eps', type=float, default=0.01)
    p.add_argument('--max-loops', type=int, default=200)
    args = p.parse_args()
    if args.trials < 1 or args.progress < 1 or not 0 < args.c <= 1: p.error('Invalid arguments')
    run(args)
