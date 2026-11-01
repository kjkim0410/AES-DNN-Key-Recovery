"""AES-256: FIPS-197 KAT, Boolean/ReLU correctness, synthetic-oracle key recovery.

Based on the supplied natural_aes_192_sym_attack_v3_poc.py and
natural_aes_all_core_sym.py. CSV files are always placed in ./results/.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from AES256_original_core import NaturalAES256, aes256_roundkeys, aes256_master_from_rk01
from AES256_original_core import bytes_to_bits, bits_to_bytes, keyrecovery_bits, AES_SBOX


def boolean_aes256_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Independent byte-oriented AES-256 rounds; shared key expansion with supplied code."""
    if len(plaintext) != 16 or len(key) != 32:
        raise ValueError('AES-256 requires 16-byte plaintext and 32-byte key')
    state = list(plaintext)
    roundkeys = aes256_roundkeys(key)

    def ark(r):
        for j, v in enumerate(roundkeys[r]):
            state[j] ^= v

    def shiftrows():
        old = state[:]
        for row in range(4):
            for col in range(4):
                state[4 * col + row] = old[4 * ((col + row) % 4) + row]

    def xtime(v):
        return ((v << 1) & 255) ^ (0x1b if v & 128 else 0)

    def mixcolumns():
        for col in range(4):
            a, b, c, d = state[4 * col:4 * col + 4]
            state[4*col+0] = xtime(a) ^ (xtime(b) ^ b) ^ c ^ d
            state[4*col+1] = a ^ xtime(b) ^ (xtime(c) ^ c) ^ d
            state[4*col+2] = a ^ b ^ xtime(c) ^ (xtime(d) ^ d)
            state[4*col+3] = (xtime(a) ^ a) ^ b ^ c ^ xtime(d)

    ark(0)
    for r in range(1, 15):
        state[:] = [int(AES_SBOX[b]) for b in state]
        shiftrows()
        if r != 14:
            mixcolumns()
        ark(r)
    return bytes(state)


def correctness(key, plaintext):
    reference = boolean_aes256_encrypt(plaintext, key)
    neural = NaturalAES256(key).encrypt_bits(bytes_to_bits(plaintext))
    exact = bool(np.array_equal(neural, bytes_to_bits(reference)))
    return exact, reference.hex(), bits_to_bytes(neural).hex(), float(np.max(np.abs(neural - bytes_to_bits(reference))))


def run(args):
    results = Path('results')
    results.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    kat_key = bytes.fromhex('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f')
    kat_pt = bytes.fromhex('00112233445566778899aabbccddeeff')
    kat_expected = '8ea2b7ca516745bfeafc49904b496089'
    ok, bc, nc, error = correctness(kat_key, kat_pt)
    print(f'FIPS-197 AES-256 KAT: {"PASS" if ok and bc == kat_expected else "FAIL"} Boolean={bc} ReLU={nc}', flush=True)
    if not ok or bc != kat_expected:
        return 1

    eq_path = results / 'aes256_equivalence_results.csv'
    eq_pass = 0
    with eq_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'plaintext_hex', 'key_hex', 'boolean_ct_hex', 'relu_ct_hex', 'exact_match', 'max_abs_error'])
        writer.writeheader()
        for trial in range(1, args.equiv_trials + 1):
            key, pt = rng.bytes(32), rng.bytes(16)
            ok, bc, nc, err = correctness(key, pt)
            writer.writerow(dict(trial=trial, plaintext_hex=pt.hex(), key_hex=key.hex(), boolean_ct_hex=bc, relu_ct_hex=nc, exact_match=ok, max_abs_error=err))
            f.flush()
            eq_pass += int(ok)
            print(f'Correctness {trial}/{args.equiv_trials}: {"PASS" if ok else "FAIL"}', flush=True)
            if not ok:
                print('STOP: ciphertext mismatch; key recovery not started.', flush=True)
                return 1
    print(f'Correctness: {eq_pass}/{args.equiv_trials} PASS; CSV={eq_path}', flush=True)

    rec_path = results / f'aes256_recovery_{args.recovery_trials}_results.csv'
    fields = ['trial', 'master_key_hex', 'recovered_key_hex', 'success', 'rk0_exact', 'rk1_exact', 'rk2_exact',
              'rk0_unresolved', 'rk1_unresolved', 'rk2_unresolved', 'rk0_loops', 'rk1_loops', 'rk2_loops',
              'elapsed_seconds', 'seed', 'epsilon', 'max_outer_loops', 'oracle_model']
    successes = 0
    start = time.perf_counter()
    with rec_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trial in range(1, args.recovery_trials + 1):
            key = rng.bytes(32)
            aes = NaturalAES256(key)
            actual_rks = aes256_roundkeys(key)
            t0 = time.perf_counter()
            recovered_rounds, unresolved, loops, exact_rounds = [], [], [], []
            for r, seed_offset in [(0, 3), (1, 7), (2, 11)]:
                oracle = aes.encrypt_bits if r == 0 else aes.oracle_from_addroundkey(r)
                recovered, nloops = keyrecovery_bits(oracle, nbits=128, eps=args.eps,
                    max_outer_loops=args.max_loops, rng_seed=args.seed + trial * 17 + seed_offset)
                missing = int(np.count_nonzero(recovered < 0))
                recovered_bytes = bits_to_bytes(recovered) if missing == 0 else None
                recovered_rounds.append(recovered_bytes)
                unresolved.append(missing)
                loops.append(nloops)
                exact_rounds.append(bool(recovered_bytes == actual_rks[r]))
            if all(rk is not None for rk in recovered_rounds):
                master = aes256_master_from_rk01(recovered_rounds[0], recovered_rounds[1])
                schedule_consistent = aes256_roundkeys(master)[:3] == recovered_rounds
                success = bool(master == key and schedule_consistent and all(exact_rounds))
                master_hex = master.hex()
            else:
                success, master_hex = False, 'UNRESOLVED'
            elapsed = time.perf_counter() - t0
            successes += int(success)
            writer.writerow(dict(trial=trial, master_key_hex=key.hex(), recovered_key_hex=master_hex,
                success=success, rk0_exact=exact_rounds[0], rk1_exact=exact_rounds[1], rk2_exact=exact_rounds[2],
                rk0_unresolved=unresolved[0], rk1_unresolved=unresolved[1], rk2_unresolved=unresolved[2],
                rk0_loops=loops[0], rk1_loops=loops[1], rk2_loops=loops[2], elapsed_seconds=f'{elapsed:.6f}',
                seed=args.seed, epsilon=args.eps, max_outer_loops=args.max_loops,
                oracle_model='r0:black-box;r1,r2:synthetic-intermediate-state'))
            f.flush()
            if trial <= 3 or trial % args.progress == 0 or not success:
                print(f'Recovery {trial}/{args.recovery_trials}: {"PASS" if success else "FAIL"}, '
                      f'rk_exact={exact_rounds}, unresolved={unresolved}, loops={loops}, time={elapsed:.2f}s', flush=True)
    print(f'SUMMARY: correctness {eq_pass}/{args.equiv_trials}; recovery {successes}/{args.recovery_trials} '
          f'({successes/args.recovery_trials:.2%}); elapsed={time.perf_counter()-start:.1f}s; CSV={rec_path}', flush=True)
    return 0 if successes == args.recovery_trials else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equiv-trials', type=int, default=10)
    parser.add_argument('--recovery-trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--max-loops', type=int, default=300)
    parser.add_argument('--progress', type=int, default=10)
    args = parser.parse_args()
    if min(args.equiv_trials, args.recovery_trials, args.max_loops, args.progress) < 1 or args.eps <= 0:
        parser.error('Counts and epsilon must be positive')
    raise SystemExit(run(args))
