"""AES-128: Boolean/ReLU equivalence first, then DNN leakage key recovery.
Requires AES128_original_core.py in same directory and numpy.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from AES128_original_core import NaturalAESBlackbox, bytes_to_bits, bits_to_bytes, keyrecovery_bits
from AES128_Boolean_ReLU_1000trials import boolean_aes128_encrypt, test_kat


def run(args):
    if args.equiv_trials < 1 or args.recovery_trials < 1 or args.progress < 1:
        raise ValueError('Trial counts and progress must be positive')
    if not (0 < args.c <= 1) or args.eps <= 0 or args.max_loops < 1:
        raise ValueError('Invalid c, eps, or max-loops')
    out = Path("results") / Path(args.output).name
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print('=== FIPS 197 Known Answer Test ===', flush=True)
    test_kat(args.c)
    print(f'=== Stage 1: Boolean/ReLU equivalence ({args.equiv_trials} trials) ===', flush=True)
    eq_success = 0
    for i in range(1, args.equiv_trials + 1):
        key, pt = rng.bytes(16), rng.bytes(16)
        boolean_ct = boolean_aes128_encrypt(pt, key)
        relu = NaturalAESBlackbox(key, args.c).encrypt_bits(bytes_to_bits(pt))
        exact = bool(np.array_equal(relu, bytes_to_bits(boolean_ct)))
        eq_success += exact
        print(f'Equivalence {i}/{args.equiv_trials}: {"PASS" if exact else "FAIL"} '
              f'Boolean={boolean_ct.hex()} ReLU={bits_to_bytes(relu).hex()}', flush=True)
        if not exact:
            print('STOP: Boolean/ReLU mismatch. Recovery stage not started.', flush=True)
            return 1
    print(f'Equivalence PASS: {eq_success}/{args.equiv_trials}', flush=True)
    print(f'=== Stage 2: DNN key recovery ({args.recovery_trials} random keys) ===', flush=True)
    fields = ['trial','master_key_hex','recovered_key_hex','success','unresolved_bits',
              'outer_loops','elapsed_seconds','seed','c','epsilon','max_outer_loops']
    successes = 0
    start = time.perf_counter()
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(1, args.recovery_trials + 1):
            key = rng.bytes(16)
            oracle = NaturalAESBlackbox(key, args.c).encrypt_bits
            t0 = time.perf_counter()
            recovered, loops = keyrecovery_bits(oracle, eps=args.eps,
                max_outer_loops=args.max_loops, rng_seed=args.seed + 100000 + i)
            elapsed = time.perf_counter() - t0
            unresolved = int(np.count_nonzero(recovered < 0))
            success = bool(unresolved == 0 and np.array_equal(recovered, bytes_to_bits(key)))
            successes += int(success)
            recovered_hex = bits_to_bytes(recovered).hex() if unresolved == 0 else 'UNRESOLVED'
            writer.writerow(dict(trial=i, master_key_hex=key.hex(),
                recovered_key_hex=recovered_hex, success=success,
                unresolved_bits=unresolved, outer_loops=loops,
                elapsed_seconds=f'{elapsed:.6f}', seed=args.seed,
                c=args.c, epsilon=args.eps, max_outer_loops=args.max_loops))
            f.flush()
            if i <= 3 or i % args.progress == 0 or not success:
                print(f'Recovery {i}/{args.recovery_trials}: {"PASS" if success else "FAIL"}, '
                      f'loops={loops}, unresolved={unresolved}, time={elapsed:.2f}s, '
                      f'key={key.hex()}, recovered={recovered_hex}', flush=True)
    print(f'=== Summary: equivalence {eq_success}/{args.equiv_trials}; '
          f'key recovery {successes}/{args.recovery_trials} '
          f'({successes / args.recovery_trials:.2%}); '
          f'elapsed {time.perf_counter() - start:.1f}s; CSV={out} ===', flush=True)
    return 0 if successes == args.recovery_trials else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equiv-trials', type=int, default=10)
    parser.add_argument('--recovery-trials', type=int, choices=[10, 1000], default=10)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--max-loops', type=int, default=200)
    parser.add_argument('--progress', type=int, default=10)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = f'aes128_recovery_{args.recovery_trials}_results.csv'
    raise SystemExit(run(args))
