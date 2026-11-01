AES-192 Boolean / ReLU correctness + DNN leakage master-key recovery
=====================================================================
Requirements: Python 3.10+, numpy

Files:
  AES192_Equivalence_and_DNN_Recovery.py  (main)
  AES192_original_core.py                (derived from user's original AES-192 PoC)
  natural_aes_all_core_sym.py            (user's original ReLU and recovery core)

Install:
  python -m pip install numpy

Run a 10-trial experiment (first 10 correctness checks, then 10 random-key recoveries):
  python AES192_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 10

Run 1000 recovery trials:
  python AES192_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 1000

All CSV outputs are created in ./results/ relative to current working directory:
  results/aes192_equivalence_results.csv
  results/aes192_recovery_10_results.csv
  results/aes192_recovery_1000_results.csv

The correctness file is overwritten each run; recovery files are named by trial count.
Results are flushed after each trial, so completed trials survive interruption.
The KAT is FIPS-197 AES-192: PT 00112233445566778899aabbccddeeff;
KEY 000102030405060708090a0b0c0d0e0f1011121314151617;
CT dda97ca4864cdfe06eaf70a0ec0d7191.

IMPORTANT SECURITY SCOPE:
  Round 0 uses standard black-box encryption access.
  Rounds 1 and 2 use an IDEALIZED synthetic intermediate-state oracle;
  these are NOT available in a normal AES encryption API.
  Therefore successful recovery is conditional on this stronger oracle model.

Validation performed for this package:
  Python syntax check, FIPS-197 AES-192 KAT, one random Boolean/ReLU
  exact ciphertext comparison. Full 10/1000-key recovery runs not completed.
  WARNING: original 192-bit key-recovery algorithm is very computationally
  expensive in pure Python. Run with enough time and use progress CSVs.
