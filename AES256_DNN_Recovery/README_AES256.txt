AES-256 Boolean/ReLU equivalence + DNN key-recovery package
===========================================================
Requirements: Python 3.10+, numpy
Install: python -m pip install numpy

10-trial validation:
python AES256_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 10

1000-trial validation:
python AES256_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 1000

CSV outputs in ./results/ relative to the working directory:
  aes256_equivalence_results.csv
  aes256_recovery_10_results.csv
  aes256_recovery_1000_results.csv

Stage 1: FIPS-197 known-answer test and randomized Boolean-vs-ReLU ciphertext equality.
Stage 2: recover rk0 using standard black-box neural oracle; rk1/rk2 using
synthetic intermediate-state oracles (stronger assumption, NOT black-box AES).
Reconstruct master key from rk0||rk1, verify rk2 and full expanded schedule.
Each trial reports true key, recovered key, round-key accuracy, unresolved bits,
loops, runtime, parameters, and oracle model. Any correctness mismatch stops recovery.

Based on the user's natural_aes_256_core_sym.py, retained as AES256_original_core.py.
Note: Full 10/1000 key recovery runs are computationally intensive and have NOT
been validated to completion in the preparation environment. A short recovery
smoke run also exceeded its execution budget. Run locally before claiming success.
