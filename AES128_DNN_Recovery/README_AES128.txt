AES-128 Boolean/ReLU equivalence and DNN leakage recovery

Requirements: Python 3.10+, numpy.
Install: python -m pip install numpy

Files (keep together):
 AES128_original_core.py
 AES128_Boolean_ReLU_1000trials.py
 AES128_Equivalence_and_DNN_Recovery.py

Run 10 equivalence checks + 10 random-key recovery attacks:
 python AES128_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 10

Run 10 equivalence checks + 1000 random-key recovery attacks:
 python AES128_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 1000

Recovery can be computationally expensive. Results are written per trial to CSV.
If any equivalence comparison fails, recovery is not run.
Success requires ALL 128 recovered bits to equal the randomly generated secret key.
Unresolved key bits are never counted as success.
Each trial uses a distinct pseudorandom generator draw; fixed --seed makes experiments reproducible.
CSV contains test keys for research verification; do not use these as real security keys.

CSV OUTPUT DIRECTORY (updated):
 All CSV output files are saved under ./results/ relative to the working directory.
 The directory is created automatically if it does not exist.
 --output myfile.csv writes ./results/myfile.csv.
 --output elsewhere/myfile.csv also writes ./results/myfile.csv.
 Examples:
  python AES128_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 10
    -> ./results/aes128_recovery_10_results.csv
  python AES128_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 1000
    -> ./results/aes128_recovery_1000_results.csv
  python AES128_Boolean_ReLU_1000trials.py --trials 10
    -> ./results/aes128_equivalence_results.csv
