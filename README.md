# AES-DNN-Key-Recovery

**Research code and reproducibility package for Boolean/ReLU equivalence and internal-activation-leakage key-recovery experiments on AES-128, AES-192, and AES-256.**

## Overview

This repository contains Python proof-of-concept experiments for three AES key lengths. Each package tests agreement between conventional Boolean AES and a ReLU-based realization, then runs a key-recovery experiment under the explicitly defined neural leakage/oracle model. The experiments are **not attacks against conventional AES encryption through its standard public interface**.

## Experimental results included

| Variant | Supplied 1,000-trial CSV | Successful trials |
| --- | --- | --- |
| AES-128 | `AES128_DNN_Recovery/results/aes128_recovery_1000_results.csv` | 1,000 / 1,000 |
| AES-192 | `AES192_DNN_Recovery/results/aes192_recovery_1000_results.csv` | 1,000 / 1,000 |
| AES-256 | `AES256_DNN_Recovery/results/aes256_recovery_1000_results.csv` | 1,000 / 1,000 |

The counts above were calculated from the **supplied CSV files** (the `success` column); they do not represent an independent rerun or audit. Results are conditional on the experiment's leakage/oracle assumptions.

## Repository structure

```text
AES-DNN-Key-Recovery/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── AES128_DNN_Recovery/
│   ├── AES128_original_core.py
│   ├── AES128_Boolean_ReLU_1000trials.py
│   ├── AES128_Equivalence_and_DNN_Recovery.py
│   ├── README_AES128.txt
│   └── results/ (... supplied CSVs)
├── AES192_DNN_Recovery/
│   ├── AES192_original_core.py
│   ├── natural_aes_all_core_sym.py
│   ├── AES192_Equivalence_and_DNN_Recovery.py
│   ├── README_AES192.txt
│   └── results/ (... supplied CSVs)
└── AES256_DNN_Recovery/
    ├── AES256_original_core.py
    ├── AES256_Equivalence_and_DNN_Recovery.py
    ├── README_AES256.txt
    └── results/ (... supplied CSVs)
```

## Requirements

- Python 3.10 or later
- NumPy

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Running the experiments

Run commands from the relevant variant directory, so that CSV files appear in its local `results/` directory.

### AES-128

```powershell
cd AES128_DNN_Recovery
python AES128_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 10
python AES128_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 1000
cd ..
```

### AES-192

```powershell
cd AES192_DNN_Recovery
python AES192_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 10
python AES192_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 1000
cd ..
```

### AES-256

```powershell
cd AES256_DNN_Recovery
python AES256_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 10
python AES256_Equivalence_and_DNN_Recovery.py --equiv-trials 10 --recovery-trials 1000
cd ..
```

For the separate AES-128 Boolean/ReLU equivalence campaign:

```powershell
cd AES128_DNN_Recovery
python AES128_Boolean_ReLU_1000trials.py --trials 1000
```

Long recovery campaigns can take substantial time. For Windows systems with multiple Python installations, use the intended Python 3.10 interpreter explicitly if `python` resolves to a different environment.

## Security model and limitations

- These experiments assume access to neural/internal-activation information or synthetic oracles beyond normal AES encryption/decryption APIs.
- For **AES-192 and AES-256**, the supplied implementations use black-box access for an initial round and **idealized synthetic intermediate-state oracles** for later round-key recovery. These oracles are stronger than ordinary black-box access.
- The provided scripts are proof-of-concept research demonstrations, not practical attacks on standard AES implementations, and do not establish that AES-128/192/256 are cryptographically broken.
- CSV files include randomly generated experimental secret keys for reproducibility. Never use them as operational secrets.
- See the variant-specific README files for implementation details and validation scope.

## Reproducibility

The `results/` directories contain the CSVs supplied with the original package, including 10- and 1,000-trial recovery runs. Scripts include seed and experiment parameter options; see each script's `--help` and the corresponding README. Equivalence checking precedes recovery and is designed to stop on a mismatch.

## Citation

When referring to this software in a paper, cite the associated publication (once available) and provide the repository URL. Suggested wording:

> The Python implementations and experimental CSV results for AES-128, AES-192, and AES-256 are available at https://github.com/kjkim0410/AES-DNN-Key-Recovery.

## License

MIT License. See [LICENSE](LICENSE).
