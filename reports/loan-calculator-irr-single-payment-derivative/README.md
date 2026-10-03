# Reproduce the loan-calculator IRR derivative report

Report: https://github.com/yanomateus/loan-calculator/issues/16

Unzip this folder. Python 3.12.13 was tested on macOS; a Unix-like platform with
`fcntl` and Python 3.10+ is required by this verifier. No installation, network,
NumPy/SciPy, GPU or paid service is needed. The original MIT-licensed wheel is
included and its checksum is checked before each run. The wheel is never edited.

Run sequentially from this directory:

```sh
python3 verify.py baseline --lock /tmp/gero-irr.lock
python3 verify.py patched --lock /tmp/gero-irr.lock
python3 verify.py restored --lock /tmp/gero-irr.lock
```

Each process extracts only package Python sources into a temporary directory.
The proposed change is applied only to the patched process's temporary copy.
The lock rejects concurrent use of the same computation slot. Use your existing
shared lock path instead when running inside a coordinated research environment.

Expected results: 35 checks per run; 22 / 0 / 22 failures respectively.
The original observations are retained under `evidence/`. New output JSON files
are written beside the verifier. Baseline/restored `groups` should be identical.
The 13 multi-payment IRR controls pass throughout. Counts are tests, not bugs.

See REPORT_EN.md for the independent references, exact 140-versus-240 witness,
version/source provenance, bounded novelty review, AI disclosure and limitations.
One defect report, two manifestations; maintainer acceptance is pending.

Original report: CC BY 4.0; original code: MIT. Retain the upstream MIT license
in vendor/LICENSE-loan-calculator.txt. SHA256SUMS covers all packaged artifacts.
