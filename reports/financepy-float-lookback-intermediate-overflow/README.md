# Reproduce the FinancePy floating-lookback finding

This packet contains the actual pinned FinancePy package source, an executed official 1.1.2 wheel, a narrow candidate, the restored source, independent-oracle code, raw outputs, tests and a bounded duplicate review. It is one calculation defect shared by equity and FX APIs.

Python 3.12 was tested on macOS ARM64. Create TWO separate environments: current main requires Numba 0.67.x; release 1.1.2 requires Numba 0.62.x. Dependency installation needs ordinary PyPI access; after dependencies are installed the numerical scripts do not access the network. Do not install the packet globally or change an existing project environment.

```sh
python3.12 -m venv .venv-current
.venv-current/bin/python -m pip install -r requirements-current.txt
.venv-current/bin/python reproduce.py native current candidate restored
.venv-current/bin/python reproduce.py tests current candidate restored

python3.12 -m venv .venv-release
.venv-release/bin/python -m pip install -r requirements-release.txt
.venv-release/bin/python reproduce.py native release
.venv-release/bin/python reproduce.py tests release
.venv-release/bin/python reproduce.py oracle
```

Run commands sequentially. The harness sets numerical thread limits to 1 and uses CPU only. It writes generated outputs under `replay-work/`, leaving archived evidence unchanged. `--strict` additionally requires byte identity to the tested platform/runtime; small cross-platform finite rounding differences can require inspection even when fixed oracle checks pass. It is not a license to alter tolerances.

Expected: 600 rows per variant; 52/0/52/52 exceptions/nonfinite or out-of-tolerance results for current/candidate/restored/release. Current/restored outputs match; 548 current/candidate rows are identical. Four regressions fail on current/restored/release and pass candidate. Four existing upstream fixed-strike lookback tests are neighboring-product controls, not floating-case coverage. The independent oracle checks 420 calls; 180 puts are preservation controls.

`evidence/original` retains the initial NumPy 2.3.5/Numba 0.62.1 run and its fresh native replay. `evidence/current-supported` contains a separate Numba 0.67.0/llvmlite 0.49.0 check of current/candidate/restored source. The upgrade changes low bits in 140 finite rows but not the 52 target failures, fixed oracle classification, 548 within-environment preservation controls or restoration result. Do not claim cross-environment byte identity.

The included FinancePy source manifest identifies 222 original files plus the separate import asset in its receipt. Source snapshots are the package/build files needed for these calls, not a claim to a full upstream checkout/test suite. Candidate and original source notices remain. The wheel and extracted release package are included for audit; the replay imports the latter without installing FinancePy over the chosen environment.

The formula-to-document demonstration is deliberately synthetic GERO adapter code. Actual public FinancePy pricing produces an exception/NaN versus a finite corrected price; `synthetic-quote-*.json` records UNAVAILABLE/READY. It is not a real quote/order/trade or evidence of customer loss. See REPORT_EN.md for remaining stability and test limits.

Vendor-first disclosure: https://github.com/domokane/FinancePy/issues/272 . No acceptance claimed. The older duplicate-review note saying no issue had been sent describes the pre-disclosure stage; VENDOR_RECEIPT.json and the later prepublication receipt supersede that status.

Licenses: LICENSE-NOTES.md and source/LICENSE. Original GERO text CC BY 4.0; scripts and FinancePy-derived changes GPL-3.0-or-later. Preserve original notices.
