# Black-Scholes hybrid evaluator v3.0

Khamit Kadyrbekov and Daniyal Kadirbekov. Working paper, 26 September 2026.
Substantial AI assistance and prospective software roles are disclosed in the paper.
No independent human peer review, market improvement or historical novelty is claimed.

## Result

On 480 new frozen exact-binary64 synthetic inputs, v3 and v2 pass all declared
mixed price/log gates. Median detailed Python API cost is 19.668 vs 90.143 us,
a 4.583-fold speedup on this machine and mixture. Of 426 nonzero prices, v3
is closer to the reference than v2 in 76, farther in 99, tied in 251. The 68
fallback inputs cost about 2.85% more. This is not uniform accuracy superiority.

Read TECHNICAL_NOTE_V3.md, SUMMARY_RU_V3.md and EXPERIMENT_V3.md. Full raw
rows and timing batches are in evidence/. Earlier negative results are retained.
Source v2 remains frozen and is the fallback; no v2 publication was overwritten.

## Reproduce

Python 3.9+, mpmath 1.3.0, clang++ with C++11, one CPU worker, no GPU.
Use a fresh output path. No network is needed once dependencies are installed.

```bash
python3 -m unittest discover -s tests -v
python3 reproduce_v3.py --output /tmp/black-scholes-v3-reproduction.json
```

The portable harness was added after the original experiment. It validates
original source/input and mpmath hashes and compiles the native comparator
sequentially in a temporary directory. Only the source-verifier and native
binary location are adapted; the replay is explicitly labeled. Its same-host
replay matches all 480 original rows. This is not independent validation on
another operating system. Original receipts are preserved. Do not run the old
lab.build_jackel in this release: it overwrites the pinned historical receipt.
No executable binaries are included. The original native binary hash is kept
as provenance, with sources and notices for a separate build.

## Citation and licensing

Version DOI: https://doi.org/10.5281/zenodo.22973900
Predecessor: https://doi.org/10.5281/zenodo.22972343
Code/evidence: https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/black-scholes-hybrid-v3
Original text/data CC BY 4.0; original code MIT. Third-party Jaeckel/Cody code
retains its original notices and terms: vendor/jackel/PROVENANCE.md.
