# Black-Scholes interval implied volatility, v5.0

**Khamit Kadyrbekov and Daniyal Kadirbekov**, 26 September 2026.
Working paper: https://doi.org/10.5281/zenodo.22976977
Predecessor: https://doi.org/10.5281/zenodo.22976212

Research utility for the set of BSM volatilities compatible with an exact
price interval or binary64 nearest-even rounding cell. Exact rational root
brackets and Arb sign witnesses. No market-model, universal performance,
independent formal verification, novelty or production-readiness claim.
Read TECHNICAL_NOTE_V5.md, SUMMARY_RU_V5.md and AUTHORSHIP.md.

The rounded ITM call100 example gives sigma from zero to about8.9407687355%.
Its upper endpoint is bracketed with width<1e-24. This is conditional on the
Arb enclosure contract and this implementation. Other parameters are exact.
`resolved` describes root-location tolerance, not economic identifiability.

## Evidence with the failed experiments retained

One unchanged solver, three disjoint pre-output-frozen corpora:

| Corpus | All original gates pass | Interpretation |
|---|---|---|
| IV v1 | 158/160 | Generator subnormal conversion and reference/expectation issues |
| IV v2 | 82/96 | Reference log/exp lost exact boundary identities |
| IV v3 | 64/64 | 56 resolved +8 deliberately insufficient budgets |

V3 has417 retained definite witnesses passing separate180/260-digit controls.
Rechecking1781 earlier witnesses with the corrected reference passes, but
is labelled **post hoc**, not new confirmation. Earlier scores stay unchanged.
24 solver test methods and3 reference identity methods pass. All development
failures, protocols, exact inputs, freezes, outputs and diagnoses are included.
The four illustrations are post-confirmation examples, not additional gates.

## Reproduce

Python3.12, python-flint==0.8.0, mpmath==1.3.0. Install the pinned requirements
in an isolated environment. Native dependency binaries are not bundled.

```bash
python -m pip install --only-binary=:all: -r requirements-iv.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s iv-tests -v
python -m unittest discover -s iv-tests-v3 -v
python -m lab.run_iv_v3 --portable --output /tmp/iv3-replay.json
python iv_range.py --spot 200 --strike 100 --time 1 --rounded-price 100
python iv_range.py --spot 100 --strike 100 --time 1 --bid 7 --ask 9
```

Always use fresh output paths. Compare `rows`, `summary` and `symmetry` against
iv-evidence/confirmation-v3.json. The portable runner checks frozen source
and input hashes, dependency versions and mpmath source hashes, allowing
platform-specific python-flint binaries. The original native hashes are
retained. Run sequentially: python-flint precision context is global.

To reproduce the earlier experiments, replace `lab.run_iv_v3` with `lab.run_iv`
or `lab.run_iv_v2` and use fresh paths. Their exit status1 is expected because
the original failed gates are deliberately retained. Corresponding outputs
must match confirmation-v1.json and confirmation-v2.json. Same-host archive
replay is not independent review or cross-platform validation.

The original package initializer imports an older price module, included for
compatibility; the interval solver does not call that price implementation.
BOOK_CHAPTER_9.md is an extract from an unfinished Russian teaching manuscript,
not a complete book or an endorsed MSU publication.

Text/data: CC BY4.0. Original code: MIT. See AUTHORSHIP.md for roles and
substantial AI assistance. MANIFEST.json gives every release-file SHA256,
except itself. No credentials or native dependency packages are included.
