# Source ledger

- Source and index: https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/dropout.cpp — implementation fact at an immutable revision.
- Added tests: `evidence/patches/tests.patch` — native reproduction and finite-difference reference, not third-party validation.
- Original and patched measurements: `evidence/before.xml`, `evidence/after.xml`, `evidence/results.json` — measured counts.
- Publication-day native run: `prepublication-check/full-results.xml` and `full-runtime-authorized.log` — 19 selected tests, zero failures.
- Historical code: https://github.com/nntrainer/nntrainer/pull/1640 — constant-zero destination appears in a 2021 public patch; discovery priority is not claimed.
- Model/device effects: untested; no extrapolation from the layer result.
