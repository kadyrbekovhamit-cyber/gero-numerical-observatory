# penaltyblog partial-hedge payoff audit

Authors: Xamit Kadirbekov and Daniyal Kadirbekov. AI-assisted investigation.

Read REPORT_EN.md for the finding, limits and exact cash ledger. The pinned upstream module is unchanged. No complete strategy patch or maintainer acceptance is claimed.

## Replay

Python 3.9+ with SciPy 1.13.1 was used. In a suitable environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 verify.py
```

This overwrites evidence/REPLAY.json with the new environment and result. The archived replay is the original observed run: five mismatches among six deliberately selected partial-mode cases; the zero-exposure and one full-hedge controls pass. It is not a prevalence estimate or a whole-package integration test.

Report: CC BY 4.0. Original verification code: MIT. The unchanged upstream module retains the licence in vendor/LICENCE.

Maintainer report: https://github.com/martineastwood/penaltyblog/issues/50

No real bets, measured customer losses, reward, external peer review or worldwide priority are claimed.
