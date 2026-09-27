# Reproducibility package

This package records a source-pinned numerical cancellation case in `amortization` 3.0.0/current `master` and a proposed stable equivalent formula.

Start with `REPORT_EN.md`, then run `reproduce.py` in an environment containing `amortization==3.0.0`. `candidate.patch` is the exact patch submitted in upstream PR #315. `decimal_grid_check.py` is independent of the package and checks the stable expression against an 80-digit Decimal reference across 640 cases.

The material uses synthetic inputs. It does not establish production frequency, customer loss or upstream acceptance.

