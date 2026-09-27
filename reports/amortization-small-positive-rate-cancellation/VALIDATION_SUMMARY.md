# Validation summary

- Upstream: `roniemartinez/amortization`
- Version: `3.0.0`
- Rechecked commit: `df32787c23d1e9d721eb302559dcea06b988b9ab`
- Python: `3.12.13`
- Baseline targeted result: 2 failures, 4 passes
  - annual rate `1e-15`: `ZeroDivisionError`
  - annual rate `1e-12`: `278.00` rather than Decimal reference `277.78`
- Patched targeted result: 6 passes
- Restored baseline result: the same 2 failures return
- Patched full suite: 22 passes, 100% project coverage
- Project pre-commit hooks: all pass
- Independent 80-digit Decimal grid: 640 combinations, 0 cent-level mismatches
- Limitation: production prevalence and downstream customer impact were not measured.

The separate negative-balance schedule behavior is excluded because prior PR #237 already documents that topic.
