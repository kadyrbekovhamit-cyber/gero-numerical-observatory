# amortization 3.0.0: small positive rates can erase interest or alter the rounded payment

Xamit Kadirbekov · Independent GERO research · 27 September 2026

**The public Python package `amortization` loses sufficiently small positive periodic rates while evaluating its annuity formula.** With principal `100000`, 360 monthly periods and annual rate `1e-15`, version 3.0.0 and current `master` at `df32787c23d1e9d721eb302559dcea06b988b9ab` raise `ZeroDivisionError`. At rate `1e-12`, they return `278.00`; an 80-digit Decimal evaluation of the same formula rounds to `277.78`.

The maintainer asked for a public GitHub report and, if possible, a pull request. [Issue #314](https://github.com/roniemartinez/amortization/issues/314) and [PR #315](https://github.com/roniemartinez/amortization/pull/315) are open. The pull request is a proposal and is not claimed as accepted or merged.

## Formula → changed schedule → measured consequence

For principal `P`, periodic rate `i` and `n` payments, the regular payment is

```text
A = P i (1+i)^n / ((1+i)^n - 1).
```

The implementation first evaluates `1 + i`. When `i` is very small, binary64 rounds that sum to `1`. The later exponent and subtraction therefore cannot recover the lost increment.

The library's schedule generator uses the returned regular amount for the first 359 rows and adjusts the final row. In the executed synthetic schedule:

| Annual rate | Current result | Stable candidate | Measured schedule consequence |
|---:|---:|---:|---|
| `1e-15` | `ZeroDivisionError` | `277.78` | current code cannot generate the 360-row schedule |
| `1e-12` | `278.00` | `277.78` | each of the first 359 amounts changes by `0.22`; the final amount changes from `198.00` to `276.98`, a `78.98` shift |

The synthetic schedule's total remains `100000.00` because per-period interest rounds to zero and the implementation adjusts the last payment. The measured effect is a changed schedule shape, not a demonstrated customer loss.

## Equivalent stable form

The same annuity formula can be written as

```text
A = P i / (1 - (1+i)^(-n)).
```

The proposed implementation retains the small increment through `log1p` and `expm1`:

```python
discount = -expm1(-period * log1p(adjusted_interest))
return round(principal * adjusted_interest / discount, 2)
```

This is the code in [PR #315](https://github.com/roniemartinez/amortization/pull/315).

## Executed implementation and independent reference

- Package: `amortization` 3.0.0.
- Current source rechecked: [`df32787c23d1e9d721eb302559dcea06b988b9ab`](https://github.com/roniemartinez/amortization/tree/df32787c23d1e9d721eb302559dcea06b988b9ab).
- Runtime: CPython 3.12.13 on macOS, one configured numerical worker.
- Reference: the same annuity formula evaluated with Python `Decimal` at 80 digits, then rounded to cents.
- Focused original/candidate/restored comparison: `2 → 0 → 2` failures across the two added small-rate cases.
- Full patched project suite: 22 tests passed with 100% project coverage.
- Project `pre-commit`: end-of-file, Ruff formatting/lint and mypy all passed.
- Independent grid: 640 combinations of principal, rate, term and frequency; zero cent-level mismatches between the candidate and the Decimal reference.

The independent grid included principals from `1` to `10000000`, annual rates from `1e-15` to `1.0`, periods from `1` to `1200`, and frequencies `1`, `12`, `24` and `52`.

## Duplicate review

Before the GitHub submission, the review covered the repository's public issues, pull requests and comments retained in the evidence package, current source history, targeted searches for `1e-15`, `1e-12`, `ZeroDivisionError`, `log1p` and `expm1`, and the GERO register. The final targeted searches returned zero exact matches.

Related public work is distinct:

- Issue #201 discusses using Decimal more generally.
- PR #249 added the exact zero-rate branch.
- PR #237 concerns negative schedule balances and is excluded from this report.

No absolute priority claim is made.

## Reproduction

Install the pinned package or check out the pinned commit, then run `reproduce.py`. Apply `candidate.patch` and rerun. The archive also contains the original/candidate/restored test logs, JUnit results, the 640-case Decimal grid, schedule measurements, current-source receipt, duplicate-search responses and GitHub submission receipts.

## Limits

- Only the public Python implementation was executed; no bank, lender or customer system was examined.
- The rates are deliberate numerical stress cases. Their production frequency was not measured.
- The schedule comparison is synthetic and uses the package's own cent-rounding behavior.
- No customer loss, contractual error, regulatory breach or production exposure is claimed.
- PR #315 remains open at publication time; upstream acceptance is not claimed.

## Disclosure

Independent GERO research by Xamit Kadirbekov. AI-assisted experiment and archival preparation. The maintainer was contacted privately first and asked that the report be moved to GitHub; the public issue and proposed correction were submitted before this publication.

<!-- GERO_PUBLICATION_LINKS_BEGIN -->
## Verified publication and reproduction links

- [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/amortization-small-positive-rate-cancellation.md)
- [Upstream issue #314](https://github.com/roniemartinez/amortization/issues/314)
- [Proposed upstream correction #315](https://github.com/roniemartinez/amortization/pull/315)
- [Complete reproducibility ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/amortization-small-positive-rate-cancellation/gero-amortization-small-positive-rate-evidence-2026-09-27.zip)

Archive SHA-256: `3c6de391f39322022b61f84b256046349a10ab70e2505f88d5c63f08cbe78f27`.

The maintainer requested a GitHub issue and optional PR before publication. Both are public; acceptance or merge is not claimed. Other platform links are pending verification.
<!-- GERO_PUBLICATION_LINKS_END -->
