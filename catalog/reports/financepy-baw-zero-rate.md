> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/gero-financepy-baw-zero-rate-audit/blob/main/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# FinancePy Barone–Adesi–Whaley zero-rate audit

Status: correction submitted upstream as [FinancePy PR #261](https://github.com/domokane/FinancePy/pull/261). The PR is not merged at the time of this record.

## Finding

The Barone–Adesi–Whaley path evaluates

```text
(2r/v²) / (1-exp(-rt)).
```

The quotient has the finite limit `2/(v²t)` as `r → 0`, but the released implementation first forms its numerator and denominator. At `r=0` both become zero and pricing raises `ZeroDivisionError`. Direct subtraction also degrades the small-rate calculation; a synthetic put at `r=1e-12` exceeds the 50-iteration secant limit.

The result is reproduced in the official `financepy-1.0.1-py3-none-any.whl`, SHA-256 `110e784122f485ae207239f44af473334f418d81bcbb958f5eda4e76956feca5`.

## Synthetic reproducer

All cases use `S=K=100`, `T=1`, and `sigma=20%`.

| Case | Released wheel | Corrected branch |
|---|---:|---:|
| American call, `r=0`, `q=2%` | `ZeroDivisionError` | `7.098257767528` |
| American call, `r=1e-12`, `q=2%` | `7.098254338311` | `7.098257870446` |
| American put, `r=0`, `q=0` | `ZeroDivisionError` | `7.965579241666` |
| American put, `r=1e-12`, `q=0` | failed to converge | `7.965579241612` |

Run `reproduce.py` once with the released wheel on `PYTHONPATH`, then with PR #261.

## Independent controls

`independent_oracle.py` imports no FinancePy code. A 2,000-step CRR tree gives `7.110251189598` for the zero-rate dividend-paying American call. The corrected BAW approximation differs by about `0.012`, while remaining continuous across zero. For `r=q=0`, early exercise adds no value to the put; the independent European Black–Scholes control is `7.965567455406`.

FinancePy's own normal-CDF approximation returns `7.965579241666`, about `1.18e-5` above the `erf` control. That pre-existing approximation difference is not part of this report.

## Correction and validation

- compute the quotient with `expm1` and its exact `r=0` limit;
- return the European value for the exact zero-rate put with non-negative dividend yield;
- allow 100 rather than 50 secant iterations for nearby positive-rate put boundaries;
- focused file: 4 tests passed;
- complete unit suite: 959 tests passed, with four pre-existing LSMC conditioning warnings;
- the same new cases fail against the unmodified wheel/source and pass after the correction.

Four bounded searches of the FinancePy issue and PR history found no matching report. This is duplicate screening, not a priority guarantee.

## Scope

The Barone–Adesi–Whaley method remains an approximation. The report does not claim exact agreement with a tree, production use, a customer position, or financial loss. Inputs and controls are synthetic.
