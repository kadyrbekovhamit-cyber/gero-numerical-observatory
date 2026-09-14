> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-mortgage-zero-small-rate.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# FinancePy mortgage payments at zero and very small rates

Independent numerical audit by Xamit Kadirbekov / GERO Research, 13 September 2026.

`BondMortgage.repayment_amount()` in FinancePy **1.1.2** raises
`ZeroDivisionError` at a zero annual rate. At sufficiently small nonzero rates,
its annuity formula either raises the same exception or returns a payment that
does not amortize the principal. `generate_flows(..., REPAYMENT)` calls this
method and inherits the problem.

The released source and the same file on upstream master commit
`2b9227fea9d832c4033421d6cd53a54316414fca` are byte-identical. SHA-256:
`1ae86f6abe8b8e86c8085664eae83ff2babf9ab4c22c022bd3f909cbb502b075`.

## Reproduction

Synthetic principal **120,000**, monthly payments, start 1 January 2025,
maturity 1 January 2055: 360 payment periods. Rates in the table are annual
fractions, not percentages.

| Annual rate | Correct monthly payment | FinancePy 1.1.2 | Final remaining principal |
| ---: | ---: | ---: | ---: |
| 0 | 333.333333333333 | `ZeroDivisionError` | No completed schedule |
| 1e-15 | 333.333333333338 | `ZeroDivisionError` | No completed schedule |
| -1e-15 | 333.333333333328 | 250.199979298351 | 29,928.007452592137 |
| 1e-12 | 333.333333338347 | 333.599972407815 | -95.990065009659 |
| -1e-12 | 333.333333328319 | 333.155764701533 | 63.924705642337 |
| 0.035 (control) | 538.853625370589 | 538.853625370588 | 7.90e-10 |

At zero interest, each repayment must be `principal / number_of_payments`.
For a nonzero rate, the oracle independently sums the present values of unit
cash flows at **80-digit Decimal precision**, then divides principal by that
sum. It does not evaluate the implementation's geometric-series quotient.

Minimal native public-API example:

```python
from financepy.products.bonds.bond_mortgage import BondMortgage
from financepy.utils.date import Date

mortgage = BondMortgage(Date(1, 1, 2025), Date(1, 1, 2055), 120000.0)
print(mortgage.repayment_amount(0.0))  # expected 333.3333333333333
# FinancePy 1.1.2: ZeroDivisionError: float division by zero
```

Full reproducible matrix:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sh run_single_core.sh .venv/bin/python reproduce.py --output released.json
```

See `evidence/released.json` and `evidence/patched.json` for all cases and
explicit tolerances in `reproduce.py`. These are unrounded binary floating-point
cash flows; no currency-minor-unit rounding policy is assumed.

## Mechanism and correction

The original code computes `p = (1 + annual_rate/frequency)**periods` and
then divides by `p - 1`. At zero the expression is `0/0`. Near zero, forming
`1 + rate` and subtracting one destroys significant digits. Very small changes
of the rate can therefore cause large changes in the payment.

`fix.patch` uses the exact zero-rate limit and `log1p`/`expm1` for nonzero
rates. The equivalent positive- and negative-rate expressions keep the
exponential argument nonpositive, avoiding unnecessary intermediate overflow.
The patch rejects a periodic rate at or below -1, for which positive discount
factors do not exist. It does not change the payment schedule, calendar,
interest-only branch or rounding policy.

Apply to the pinned source checkout:

```bash
git checkout 2b9227fea9d832c4033421d6cd53a54316414fca
git apply /path/to/this-report/fix.patch
python -m pytest -q unit_tests/test_FinBondMortgage.py unit_tests/test_FinBondMortgageRates.py
```

To run the same oracle against that checkout, set `PYTHONPATH` to its root
before invoking `reproduce.py`. The Python environment must contain the
FinancePy dependencies. The source hash in each JSON output identifies which
implementation was tested.

The correction is submitted as [FinancePy PR #257](https://github.com/domokane/FinancePy/pull/257),
commit `41f38f04eec679ca95c1796886fd677c6f0a05bc`. At submission the PR is open and unmerged;
its live page is authoritative for upstream review and CI status.

## Evidence and scope

- 144 native-API scenarios: 4 payment frequencies, 3 terms and 12 rates.
  **108 fail before the correction; 0 fail after.** These are scenarios,
  not separate defects.
- 64 focused tests pass, including the two existing mortgage tests,
  high-precision payment checks, full amortization, interest-only controls
  and invalid-rate handling.
- The complete local unit suite passes: **1020 tests**, with 4 warnings.
- Restoring the original implementation makes **40** new repayment tests fail;
  16 ordinary-rate controls still pass. The corrected source was restored
  and its SHA-256 verified.
- All computations ran sequentially on CPU with numerical threads limited
  to one. No GPU calculations or performance benchmark was used.

The inputs are synthetic boundary cases. The tested ordinary 3.5% mortgage
rate passes in the released version. This report does not establish a deployed
lender's exposure, customer loss, exploitability or a currency-rounding policy.
The small nonzero rates isolate the numerical mechanism; the zero-rate case
is an exact mathematical boundary.

Public GitHub searches for mortgage, repayment, `repayment_amount`, zero rate
and `expm1` found no matching report within the searched title/body scope.
The related issues #29 and #220 concern product support, not this formula.
Search results are saved in `evidence/duplicate-search.json`; they do not prove
worldwide novelty.

## Sources

- [Released implementation, V1.1.2](https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_mortgage.py)
- [Pinned upstream source](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/financepy/products/bonds/bond_mortgage.py)
- [Existing mortgage unit tests](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/unit_tests/test_FinBondMortgage.py)
- [FinancePy package](https://pypi.org/project/financepy/1.1.2/)

Source excerpts and the correction retain FinancePy's GPL-3.0-or-later
license. The included original audit scripts are also supplied under that
license; see `LICENSE`.
