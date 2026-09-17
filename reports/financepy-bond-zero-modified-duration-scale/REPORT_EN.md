# FinancePy BondZero: verifying the upstream fix for a 10,000-fold duration error

**A one-year zero-coupon bond reported modified duration 9,522.90 instead of approximately 0.95229. The upstream source now returns the correct value for its existing finite-difference convention; the official PyPI 1.1.2 wheel still reproduces the defect.**

Independent GERO research by Xamit Kadirbekov. Original report: 16 September 2026. Upstream correction and independent replay verified: 17 September 2026. This archives [existing issue 267](https://github.com/domokane/FinancePy/issues/267), rather than announcing a second discovery.

## Status first

Maintainer commit [`bb10c3936e078a5212694d017746453175151c09`](https://github.com/domokane/FinancePy/commit/bb10c3936e078a5212694d017746453175151c09), dated 16 September 2026, removes the extra `* 10000` in `BondZero.modified_duration`. Its changelog also records a `bond_zero` units correction. The tested current master is `4c7cd50bdadd6efc9ac74fa81e93374397d18e3e`.

The independent follow-up executes that **unmodified current source**. It is not a fresh local candidate patch. The [verification was sent in the existing issue](https://github.com/domokane/FinancePy/issues/267#issuecomment-5713269134). The issue remained open at review. There is no claim of explicit maintainer attribution, a merged GERO pull request or a new packaged release.

The latest official PyPI metadata still identifies version **1.1.2**. That wheel was tested separately and remains affected. A fix in the source repository does not establish that an installed release contains it.

## Exact scenario and units

Issue and settlement are 1 January 2025; maturity is 1 January 2026. Issue price is 95 per 100, yield is the fraction `0.05`, and default ACT/365F gives exactly one year. The independent discounted cash flow is

`P(y) = 100 / (1+y)`.

For a yield increase of one basis point, `h = 0.0001`, the existing method uses the forward secant

`D_h = [P(y) − P(y+h)] / [h × P(y)] = 1 / (1+y+h)`.

| Quantity | Recorded value or independent expectation |
| --- | ---: |
| Dirty price per 100 | 95.23809523809524 |
| Actual price decrease for +1 bp | 0.009069431029246289 |
| Original source / release modified duration | 9522.902580708604 |
| Corrected upstream modified duration | 0.9522902580708603 |
| Independent one-bp forward-secant expectation | 0.9522902580706599 |
| Exact infinitesimal derivative-based duration | 0.9523809523809523 |

The original duration predicts a one-bp price decrease of approximately **90.6943 per 100**; actual repricing gives **0.00906943**. This is a synthetic price-to-risk inconsistency. It is not evidence of a trade, portfolio loss or a customer being charged that amount.

The small difference between 0.952290 and 0.952381 is the expected finite-difference approximation. This report does not count that difference as a defect or claim the one-sided secant is an exact derivative.

```python
from financepy.products.bonds.bond_zero import BondZero
from financepy.utils.date import Date

settle = Date(1, 1, 2025)
bond = BondZero(settle, Date(1, 1, 2026), 95.0)
print(bond.dirty_price_from_ytm(settle, 0.05))
print(bond.dv01(settle, 0.05))
print(bond.modified_duration(settle, 0.05))
```

## Why the multiplier was wrong

`dv01()` obtains the dirty-price difference for a fractional-yield bump of `0.0001`. `dollar_duration()` already multiplies that difference by `10000`, which divides it by the bump. Dividing dollar duration by the original dirty price therefore gives `D_h`.

The old `modified_duration()` multiplied this result by `10000` again:

```diff
-        md = dd / fp * 10000
+        md = dd / fp
```

`Bond.modified_duration()` and `BondFRN.modified_duration()` use normalization by price without the second conversion. The `BondZero.dollar_duration` prose also calls its result a price change for 100 bp, despite the displayed factor representing sensitivity per unit fractional yield. Our oracle follows the executed price/bump identity; it does not depend on that inaccurate sentence. The observed method-level multiplier defect and this units wording are distinguished.

## Independent execution and mutation

The frozen matrix has **1,320 distinct method-input scenarios**, including 1,200 positive horizons and 120 maturity controls. It combines five settlement dates including leap-day boundaries; eleven remaining day counts from zero to 10,950; six fractional yields from −0.05 to 1.0; ACT/360 and ACT/365F; and two issue prices. There are 660 combinations before varying issue price.

The oracle discounts a single terminal cash flow. For remaining time at most one year it uses the library's simple-interest convention; for longer time it uses annual compounding. Python calendar arithmetic supplies actual days. Independent mpmath calculations at 80 and 120 decimal digits agree within the recorded precision bound. Inputs and tolerances were frozen before the original evaluation. Duration tolerance is absolute and relative `1e-8`, using `math.isclose`'s maximum criterion.

| Execution | Duration mismatches / 1,320 | Price-to-risk identity mismatches |
| --- | ---: | ---: |
| Original source `2b9227fe` | 1,200 | 1,200 |
| Unmodified corrected upstream `4c7cd50b` | 0 | 0 |
| Current source with only `* 10000` restored | 1,200 | 1,200 |
| Official PyPI 1.1.2 | 1,200 | 1,200 |

The largest corrected duration error against the high-precision forward-secant oracle is `3.56e-11`. All 120 maturity controls remain zero. Independent prices, bumped prices, DV01, dollar duration, Macaulay duration and positive-horizon yield round trips pass their fixed checks. Recorded prices, accrued interest, principal amounts and inverse yields are unchanged by removing the multiplier.

The two relevant current upstream bond/portfolio test files pass **20 tests** on both original and current packages. All **27 independent regressions** pass current upstream; restoring the multiplier produces **24 failures and three passing maturity controls**. An initial preparation attempt omitted three CSV fixtures and produced file-not-found errors; the official fixtures were restored and the tests passed. These are focused results, not a full-suite result.

## Reproduction and provenance

The archive includes original and current source tarballs, the official wheel, complete Git blob inventories, frozen inputs, oracle and audit code, regressions, raw observations, environment records and exact upstream change. `SOURCE.json` records the versions and SHA-256 hashes.

Use Python 3.12 with the versions in `requirements-repro.txt`, then run:

```sh
python -B reproduce.py --out /absolute/path/to/new-output
```

The runner makes no network calls. It verifies both source archives against their Git inventories, verifies the wheel's RECORD, regenerates the independent 80/120-digit oracle, and executes all four real packages in separate sequential processes. It checks the source identities after execution. The retained harness label `candidate` means **unmodified corrected upstream** in this archive; `SOURCE.json` makes the mapping explicit.

The executed environment reused an existing dependency installation: Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, Numba 0.62.1 and mpmath 1.3.0. It was not a fresh dependency installation. Numerical thread counts were configured to one; no Mac GPU was used.

## Prior work and limits

Review covered 260 public issue/PR title-body records, 30 target-file history entries, four focused searches, current source/changelog, and 106 canonical GERO publication IDs. Own issue 267 is retained. Issue 227 is a swap-PV01 convention discussion; issue 254 concerns final ex-dividend redemption; issue 256 concerns principal cash units. These are different paths. Search coverage is bounded, not worldwide proof of novelty.

This establishes the corrected duration identity for the stated synthetic scenarios and the continued defect in the tested released wheel. Other yield conventions, invalid inputs, all vector/batch shapes, performance, the full test suite, bank deployment and actual financial losses are outside the evidence. The report does not recommend an investment or replace a production risk-model review.

Original source and tests retain GPL-3.0-or-later and their notices. Original GERO prose is CC BY 4.0. AI assisted source review, harness preparation and editing; quantitative claims come from recorded executions and independent mathematical references. No affiliation with FinancePy is implied.
