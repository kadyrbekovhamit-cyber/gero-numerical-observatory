> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/gero-pyliferisk-geometric-mortality-audit/blob/5fdb75e697066c1ecc46327bee0b33c4525c8899/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# pyliferisk geometric valuations discard supplied mortality assumptions

**Xamit Kadirbekov · GERO Research · 14 September 2026**

Status: reproduced on the official PyPI 1.12.0 wheel; correction submitted as
[upstream PR #17](https://github.com/franciscogarate/pyliferisk/pull/17).
The patch is proposed, not an accepted or released upstream fix.

## A zero-growth annuity changes value

An annuity with zero growth should equal the corresponding level annuity when
both use the same interest rate and mortality assumptions. In the released
library, `qax()` instead resets mortality to the original table's 100% level.

```python
from pyliferisk import Actuarial, ax, qax
from pyliferisk.mortalitytables import GKM95

mt = Actuarial(nt=GKM95, perc=75, i=0.03)
print(ax(mt, 60))       # 14.93597449181164
print(qax(mt, 60, 0))   # released: 13.616328838004055
```

This is a synthetic annual whole-life annuity-immediate of one unit, age 60,
interest 3%, and mortality rates scaled to 75% of GKM95. A direct conditional
survival cash-flow sum returns 14.935974491811635. The released geometric result
is approximately **8.8353% lower**. This is a library pricing discrepancy, not a
claim of actual insurer exposure or customer losses.

| Mortality scale | Level annuity / direct cash-flow value | Released zero-growth `qax` |
| --- | ---: | ---: |
| 50% | 16.80035314426194 | 13.616328838004055 |
| 75% | 14.93597449181164 | 13.616328838004055 |
| 100% control | 13.616328838004055 | 13.616328838004055 |

The same cause affects `qAx`, `qax`, `qaax`, `qaxn`, `qaaxn`, `qtax`, and `qtaax`.
Public `annuity()` dispatch reaches affected helpers. Tables supplied directly
as survivor counts (`lx`) or mortality probabilities (`qx`) have no original
`nt`; geometric revaluation can then raise an exception or use an empty table.

## Cause and proposed correction

The helpers adjust interest to `j = (i - growth) / (1 + growth)` and construct
`Actuarial(nt=mt.nt, i=j)`. That changes both the discount rate and the mortality
table: the constructor defaults `perc` to 100 and cannot recover custom `lx` or
`qx` from `nt=None`.

The correction supplies copies of the effective mortality data instead:

```python
mtj = Actuarial(lx=mt.lx[:], qx=mt.qx[:], i=j)
```

Only the discount rate changes. Copies prevent constructor writes from aliasing
the caller's lists. The patch preserves existing growth timing and fractional
payment approximations. It does not change the separate undefined-variable
paths `qAxn`/`qtAx`, unfinished helpers, deferred geometric dispatch, or unrelated
fractional-payment issues. Those are not claimed as corrected here.

## Verification

- **630 annual scenarios**, using independent direct sums at 60-digit Decimal
  precision: **498 failures before, zero after**. The 498 include 198 exceptions;
  exceptions are not an additional count.
- GKM95 at 50%, 75%, and 100%; custom survivor and probability tables; interest
  rates -1%, 0%, and 3%; growth -2%, 0%, and 2%; two ages and seven helpers.
- Four standard-library regression test methods pass. They also check zero
  growth with annual, semiannual, quarterly and monthly payments, public
  non-deferred `annuity()` dispatch, and unchanged caller state.
- Running the new tests against the original released module produces
  **82 failing and 42 errored subcases**. These are subcases, not 124 test methods.
- The Decimal oracle uses the supplied survivor table and sums each actual
  conditional payment directly; it does not use commutation functions or an
  `Actuarial` reconstruction at the transformed rate. Agreement tolerance is
  `rel_tol=2e-12, abs_tol=2e-12`.

All calculations ran sequentially on CPU with Python 3.9.6 and the standard
library. No GPU or parallel worker was used. There is no performance benchmark.
The suite establishes these cases, not correctness of the entire library.

## Reproduce

The official wheel is included for reproducibility under its GPL license.
The `vendor/released` and `vendor/patched` directories contain the relevant
source snapshots. From this directory:

```sh
python3 -B verify.py --source vendor/released --output released-rerun.json
python3 -B verify.py --source vendor/patched --output patched-rerun.json
PYTHONPATH=vendor/patched python3 -B -m unittest discover -s tests -v
```

The verifier records mismatches in JSON; it exits normally for an expected
failing baseline, so inspect the `failures` and `exceptions` fields.

Official wheel `pyliferisk-1.12.0-py3-none-any.whl` SHA-256:
`1c2ab2b902424f33d408d5f4e13badd98d3d0ca26462d83562bb6b21cd2bca53`.
The released `__init__.py` is byte-identical to upstream master
`5c28ca34a30f4e350f604f4b35bf738c816e59bc` (17 September 2023).
Correction commit: `0458d03ca4d03a73249df3e70c033167235962fa`.

## Duplicate screening and sources

On 14 September 2026, screening covered all 14 available upstream issue/PR
records and all 8 available issue comments before submission. GitHub issue
searches scoped to the repository for `geometric`, `perc`, and `qax` each
returned zero matches. The earlier reports #15 (deferred mortality) and #16
(deferred fractional-payment adjustment) concern different expressions.
No matching earlier report was found in that scope. This is not a guarantee of
worldwide novelty or a claim that no private report exists.

- [Official PyPI release and artifact hashes](https://pypi.org/project/pyliferisk/1.12.0/)
- [Pinned upstream source](https://github.com/franciscogarate/pyliferisk/blob/5c28ca34a30f4e350f604f4b35bf738c816e59bc/pyliferisk/__init__.py)
- [Project documentation of the `perc` parameter](https://github.com/franciscogarate/pyliferisk#quick-start)
- [Proposed upstream correction and review status](https://github.com/franciscogarate/pyliferisk/pull/17)

Original code and mortality tables: Francisco Garate and contributors,
GPL-3.0-or-later. This report, regression tests, and reproduction scripts are
also provided under GPL-3.0-or-later. See `LICENSE`.
