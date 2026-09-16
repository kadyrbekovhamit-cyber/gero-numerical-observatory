# FinancePy annuity pricing depends on prior payment calls

Independent GERO research by Xamit Kadirbekov, 16 September 2026.
Case: `financepy-annuity-call-order-face-cache`.

`BondAnnuity` reuses cached cash-flow amounts whenever the settlement date is
unchanged, even if the requested face amount changes. A previous call to
`calculate_payments()` or `print_payments()` can therefore change the subsequent
quoted annuity price. This is one state-dependent implementation defect.

## Concrete result

Take a 5% semiannual annuity from 20 June 2018 to 20 June 2019, ACT/360 accrual,
with a flat zero discount rate. The price per 100 is
`100 * 0.05 * (183+182)/360 = 5.069444444444...`.

| Call sequence on a fresh object | Original | Candidate |
|---|---:|---:|
| Price directly | 5.069444444444445 | 5.069444444444445 |
| Calculate face 100 payments, then price | 506.94444444444446 | 5.069444444444445 |
| Calculate face 1, then request face 100 payments | amounts still for face 1 | amounts for face 100 |

Printing payments before pricing exhibits the same contamination. Pricing
before printing can instead leave the printed amounts in face 1 units.

## Real implementation and candidate

Current master was rechecked at
`2b9227fea9d832c4033421d6cd53a54316414fca`. The extracted official PyPI 1.1.2
package reproduces the results and has identical `bond_annuity.py` bytes.
The import banner says 1.1.0; version attribution uses source/release evidence.

The public pricing methods request `calculate_payments(settle_dt, 1.0)` and
then multiply the discounted flows by `self.par`, which is 100. The date-only
early return bypasses this normalization if another face was previously used.

`candidate.patch` removes that early return so that each call rebuilds amounts
for its requested face. This conservative correction also regenerates the date
schedule. Runtime/performance effects have not been benchmarked. A subsequent
optimization can cache schedule dates independently of face-dependent amounts.
The candidate is not an upstream-accepted correction.

## Executed verification

The predeclared grid contains 864 distinct combinations: three settlement dates
(including 29February2024), one/five years, four payment frequencies,
coupons 0/1%/5%, flat continuously compounded rates −2%/0/3%, and prior/requested
faces 0/1/100/1,000,000. All inputs are synthetic.

- Original: 432 failing scenarios; candidate: 0; restored original: 432;
  official released package: 432. Each failing scenario is observed through six
  overlapping checks; 2,592 failed assertions do not mean 2,592 independent bugs.
- Independent dated-cash-flow sums use Python calendar-day differences,
  ACT/360 accrual and ACT/365F exponential discounting at 80 and 120 decimal digits.
  These two precision runs agree after conversion to binary64. The fixed
  tolerance is `2e-11 * max(1, abs(expected))`.
- Calendar generation is not independently audited: emitted payment dates
  are accepted as the declared cash-flow dates. Fresh prices and fresh-face
  payments pass the independent oracle in every scenario.
- All 432 previously passing complete rows are unchanged. Fresh prices,
  fresh-face payments and all emitted payment dates are unchanged across the
  entire matrix. All original, restored and release rows are exactly equal.
- Five existing annuity tests pass on original and candidate. Fourteen focused
  regressions pass on candidate; restoring the early return yields 12 fail / 2 pass.
- 230 original package-file hashes are verified; only the stated candidate file
  differs. Each variant runs in a separate process with separate Numba cache.

This is not a full-suite, clean dependency-install or performance benchmark.
It does not establish real-bank deployment, trade errors or customer losses.

## Duplicate review

The bounded review covered 257 public upstream issue/PR title/body records,
four focused searches, 26 target-file history summaries, the relevant changelog,
PR #93's discussion and the current 98-report GERO catalog. No exact match was found.
Search indexing is incomplete in practice: the direct search for BondAnnuity
returned zero, while manual review found PR #93 mentioning its tests. The broader
title/body review was therefore retained.

PR #93 migrates annuity tests; it does not report this cache/face defect. PR #256's
published face-scaling report concerns accrued interest in Bond, BondFRN and
InflationBond, not this class or date-only payment cache. Cash-settled swaption
issue #262 and mortgage PR #257 address different methods. No claim is made to
having exhaustively searched every historical discussion or private report.

## Reproduce

Use Python 3.12 and the versions in `requirements.txt`; no model or paid service
is required. The archive contains baseline/candidate/mutation/release packages.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
.venv/bin/python verify.py
```

The runner configures one numerical thread and runs variants sequentially.
`minimal.py` also runs with FinancePy 1.1.2 installed, or with the selected source
checkout on `PYTHONPATH`. Raw results, logs, patch and hashes are retained.

Upstream source remains GPLv3; see `UPSTREAM-LICENSE.txt`. Investigation and
artifact preparation were AI-assisted; the numerical results were executed.

## Maintainer submission and evidence

[Official issue #268](https://github.com/domokane/FinancePy/issues/268) contains the reproducer and candidate patch. Submitted does not mean accepted.

[Immutable research archive](https://github.com/user-attachments/files/32270325/gero-financepy-annuity-call-order-research-2026-09-16.zip). SHA-256: `b2b673aa86e3d6a0184011e961977f221850fa87b2dd005b229f20e4afe94340`. Historical preparation-time status inside the archive is retained; live publication receipts are maintained separately.

## Verified publication links

[github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/financepy-annuity-call-order-face-cache.md) · [huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-annuity-call-order-face-cache.md) · [linkedin](https://www.linkedin.com/feed/update/urn:li:share:7505866824627081217/) · [youtube](https://youtube.com/shorts/BZ3oSwTksaI) · [gero](https://www.gero.uz/research/articles/financepy-annuity-call-order-face-cache.html) · [zenodo DOI](https://zenodo.org/records/22791449)
