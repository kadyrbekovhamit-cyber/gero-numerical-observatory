> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/gero-financepy-bond-principal-face-audit/blob/main/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# FinancePy accrued-interest unit-scaling audit

FinancePy 1.1.2 scales a bond's dirty-price component by the requested face
amount inside `Bond.principal()`, but subtracts accrued interest calculated for
face `1.0`. The resulting cash value is not linear in face amount.

The same released unit mismatch occurs in `BondFRN.principal()` and
`InflationBond.inflation_principal()`. A related path,
`Bond.clean_price_from_survival_curve()`, returns a clean price per 100 of par
but subtracts accrued interest for face `1.0`. The two public BondFRN cash-value
methods also use previous/next coupon dates before initializing them when
called on a newly constructed object.

For the included semiannual 6% coupon bond at a 5% yield:

| Face | FinancePy 1.1.2 | Clean-value identity | Overstatement |
| ---: | ---: | ---: | ---: |
| 100 | 102.6945545291 | 101.7264329821 | 0.9681215470 |
| 1,000,000 | 1,027,043.3255671025 | 1,017,264.3298212462 | 9,778.9957458563 |

The invariant is:

```text
principal(face) = clean_price_per_100 * face / 100
```

Face `1.0` is a control: it passes because the unscaled accrued amount happens
to match that face. The two larger faces fail, and direct proportionality also
fails.

The expanded released-wheel reproduction reports seven formula mismatches:
two face-scaling failures in each of the three principal methods and one
dirty-minus-accrued identity failure in the survival-curve clean price. It also
records the two fresh-object `AttributeError` failures in BondFRN separately;
those are state-initialization defects, not additional formula mismatches.

| Released method | Synthetic check | Observed difference |
| --- | --- | ---: |
| `Bond.principal()` | face 1,000,000 | 9,778.9957458563 |
| `BondFRN.principal()` | face 1,000,000 | 2,336.2132193398 |
| `InflationBond.inflation_principal()` | face 1,000,000 | 228.6375142699 |
| `Bond.clean_price_from_survival_curve()` | clean price per 100 | 0.9681215470 |

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py
```

The reproducer uses the released PyPI wheel, public API calls and synthetic
dates and amounts. It contains no customer or transaction data.

## Correction

[FinancePy PR #256](https://github.com/domokane/FinancePy/pull/256) calculates
accrued interest in the same units as each returned cash value and initializes
BondFRN accrual state before dirty pricing. The local complete unit suite
reports 962 passes. Restoring each released calculation makes its new
regression fail. The pull request remains open and unmerged; its current
upstream CI status is recorded on the PR itself.

This is an independent numerical-correctness report. It does not establish
production exposure, financial loss, security impact or acceptance of the
submitted correction.

## One-minute evidence brief

The `media` directory contains the 55-second English vertical brief for the
first `Bond.principal()` case, English SRT and WebVTT captions, and the original
thumbnail. Its narration is synthetic and disclosed on screen; the motion
cards are original and contain no third-party imagery or music. The three
additional methods were confirmed after that short was rendered and are
documented in this expanded reproducer instead of being retroactively inserted
into the video.

The full claim ledger, production script and visual QA record are preserved in
the GERO publication package linked from the report page.
