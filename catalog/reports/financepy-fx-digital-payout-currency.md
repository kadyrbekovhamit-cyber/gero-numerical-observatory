> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-fx-digital-payout-currency.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# FinancePy misprices FX digital payouts by payout currency

Independent numerical audit by Xamit Kadirbekov / GERO Research, 13 September 2026.

FinancePy's FX digital products accept a notional/payout currency that can be
either the domestic or foreign member of the currency pair. Three formula
paths in FinancePy **1.0.1** use the wrong probability or discount term:

1. `FXDigitalOption` uses `N(d2)` for a foreign-currency payout. In domestic
   units that payoff is `S_T 1(condition)`, an asset-or-nothing payoff whose
   value uses `N(d1)`.
2. `FXDoubleDigitalOption` discounts a domestic-currency interval payout with
   the foreign discount factor instead of the domestic factor.
3. The same double-digital class uses `d2` rather than `d1` for a
   foreign-currency interval payout.

The same formulas remain in upstream commit
`2b9227fea9d832c4033421d6cd53a54316414fca`.

## Synthetic counterexample

Use EURUSD spot 1.20, a one-year expiry, domestic rate 5%, foreign rate 1%,
volatility 20%, single strike 1.25, and interval strikes 1.10 and 1.40.

```text
foreign single call: released 0.5449552491 | oracle 0.6397169446
foreign single put : released 0.6430720022 | oracle 0.5483103068
domestic interval  : released 0.4474588468 | oracle 0.4298665323
foreign interval   : released 0.5369506162 | oracle 0.5338519318
```

The two domestic single-digital controls agree with their independent
cash-or-nothing formulas. This isolates the affected paths rather than
asserting that every digital valuation is wrong.

## Independent formulas

For an FX rate `S_T` quoted as domestic currency per unit of foreign currency:

```text
PV[1_domestic * 1(S_T > K)] = df_domestic * N(d2)
PV[1_foreign  * 1(S_T > K)] = S_0 * df_foreign * N(d1)
```

The second identity follows directly by converting the foreign unit at
expiry: its domestic payoff is `S_T * 1(S_T > K)`. Interval payouts are the
difference between the corresponding terms at the lower and upper strikes.
`reproduce.py` implements these controls with Python's `math.erf`, not
FinancePy's vector normal-CDF approximation.

## Reproduce the released result

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --expect-mismatches 4
```

## Correction and validation

The correction is submitted as
[FinancePy PR #260](https://github.com/domokane/FinancePy/pull/260), commit
`1a78f85d4576f20d1671338bad49640a9dbb7b87`.

Local validation:

- six focused oracle tests pass after the correction;
- all 964 unit tests pass, with four pre-existing LSMC `RankWarning`
  messages;
- both official upstream CI workflows pass on the correction commit;
- running the new tests against the previous source fails the four affected
  cases and passes the two unaffected domestic single-digital controls;
- the two existing FX digital regression scripts complete.

Bounded GitHub searches for `FXDigitalOption d1 d2`, `foreign currency digital
option`, `FXDoubleDigitalOption discount rate`, and `digital payout currency`
found no prior FinancePy issue or pull request for these formulas. This is not
a worldwide novelty or priority guarantee.

All market inputs are synthetic. This report establishes library-level
formula discrepancies; it does not establish use by a financial institution,
a customer position, a production deployment, or financial loss.

## Sources

- [FinancePy 1.0.1 on PyPI](https://pypi.org/project/financepy/1.0.1/)
- [Current tagged FinancePy single-digital implementation](https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/fx/fx_digital_option.py)
- [Current tagged FinancePy double-digital implementation](https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/fx/fx_double_digital_option.py)
- [FinancePy correction PR #260](https://github.com/domokane/FinancePy/pull/260)

The reproducer and report are supplied under GPL-3.0-or-later; see `LICENSE`.
