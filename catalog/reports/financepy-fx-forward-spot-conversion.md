# FinancePy FXForward: today's foreign-currency NPV uses a future exchange rate

Independent numerical audit by Xamit Kadirbekov / GERO Research. Evidence dated 16 September 2026. Version 1.0.0.

The official FinancePy 1.1.2 wheel and the current `fx_forward.py` at commit `87779f5e1bd99b1a0d2eae241d2c453488b0c76d` contain the same source bytes. `FXForward.value()` computes a domestic present value, then divides it by the contract's current forward exchange rate to produce `npv_for`. Today's currency conversion should use the current spot rate. This report uses `spot_days=0` throughout to isolate that issue from settlement conventions.

For a contract receiving EUR 100 and paying USD 120 after one year:

| Input or result | Value |
|---|---:|
| Spot, USD per EUR | 1.25 |
| Domestic discount factor | 0.95 |
| Foreign discount factor | 0.98 |
| Today's USD value | 8.50 |
| Today's EUR value from discounted legs | 6.80 |
| Released `npv_for` | 6.591836734693896 |
| Corrected `npv_for` | 6.800000000000020 |

The independent reference discounts each currency leg first:

\[
V_{USD}=100(0.98)(1.25)-120(0.95)=8.50,
\qquad
V_{EUR}=100(0.98)-\frac{120(0.95)}{1.25}=6.80.
\]

Equivalently, today's currency amounts must satisfy `npv_dom = spot * npv_for`. Dividing a present value by a forward rate mixes the valuation and future dates. The same discounted-leg valuation approach is used in [OpenGamma Strata's FX single pricer](https://github.com/OpenGamma/Strata/blob/main/modules/pricer/src/main/java/com/opengamma/strata/pricer/fx/DiscountingFxSingleProductPricer.java).

## Correction and verification

The correction replaces `self.npv_for = v / new_fwd_fx_rate` with `self.npv_for = v / spot_fx_rate`.

- Native released FinancePy against a separate 60-digit Decimal discounted-leg calculation: **648 scenarios**, 1,296 value comparisons per implementation. **504 scenarios fail before; zero after**. All failures concern foreign NPV; domestic NPV passes. Both notional currencies, multiple maturities and notionals are covered.
- **144 equal-discount controls** pass before and after; in those controls, the forward rate equals spot and the old implementation happens to agree.
- An additional **64 scenarios using FinancePy's real `FlatDiscountCurve`**: 32 fail before, zero after, including 16 vector-spot scenarios. Notional amounts and currency labels are retained. Positive, negative and zero rates are included.
- The existing upstream regression hardcodes the incorrect foreign value. Its corrected expectation and a spot-conversion identity reject the released formula and pass with the candidate. The full FinancePy suite was not run in this lightweight cycle.

The self-contained source and test patch is [financepy-fx-forward-with-regression.patch](financepy-fx-forward-with-regression.patch). The code-only patch is [financepy-fx-forward-spot-conversion.patch](financepy-fx-forward-spot-conversion.patch).

Reproduction in a Python 3.12 environment with FinancePy 1.1.2 dependencies:

```sh
python -m pip install evidence/financepy-1.1.2-py3-none-any.whl
python prepare_release.py
```

Then run sequentially:

```sh
python -B financepy_fx_forward.py
python -B financepy_integration.py
python -B financepy_regression_check.py
```

The scripts set numerical thread counts to one and limit CPU time. No production/customer data is involved. The initial attempt to disable Numba globally encountered an import incompatibility with its vectorized functions; the successful native runs use the library's normal Numba support with one worker. That import failure was not counted as a numerical defect.

## Duplicate screen and limits

Three GitHub searches returned 2, 3 and 24 records respectively. Titles and available search bodies were inspected; the closest records #244, #241 and #102 were read with their comments. They concern curve calibration or option-model support, rather than converting a present value by a future rate. No direct duplicate was identified in this bounded screen. This is not a worldwide priority claim.

Sources: [released package](https://pypi.org/project/financepy/1.1.2/), [pinned implementation](https://github.com/domokane/FinancePy/blob/87779f5e1bd99b1a0d2eae241d2c453488b0c76d/financepy/products/fx/fx_forward.py), [pinned existing test](https://github.com/domokane/FinancePy/blob/87779f5e1bd99b1a0d2eae241d2c453488b0c76d/unit_tests/test_FinFXForward.py). Hashes and exact search queries are in `evidence/sources.json` and `evidence/duplicate-screen.json`.

An independent source-level candidate remains: `value()` discounts to expiry while `forward()` uses delivery. Nonzero spot lag needs a separate audit and is not corrected here. No bank usage, customer loss, investment advice or performance improvement is asserted.


## Reuse and attribution

Report and original audit code: Xamit Kadirbekov / GERO Research, 2026. GPL-3.0-or-later applies to this FinancePy-derived package; upstream notices and the bundled wheel are retained. Public issue-search material is included for attribution and duplicate screening, not as a claim of authorship. No company affiliation or maintainer acceptance is implied.
