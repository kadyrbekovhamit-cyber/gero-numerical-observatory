# QuantLib: deterministic ITM probabilities and cap/floor deltas

**Two implementation defects, one report.** Independent GERO research by Xamit Kadirbekov, 14 September 2026.

At zero standard deviation, QuantLib 1.43 returns the opposite deterministic probability in `blackFormulaAssetItmProbability`. Its Bachelier counterpart returns an option payoff amount instead of a probability. Both defects are present in current source `ca953b7ebdb400f2839d86f18eeb0d4a2a4a30a4` and affect reported cap/floor optionlet deltas in the tested zero-volatility cases. The tested option prices themselves remain correct.

## A four-number reproduction

For a call with forward 120 and strike 100, with zero uncertainty:

| Calculation | Released result | Mathematical result |
|---|---:|---:|
| Black asset ITM probability | 0 | 1 |
| Bachelier asset ITM probability | 20 | 1 |
| Black call price, discount factor 1 | 20 | 20 |
| Bachelier call price, discount factor 1 | 20 | 20 |

For the Black call, changing standard deviation from exactly zero to `1e-6` changes the probability from 0 to approximately 1. A put with forward 80 and strike 100 exhibits the same reversal. Bachelier returns 20 for either in-the-money example; if the numbers are expressed in smaller units, its erroneous probability changes with those units.

These are model probabilities/normalized forward sensitivities, not forecasts of real-world investment outcomes.

## Contract and independent reference

The official [Black helper declaration](https://github.com/lballabio/QuantLib/blob/ca953b7ebdb400f2839d86f18eeb0d4a2a4a30a4/ql/pricingengines/blackformula.hpp#L249) specifies the asset-measure ITM probability. The [Bachelier declaration](https://github.com/lballabio/QuantLib/blob/ca953b7ebdb400f2839d86f18eeb0d4a2a4a30a4/ql/pricingengines/blackformula.hpp#L405) specifies a normal-CDF probability.

Away from the strike, a deterministic call is in the money exactly when F > K; a put is in the money exactly when F < K. The probability is therefore 1 or 0. No numerical integration, model calibration or second library is needed for that reference. At positive standard deviation, the independent checks use Python's `math.erfc` for the normal CDF.

The audit also checks probability bounds, positive changes of units, overload agreement, continuity toward zero away from the strike, and consistency with price derivatives. Black's price decomposition using its cash and asset probabilities fails on the same 66 deterministic inputs. At the strike, the existing strict-ITM convention returns zero at exactly zero variance; the patch preserves it. The audit does not claim a continuous 0.5 limit at that point.

## Source cause and correction

In the [Black branch](https://github.com/lballabio/QuantLib/blob/ca953b7ebdb400f2839d86f18eeb0d4a2a4a30a4/ql/pricingengines/blackformula.cpp#L593), the zero-standard-deviation comparison is reversed. The candidate changes `<` to `>`.

In the [Bachelier branch](https://github.com/lballabio/QuantLib/blob/ca953b7ebdb400f2839d86f18eeb0d4a2a4a30a4/ql/pricingengines/blackformula.cpp#L950), the branch returns the positive part of the payoff. The candidate returns its strict positivity indicator and forms the standardized variable only after handling zero standard deviation. Positive-standard-deviation formulas are unchanged.

## Executed validation

Released tests use the actual compiled QuantLib 1.43 macOS ARM64 wheel. Current-source tests compile complete upstream C++ translation units and their dependencies with Apple Clang 17, C++17, `-O2`, Boost 1.88.0 and standard-library shared pointers. They do not substitute a copied formula for the implementation.

| Check | Inputs | Released/current baseline failures | Corrected failures | Restored-source failures |
|---|---:|---:|---:|---:|
| Black probabilities | 420 | 66 | 0 | 66 |
| Bachelier probabilities | 240 | 24 | 0 | 24 |
| Direct C++ cap/floor engine deltas | 16 | 6 current-source | 0 | 6 |
| Full Cap/Floor instruments via released wheel | 32 optionlets | 12 | Not a rebuilt wheel | Not applicable |

The 660 helper inputs comprise 90 failing cases and 570 controls. The controls include positive standard deviations, calls and puts, different units, shifted Black inputs, negative Bachelier forwards, and strict-ITM boundary cases. Eleven of the erroneous Bachelier probabilities exceed 1; the others illustrate why range checks alone are insufficient. Both C++ overloads were exercised on every helper input; delegating overloads are not counted as extra independent cases.

All **12 focused upstream BlackFormula test cases pass** after correction: ten existing cases and two added regressions, 72 assertions including fixture checks. The original implementation and the restored-source mutation each fail the two new cases with 42 failed assertions. The entire QuantLib test suite was not built or run.

The mutation file is byte-identical to the original tested source. Baseline and mutation outputs agree. Prices in the direct engine tests are unchanged by the correction, and finite differences agree with the corrected deltas.

## Measured downstream consequence

Using released, complete `Cap` and `Floor` instruments, synthetic one-million notionals, two future EURIBOR coupon periods, a 5% flat forwarding curve and a 3% discount curve:

- A zero-volatility Black cap struck at 4% has NPV **9,807.39003756** and reports normalized optionlet deltas **[0, 0]**. Price bumps imply **[1, 1]**.
- A zero-volatility Black cap struck at 6% has zero NPV and reports **[1, 1]**; price bumps imply **[0, 0]**.
- Bachelier's in-the-money cap/floor deltas are scaled by the forward/strike difference instead of being unit indicators.

The 32 released optionlet observations include both pricing engines and positive-volatility controls. Twelve reported deltas disagree; all independent references agree with normalized price bumps. These deltas are normalized coefficients, not currency-valued DV01s.

The independent C++ engine experiment invokes the actual Black and Bachelier engines with validated synthetic arguments. Its six delta failures disappear after the helper corrections and return with the original source. That is a measured effect on risk outputs in these public-library examples. It does not establish use by a bank, any trading loss, a production hedge, or changes to option prices.

## Reproduce

For the smallest released example:

```bash
python3 -m venv .venv
.venv/bin/pip install QuantLib==1.43
.venv/bin/python minimal_repro.py
```

The evidence package includes `reproduce.py`, `reproduce_bachelier.py`, `downstream_capfloor.py`, the native probes, `build_native.py`, `bootstrap_native.py`, the patch, raw JSON, build commands, focused test logs, source archive and SHA-256 manifest. `REPRODUCE.md` gives the full sequence. External dependencies have pinned download URLs and checksums; the 137 MB Boost distribution and platform-specific wheel are not duplicated inside the evidence ZIP.

## Prior work and limitations

A bounded review of the canonical 91-entry GERO catalog, current GitHub issues/PRs, file history and web search found no exact earlier report of these two branches. PR674 introduced the helpers and delta outputs in 2019; its discussion addresses implementation/API work. It is acknowledged as source history, not treated as an earlier report of these defects. Generic expiry-zero Greeks and low-volatility American solver reports concern different code paths. Search results and interpretation are preserved in `DUPLICATE_REVIEW.md` and the evidence directory.

The scope is these helpers and specified cap/floor paths at zero standard deviation, with positive-standard-deviation controls. No real trades, customer records, insurer data, hardware deployments or production portfolios were tested. This report does not claim exhaustive novelty, a security vulnerability, maintainer acceptance or a completed full-library audit.

Report: CC BY 4.0. Original GERO harness code: MIT. QuantLib and Boost retain their upstream licenses and copyright notices. AI-assisted research and archival preparation; all stated numerical results come from the preserved executable experiments.

Frozen evidence: [ZIP archive](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/catalog/artifacts/gero-quantlib-zero-stddev-evidence-2026-09-14.zip). SHA-256: `10d5d7b5f5d291eb6f39c0798c3a3026c99943934c2facff03127a7dd86588df`.
