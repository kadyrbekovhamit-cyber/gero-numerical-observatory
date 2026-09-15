# FinancePy Merton calibration depends on the monetary unit

Independent GERO research by Xamit Kadirbekov. Native Python execution on macOS arm64, 14–15 September 2026. Evidence edition prepared for publication on 15 September 2026. AI-assisted preparation; author-reviewed claims and executable evidence.

Changing the common monetary unit of equity value and debt face changes the output of `MertonFirmMkt`, including the inferred asset volatility and default probability. The tested inputs describe exactly the same synthetic firm. The calibration objective combines a squared monetary residual with a squared volatility residual without normalization. Its optimizer result is consumed without a fit-quality check.

## Concrete example

Generate market observations independently from a firm with asset value 140, debt face 100, maturity one year, risk-free rate and asset growth both 5%, and asset volatility 25%. The analytical observations are equity value 45.63363370957471 and equity volatility 0.7306450094667433. The analytical physical default probability is 0.07767452345776465.

All equity and debt monetary inputs are then multiplied by the same factor. Maturity, rates and equity volatility are unchanged.

| Monetary multiplier | Original inferred asset volatility | Original default probability | Corrected default probability |
| ---: | ---: | ---: | ---: |
| 0.000001 | 79.1306% | 74.7802% | 7.76746% |
| 0.001 | 25.0001% | 7.76769% | 7.76746% |
| 1 | 25.0000% | 7.76745% | 7.76746% |
| 1,000 | 233.7713% | 90.9102% | 7.76746% |
| 1,000,000 | 302.7003% | 95.4947% | 7.76746% |

For example, the multiplier 1,000,000 represents converting amounts from millions of currency units to individual units. It does not represent increasing the firm's economic size relative to its debt.

The largest-unit original fit nearly reproduces the monetary equity price but returns equity volatility 3.311276376763251 instead of the supplied 0.7306450094667433. The altered default probability is therefore a downstream consequence of an inaccurate calibration, not a change to the Merton probability formula.

## Invariant and isolated correction

Merton equity valuation is homogeneous in asset value and debt face: multiplying both by a positive constant multiplies equity value by that constant. Leverage, equity volatility, asset volatility and default probability are unchanged. This also follows directly from the dimensionless ratio in the model's normal-CDF arguments.

The original objective is `(E - E_model)^2 + (sigma_E - sigma_E_model)^2`. Converting monetary units rescales only its first term and changes the coordinate scale of its optimizer. The candidate normalizes the monetary inputs and inferred asset coordinate by debt face before calling the same optimizer, then converts the inferred asset value back. It leaves the equations, optimizer, tolerance and initial-guess rule otherwise unchanged.

The exact control mutation restores the original source file. The baseline source and the released package both produce exactly the same recorded scenario rows as that mutation.

## Recorded tests

The main grid consists of 24 known parameter sets multiplied by five monetary scales: asset/debt ratios 1.1, 1.4 and 2; asset volatilities 20% and 40%; maturities 0.5 and 2 years; rates -1% and 5%; fixed growth 3%. Independent `math.erfc` normal probabilities generate the market observations and check repricing. These 120 observations are related scenarios, not independent production trials.

| Result | Current source | Release 1.1.2 | Candidate | Mutation |
| --- | ---: | ---: | ---: | ---: |
| Failed parameter/repricing round trips | 94 / 120 | 94 / 120 | 0 / 120 | 94 / 120 |
| Failed comparisons with the unscaled scenario | 87 / 120 | 87 / 120 | 0 / 120 | 87 / 120 |
| Scalar-versus-batch mismatches | 0 / 120 | 0 / 120 | 0 / 120 | 0 / 120 |

Checks use fixed relative tolerance 2e-5 for parameter recovery and independent repricing, and absolute tolerance 2e-6 for default probability. These allow the existing normal-CDF approximation and small solver error; none were relaxed after observing results. All tolerances and raw values are supplied. The existing upstream `test_FinModelMerton.py` passes with the candidate: one test passed. The full FinancePy test suite was not run.

The baseline reports a non-success optimizer status in 102 of 120 scenarios, and the candidate still reports one in 39. All candidate outputs pass the declared independent numerical checks. The correction does not replace the optimizer, add a fit-quality gate or guarantee convergence outside this grid. The remaining statuses are explicitly retained in the evidence.

## Provenance and limits

Current source: [FinancePy commit 2b9227f](https://github.com/domokane/FinancePy/commit/2b9227fea9d832c4033421d6cd53a54316414fca). The independently downloaded PyPI distribution is FinancePy 1.1.2; the current repository's printed banner still says 1.1.0. Both tested Merton source files and observed grid behavior are checked independently rather than identifying the current source from its banner.

Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, Numba 0.62.1, llvmlite 0.45.1 and pandas 2.3.3. These numerical dependency versions meet the inspected project requirements. Matplotlib is not needed by these tests and was not installed. The source archive contains 773 files, all checked unchanged after execution; caches live outside the pinned source.

This is a calibration implementation finding on synthetic data. It does not establish bank deployment, incorrect real customer ratings, financial losses or regulatory impact. The patch is a local candidate and has not been accepted upstream. Broader parameter regimes, invalid inputs and other optimizer/platform versions remain outside the executed coverage.

See `DUPLICATE_REVIEW.md`, `REPRODUCE.md`, `SOURCE.json`, the local patch, raw result JSONs and `evidence/paired-verification.json`. FinancePy source retains its original GPL license; this report does not relicense the library.

## Publication links

[Hugging Face](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-merton-monetary-scale.md) · [GERO](https://www.gero.uz/research/articles/financepy-merton-monetary-scale.html) · [Zenodo](https://zenodo.org/records/22757428) · [LinkedIn](https://www.linkedin.com/feed/update/urn:li:share:7505354627723431938/)

Evidence SHA-256: `026fb5188cd696896daf068e648ba730bc80d5b160a81f051c9164bb003326fa`.
