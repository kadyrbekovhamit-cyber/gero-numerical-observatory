# FinancePy: premature series truncation produces an invalid option price

**A nonnegative cash-at-expiry double-touch payoff was priced at −0.1733515747.** Its complementary no-touch price was 1.1717090895, above the discounted maximum cash payment of 0.9983575147. Their sum was nevertheless correct.

Independent GERO research by Xamit Kadirbekov. Original public report: 15 September 2026. Current-source replay and archival preparation: 17 September 2026. Actual public FinancePy CPU implementation and synthetic inputs; no customer records or production-loss claim. AI assisted implementation, test and editorial preparation.

## Minimal case and the invariant

Use value date 1 January 2026, expiry 30 days later, spot 100, lower/upper barriers 80 and 125, volatility 0.2, foreign rate 0, domestic rate `0.5 * sigma * sigma`, payment 1, continuous compounding and ACT/365F curves. Both contracts pay cash at expiry; the double-touch contract requires at least one barrier hit. The no-touch contract pays if neither barrier was hit.

For positive payment K and domestic discount factor D, each price must lie in `[0, K D]`; the two prices sum to `K D`. This follows directly by discounting complementary nonnegative payoff indicators under the model. It does not depend on which numerical expansion computes the probabilities.

| Quantity | Original public API | Independent expected result |
|---|---:|---:|
| No-touch price | 1.1717090894757467 | 0.9981587590142639 |
| Touch price | −0.17335157473444152 | 0.00019875572704133946 |
| Discounted maximum payment | 0.9983575147413052 | 0.9983575147413052 |

The negative result is reproducible numerical behavior, not a measured trading loss. Checking only touch/no-touch parity misses it because the implementation constructs the second price by subtraction.

## Two truncation mistakes

The no-touch pricer uses a sine series. It stops when one individual contribution is small. In zero log drift, even-numbered coefficients can be zero while later odd terms remain material; a midpoint starting spot also creates zero sine factors. A small individual term is therefore not a bound on the remaining tail.

Removing that early exit alone leaves failures. For log-barrier width Z, the code's stated damping criterion is `exp(-0.5 * sigma^2 * (n*pi/Z)^2 * T) <= eps`. Solving it gives `n >= Z/(pi*sigma) * sqrt(2*log(1/eps)/T)`. The original threshold is half this value.

The proposed patch removes the single-term exit and corrects that threshold. It preserves the existing minimum of 50 terms and cap of 2,000. It does not clip invalid prices into the permitted interval. The finite cap and remaining numerical regimes mean this is not a proof of universal convergence.

## Executed versions and independent comparison

The original experiment used commit `2b9227fea9d832c4033421d6cd53a54316414fca`. Fresh replay uses current master `4c7cd50bdadd6efc9ac74fa81e93374397d18e3e`; the target file has identical bytes. The official PyPI 1.1.2 wheel was also executed. Its bundled source is unchanged.

The independent reference integrates the method-of-images absorbing Brownian transition density between the log barriers. Its variables are `x = log(S/L)`, `Z = log(U/L)`, `nu = r_d-r_f-sigma^2/2` and diffusion scale `sigma*sqrt(T)`. It uses 60-decimal mpmath arithmetic and 81 reflected images, without FinancePy's sine coefficients or truncation rule. Seven specified representative/extreme cases were compared with 121 images; differences were below 1e-45. The reference uses the effective rates reconstructed from the actual curve discount factors, avoiding a silent mismatch in floating-point input conventions.

The grid has **480 parameter scenarios, each with two prices**: 432 nominal zero-log-drift scenarios and 48 nonzero-drift controls. This is not 960 independent scenarios. Volatilities are 0.1/0.2/0.4, half log widths 0.1/0.25/0.5, maturities 7/30/180/730 days, several log-barrier positions, rates and cash payments. Exact combinations and inputs are in grid.py and JSON rows. Error tolerance is 2e-10 after dividing no-touch price by discounted payment; range slack is 1e-12.

| Implementation | Oracle failures / 480 | Price-bound failures / 480 | Focused test failures / 89 |
|---|---:|---:|---:|
| Original current master | 360 | 72 | 56 |
| Remove early exit only | 48 | 24 | 13 |
| Final candidate | 0 | 0 | 0 |
| Restore original implementation | 360 | 72 | 56 |
| Official 1.1.2 | 360 | 72 | 56 |

Maximum corrected normalized error was 1.1102230246251565e-15. All 48 nonzero-drift grid controls and all 120 previously passing scenarios remain within the declared tolerance; they are **not claimed byte-identical** after correction. Touch/no-touch parity passes every variant, with normalized sum error at most about 2.22e-16. Current-master numerical rows reproduce the earlier experiment's rows; this does not imply byte identity of complete files containing changed provenance metadata.

## Proposed upstream regression tests

The submitted PR adds 89 tests: one minimal negative-price regression, 72 independent zero-drift reflected-density comparisons and 16 nonzero-drift bounds/parity checks. They call the actual public pricing API and use standard-library Gaussian interval calculations, without adding mpmath as an upstream test dependency. The exact PR plus 14 adjacent existing FX tests passed locally: **103 tests**.

Current master renamed the curve class/module to FlatDiscountCurve. The released-version test copy uses a one-line test-only import alias for DiscountCurveFlat; no library compatibility monkeypatch was applied. An initial collection error from the obsolete import was resolved before the authoritative runs and is retained locally as an operational record.

## Maintainer and duplicate status

The [maintainer independently reproduced the invalid prices](https://github.com/domokane/FinancePy/issues/266#issuecomment-5698752302) and requested a focused fix. [PR270](https://github.com/domokane/FinancePy/pull/270) is open, with both observed GitHub checks successful as of 17 September. These checks do not establish acceptance, merging or availability in a new package release.

Before publication, current source/history, public issues/PRs and GERO catalogs were checked. The known original issue266 was found and is credited; no separate exact duplicate or intervening fix was found in the reviewed material. This is the archival release of that existing report, not a new discovery on the publication date. Source links and the bounds of the review appear in SOURCE_LEDGER.md.

## Reproduce and inspect

See README.md for a clean environment and `python -B reproduce.py --output /path/to/new-output`. The package contains the complete pinned current-source archive, official release wheel, their licenses, candidate, 89 tests, independent oracle and recorded outputs. It verifies vendor/source hashes, copies pristine variants, recomputes the reference, runs all variants sequentially and compares the expected failure counts. Output must be a new directory. It does not modify a user's installed FinancePy package.

Reference environment: macOS arm64, Python 3.12.14, NumPy 2.3.5, Numba 0.62.1, SciPy 1.16.3, mpmath 1.3.0 and pytest 9.1.1, with one configured numerical worker. Environment details are preserved in replay receipts. Dependency wheels are not bundled; initial environment setup needs network access.

## Limits

This study covers the displayed cash-at-expiry contracts and sampled positive-volatility, positive-time inputs with spot strictly between positive barriers. It is not a Monte Carlo validation, Greeks study, performance benchmark or complete local upstream-suite run. It does not establish how often these inputs occur, use by any bank, live trade mispricing, customer losses, or behavior in all market regimes. The remaining term cap is explicitly outside any universal convergence claim.

The short video explains these recorded results. It uses original GERO graphics and synthetic English narration via edge-tts; no video or audio binary is included in this research archive.

## Video and archive

[31-second English video](https://youtube.com/shorts/DbnhzRiMV1s) · [Archive DOI](https://doi.org/10.5281/zenodo.22805090).

Archive SHA-256: `2da5da96c0f8a06c3a0d19df094a9769c8b1584c00cd8a3ac5a5a24aaca6d962`.

## Publication records

- [Github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/financepy-double-no-touch-series-truncation.md)
- [Gero](https://www.gero.uz/research/articles/financepy-double-no-touch-series-truncation.html)
- [Huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-double-no-touch-series-truncation.md)
- [Zenodo](https://zenodo.org/records/22805090)
- [Linkedin](https://www.linkedin.com/feed/update/urn:li:ugcPost:7506212381824315392/)
- [Youtube](https://youtube.com/shorts/DbnhzRiMV1s)

[Download the frozen reproduction archive](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/financepy-double-no-touch-series-truncation/gero-financepy-double-no-touch-research-2026-09-17.zip). No video binary is stored in this repository.
