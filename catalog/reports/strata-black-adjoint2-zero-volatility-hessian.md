# Strata: a zero-volatility Hessian prevents custom-model option pricing

Xamit Kadirbekov · Independent GERO research · 19 September 2026 (Asia/Tashkent)

A finite option price does not guarantee that its derivatives are usable. In the official OpenGamma Strata 2.12.74 implementation, `priceAdjoint2(110, 100, 1, 0, true)` returns price **10** and correct first derivatives, but its entire 3-by-3 Hessian is **NaN**. Away from the payoff kink, the zero-volatility boundary has zero second derivatives.

The discrepancy also interrupts another actual Strata calculation. Through the documented custom-volatility-provider interface, a constant-zero test model causes `SabrExtrapolationRightFunction` to throw while constructing its price extrapolator. A narrow Black-formula correction restores the native constructor and subsequent price outputs; restoring the original implementation reproduces the interruption.

## Executed chain: derivative → fitting decision → unavailable price

The following chain was executed on the official release, pinned current classes, local candidate and restored original:

1. `BlackFormulaRepository.priceAdjoint2` produces a NaN strike curvature at exact zero volatility.
2. `SabrExtrapolationRightFunction.computesFittingParameters` incorporates it into `priceK[2]`. Its existing small-price fallback requires all three values to be small; the NaN comparison fails.
3. The native root bracketer then throws `MathException: Failed to bracket root: function invalid at x = -1.0 f(x) = NaN`. The pricing object is not constructed, so the requested prices cannot be returned through it.

For forward 0.03, cutoff 0.06, expiry 1, tail parameter 4 and a constant-zero volatility provider:

| Measured output | Released / current / restored | Local candidate |
|---|---|---|
| Construct native extrapolator | Exception | Success |
| Call price at strike 0.015 | Unavailable through this object | 0.015 |
| Put price at strike 0.09 | Unavailable through this object | 0.06 |
| Call price at strike 0.09 | Unavailable through this object | 5.6699831977150385e-40 |

The six zero-provider scenarios use forwards 0.03, 0.05 and 1, cutoff twice the forward, and expiries 1 and 5. **6/6 constructions fail before, 0/6 fail with the candidate, and 6/6 fail after restoration.** Fifteen other downstream scenarios retain identical outputs.

The constant provider is a **GERO test fixture** with exact zero first and second derivatives. The constructor, fitting logic, exception, fallback and pricing methods are actual Strata code. This is evidence about a supported extension path, not a deployed bank integration. The default Hagan provider floors its second-adjoint volatility at 1e-6 (or can become NaN earlier), so these results do not demonstrate this exact failure in the default Hagan/CMS configuration.

The native fallback uses parameters [-100, 0, 0]. Its positive call tail is an intentional small approximation, not exactly the deterministic zero price. The test does not claim that every downstream sensitivity is repaired; SABR-parameter sensitivities and other fitting regions were not measured.

## Independent mathematical result

For positive finite forward F, strike K and expiry T, with F different from K, the zero-volatility payoff is locally linear in F and K. At F=110, K=100, T=1, a call therefore has price 10, gradient [1,-1,0,0] and zero Hessian in [F,K,volatility]. Volatility derivatives are interpreted from the right in the nonnegative domain.

For fixed non-ATM log-moneyness m=log(F/K), normal density in the Black formula contains `exp(-m*m/(2*sigma*sigma*T))`. As sigma approaches zero from above, this term vanishes faster than any inverse power of sigma. The mixed and volatility second derivatives tend to zero as well. This reasoning excludes ATM; assigning a zero Hessian at its payoff kink is not supported.

The existing second backward sweep divides by powers of `volPeriod` and combines zero densities with infinite intermediates. The candidate adds the ordinary deterministic limit before this sweep, restricted to finite positive inputs, exact zero volatility, F and K <= the class's LARGE threshold, and abs(F-K) >= its SMALL threshold. Near-ATM, large-level reference conventions and all positive-volatility paths are retained.

## Versions and controls

- Official report-tool release: **2.12.74**, JAR SHA-256 `0fb4c6c778f25b7c6fe13c001f47a4e404868df55e4dd9841c752128bb864125`.
- Current source pin: **987932ee95bf53e2baaff9a6b8e738a00f558b10**.
- Black source SHA-256: `029f9e8708292cc015c5302d0119abfed1760eaa2d6ddc0fcb9274fba32dd22c`.
- Execution: macOS arm64, OpenJDK 25.0.4.1, one configured JVM processor with SerialGC and interpreted execution. Current affected classes were compiled against released dependencies; this is not a complete build of main.

| Direct test state | Non-ATM zero-volatility failures |
|---|---:|
| Official release | 180 |
| Pinned current source | 180 |
| Local candidate | 0 |
| Original restored | 180 |

The direct grid has **866 distinct input rows**. All **686 non-target rows** remain identical. Within those controls, 644 positive-volatility rows agree with an independently formulated 80-digit mpmath oracle within relative tolerance 2e-8 and absolute tolerance 1e-10. Fourteen rows reuse existing upstream test inputs; no full upstream JUnit run is claimed.

Remaining rows deliberately preserve 12 tiny-positive-volatility stress failures at 1e-200, six ATM conventions and 24 boundary controls. Downstream, six tiny-positive failures and the default-alpha-zero failure likewise remain unchanged. These are not counted as fixed. The 21-row downstream grid extends the evidence for the same implementation defect; it is not 21 new defects or a second discovery.

The archive contains the exact minimal Java snippet, candidate patch, pinned source, test fixtures, independent oracle, all raw output sets and replay receipts. No production frequency, bank/customer exposure, charges, losses, security issue or legal violation was measured. The effect demonstrated here is loss of availability of a pricing calculation under the stated synthetic custom-model conditions.

## Sources, duplicate review and disclosure

- [Pinned Black implementation](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/option/BlackFormulaRepository.java#L217).
- [Native extrapolator fitting](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/option/SabrExtrapolationRightFunction.java#L368).
- [Documented custom-provider interface](https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/impl/option/SabrExtrapolationRightFunction.html).
- [Hagan second-adjoint floor](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/volatility/smile/SabrHaganVolatilityFunctionProvider.java#L381).
- Historical work [PR843](https://github.com/OpenGamma/Strata/pull/843/files) introduced the method; [PR939](https://github.com/OpenGamma/Strata/pull/939/files) changed the Hagan floor and separate finite differences. Both are credited, not presented as our work.

Bounded all-state searches included issue/PR titles, bodies and comments, actual relevant diffs, public documentation and the GERO catalog. No exact previous report or proposed correction for this specific Black zero-volatility Hessian was found. Private, unindexed or differently described reports may not have been captured. This is separate from the earlier Normal-IV initial-guess case.

**Vendor-first report:** [OpenGamma Strata issue2797](https://github.com/OpenGamma/Strata/issues/2797). Sent and publicly verified before distribution of this report. No acknowledgment or acceptance is claimed. Source review, test/code preparation and documentation were AI-assisted.


[Complete offline reproducibility archive](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/strata-black-adjoint2-zero-volatility-hessian/gero-strata-black-zero-volatility-evidence-2026-09-19.zip) · SHA-256 `3a6a81f884c71c72d76a93540140ec12ce2ad9f65e67267a50bce4fb3e93eff5`. The ZIP includes the official JAR, pinned source, oracle and all output sets. Video has not yet been produced.

## Verified publication links

- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/strata-black-adjoint2-zero-volatility-hessian.md)
- [gero](https://www.gero.uz/research/articles/strata-black-adjoint2-zero-volatility-hessian.html)
- [zenodo](https://zenodo.org/records/22840914)
- [linkedin](https://www.linkedin.com/feed/update/urn:li:share:7506941906036334592/)
- [huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/strata-black-adjoint2-zero-volatility-hessian.md)

Zenodo DOI: `10.5281/zenodo.22840914`. YouTube: queued, not produced or uploaded.
