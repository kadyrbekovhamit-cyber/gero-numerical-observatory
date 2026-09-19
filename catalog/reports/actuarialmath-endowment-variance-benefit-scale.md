# Endowment variance scales the benefit twice in actuarialmath

Xamit Kadirbekov · GERO · 19 September 2026

A one-year payout of 2 with probability 1/4 or 1 with probability 3/4 has variance 0.1875. The public `LifeTable.endowment_insurance(..., moment=VARIANCE)` path returns 0.75. Its positive first and second moments are correct; the variance caller applies the death-benefit scale a second time.

This is one newly verified implementation cause, separate from the previously reported whole-life mean-squaring and ConstantForce benefit-omission issues. It is not a claim of a recent regression: the inspected source history contains this pattern from June 2023.

## Executed versions and independent reference

- Current main: `7d18f11ad304898f177b7922b3c53f70e4c2b4f4`, refreshed 19 September 2026. All 91 cached source files match the fresh upstream Git tree and were hash-verified after execution.
- Separately extracted official PyPI **1.1.0** wheel, SHA-256 `b19990e4378aaa19fe6bc1182b4269faec6617cb62b0677fea1e624fbbb3ff6f`. Its 22 package files were checked against the official artifact; the target source matches current main. Source-tree project metadata says 1.0.1; that is not the PyPI artifact version.
- Python 3.12.14, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, matplotlib 3.10.8 and IPython 9.17.1, macOS arm64. CPU-only, one configured numerical worker; no media playback.

The independent oracle enumerates every outcome in finite one-, two- and three-year synthetic lifetime tables, using exact rational probabilities, benefits and discount factors. It computes the mean and variance directly from payouts, without calling another actuarial formula. For a one-year contract:

`Var(Z) = q * (1-q) * (b-e)^2 * v^2`.

The 1,300-scenario grid includes zero and certain death probabilities, zero/unit/non-unit benefits, unequal and default equal endowments, deterministic controls and rates 0%, 1%, 5% and 20%. The preselected absolute variance tolerance is `256 * machine_epsilon * max(1, b^2, e^2)`; complete inputs and individual tolerances are archived.

| Implementation | Variance mismatches / 1,300 | Variance-support bound violations |
|---|---:|---:|
| Current main |483|375|
| Official 1.1.0 |483|375|
| Narrow local candidate |0|0|
| Original line restored |483|375|
| Earlier whole-life correction only |483|375|

First and second moments pass in all variants and remain exactly equal in the recorded outputs. All three reported outputs remain identical in 794 scenarios. Another 23 previously passing scenarios change only within the retained rounding allowance (maximum absolute change about `4.0e-15`); do not describe all 817 prior passes as bitwise unchanged. Four of six focused regression methods fail on original/release/restored source, with six failure entries because the unit-conversion method has three failing subtests; all six methods pass with the candidate. This is not a full upstream-suite result.

## Formula → serialized document → measured decision

The real upstream APIs `endowment_insurance` and `Life.portfolio_percentile` were executed in sequence. GERO supplies their parameter wiring, writes a synthetic risk-summary JSON/HTML document, then an example monitor reads that JSON and compares the reported amount with a disclosed limit. This is a demonstration adapter, not an insurer's application.

For 100 independent one-year policies, each pays 100 on death (probability 0.25) or 50 on survival, at zero interest:

- Mean aggregate payout: **6,250**, unchanged.
- Correct single-policy variance: **468.75**; affected variance: **4,687,500**.
- Native normal-approximation 95th-percentile estimate: **41,862.125661** before the correction, **6,606.121257** after it. The change is **35,256.004405** monetary units.
- Synthetic monitor limit: **7,000**. Its serialized-document decision changes from `EXCEEDS_EXAMPLE_LIMIT` to `WITHIN_EXAMPLE_LIMIT`.
- The exact finite-distribution 95th percentile is **6,600**. The corrected normal approximation retains its separate **6.121257** approximation difference; it is not presented as an exact quantile. Maximum possible aggregate payout is **10,000**.

A second example demonstrates the opposite direction. With death benefit zero and survival benefit 100, the affected variance becomes zero. The native estimate is 7,500 instead of the corrected 8,212.242513; a synthetic limit of 7,900 is incorrectly passed. The exact discrete percentile is 8,200. Two further unit-benefit/deterministic controls remain unchanged. In four downstream scenarios, two amount and decision discrepancies become zero with the candidate and return when the original line is restored.

These differences concern computed amounts and example decisions. No premium charged, reserve requirement, solvency rule, real underwriting decision, insurer deployment or customer loss has been inspected.

## Cause and candidate correction

[Endowment variance source](https://github.com/terence-lim/actuarialmath/blob/7d18f11ad304898f177b7922b3c53f70e4c2b4f4/src/actuarialmath/insurance.py#L193) recursively obtains moments with `b` and `endowment` already included. It then calls a helper that multiplies their variance by `b**2` again.

The candidate changes only that endowment caller from `b=b` to `b=1`. The shared helper remains unchanged for callers providing unit-benefit moments. This restores quadratic monetary-unit scaling; the original path scales as the fourth power when both benefits are rescaled.

The earlier whole-life patch alone was executed as a negative attribution control and leaves all 483 mismatches. The ConstantForce class is not on the tested LifeTable path. Positive moments are not patched.

## Duplicate review and limitations

The recorded bounded search refreshed all five upstream issue/PR bodies, three comments, the full current two-file PR #2 diff, nine `insurance.py` history diffs, the guide repository's empty issue list, eight exact GitHub searches and the canonical GERO catalog/previous reports. No exact prior report or proposed correction was found. Private, deleted and unindexed material and every fork were not searched; global priority is not proved.

[Official insurance guide](https://actuarialmath-guide.readthedocs.io/en/latest/insurance.html) documents the moment/variance contract. Runtime evidence covers the discrete finite-state paths described above. Continuous endowment paths, full upstream tests, performance and real insurer exposure were not evaluated. Other known library defects remain outside this narrow patch. Research and preparation were AI-assisted. Developer submission and public-distribution receipts are tracked separately; a local report does not imply either.


Developer report sent and independently read back on 19 September 2026: https://github.com/terence-lim/actuarialmath/issues/6 . Acknowledgment or acceptance is not claimed. Broader publication and video remain separately tracked.

## Publication and reproduction

[Zenodo / DOI10.5281/zenodo.22843200](https://zenodo.org/records/22843200) · [Complete reproducibility ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/actuarialmath-endowment-variance-benefit-scale/gero-actuarialmath-endowment-variance-evidence-2026-09-19.zip). SHA-256 `59f6da29073136146139e9feade3fa504eec7c1ff37c7c2ffd16b8f65c041c2c`. All six verified publication links are listed below.

## Verified publication links

- [zenodo](https://zenodo.org/records/22843200)
- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/actuarialmath-endowment-variance-benefit-scale.md)
- [gero](https://www.gero.uz/research/articles/actuarialmath-endowment-variance-benefit-scale.html)
- [youtube](https://youtube.com/shorts/h8JZ-kw4Ot0)
- [linkedin](https://www.linkedin.com/feed/update/urn:li:share:7507014387195531264/)
- [huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/actuarialmath-endowment-variance-benefit-scale.md)

Zenodo DOI: `10.5281/zenodo.22843200`. YouTube: public, 32-second English Short.

[Evidence-archive follow-up in the existing developer issue](https://github.com/terence-lim/actuarialmath/issues/6#issuecomment-5740912028) was sent and read back. No maintainer acknowledgment or acceptance is claimed.
