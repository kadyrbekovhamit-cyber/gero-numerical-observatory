# FinancePy adjusted binomial: invalid support points produce negative probabilities and shift the mean

Independent GERO research by Xamit Kadirbekov. Original finding and maintainer issue: 15 September 2026. Pristine current-source replay and archival preparation: 17 September 2026. AI-assisted research and editorial preparation; retained executed evidence governs the claims.

**Confirmed on the actual Numba-compiled FinancePy implementation and the released PyPI 1.1.2 package.** For four independent, equal-loss credits with default probabilities `[0.01, 0.01, 0.5, 0.99]`, the adjusted-binomial function assigns probability `−0.1127312875` to three defaults. Its public credit-tranche calculation returns `1.1004521991` for a quantity that must lie between zero and one. A separate two-credit example changes the required mean from `1.60` to `1.65`.

The affected method is deliberately approximate. This report checks valid probability mass and its intended moment constraints; it does not require an approximate distribution to equal the exact distribution at every point for general portfolios.

## Implementation and scope

- Repository: [domokane/FinancePy](https://github.com/domokane/FinancePy).
- Original source: `2b9227fea9d832c4033421d6cd53a54316414fca`. Fresh replay uses current master `4c7cd50bdadd6efc9ac74fa81e93374397d18e3e`; the target bytes are identical. The complete current archive has 792 verified files, including 233 pristine package files; the official wheel has 219 package files.
- Target: [`financepy/models/loss_dbn_builder.py`](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/financepy/models/loss_dbn_builder.py), `indep_loss_dbn_hetero_adj_binomial`.
- Independently executed released distribution: **FinancePy 1.1.2**, the latest version returned by PyPI during this check. All 219 package files match the saved official wheel; target bytes match current source.
- Python 3.12.14, NumPy 2.3.5, Numba 0.62.1, macOS 15.5 arm64. One configured numerical thread. Native compiled signatures are saved, rather than using Numba's `.py_func` or a rewritten approximation.
- Loss ratios are **all one**. Thus each default contributes one loss unit. Unequal individual loss amounts are outside the verified correction scope.

Both the current-source replay and the official wheel print a 1.1.2 banner with different build dates. The exact Git pin and official wheel checksum identify the executed versions.

## Mathematical invariants

For independent Bernoulli default indicators with stored probabilities `p_i`, and equal loss sizes, let `N` be the number of defaults. The probability mass on `0,…,n` must be nonnegative and sum to one. Its exact moments are

`E[N] = sum(p_i)` and `Var(N) = sum(p_i * (1 - p_i))`.

These follow directly from independence. The main oracle evaluates them with Python `Fraction` on the exact stored binary inputs, independently of FinancePy. For one and two credits only, it additionally constructs the entire exact probability polynomial by rational convolution. On a three-point support, normalization and the first two moments determine the entire distribution.

The adjusted-binomial algorithm and the idea of matching moments are existing work, not a new mathematical method: see [O'Kane's paper record](https://ssrn.com/abstract=2283729). [QuantLib's independent implementation](https://github.com/lballabio/QuantLib/blob/master/ql/experimental/credit/binomiallossmodel.hpp) documents the two-moment construction and selects adjacent floor/upper support points. QuantLib was reviewed as source context, not executed as a pricing oracle in this experiment.

## Minimal observed examples

```python
import numpy as np
from financepy.models.loss_dbn_builder import indep_loss_dbn_hetero_adj_binomial

p = np.array([0.7, 0.9])
q = indep_loss_dbn_hetero_adj_binomial(2, p, np.ones(2))
print(q, np.arange(3) @ q)
# Original: [0.035, 0.280, 0.685], mean 1.6500000000000001
# Exact:    [0.030, 0.340, 0.630], mean 1.6
```

For `[0.01, 0.01, 0.5, 0.99]`, the original probability of three defaults is `−0.11273128751420225`. A probability cannot be negative even when its calculation uses an approximation.

The original code computes both support points with nearest-integer `round`. This can put both points above the mean, collapse them at the upper endpoint, or produce a two-point gap at ties-to-even boundaries. The subsequent redistribution formulas assume **two adjacent support points bracketing the mean**. Depending on the input, the mismatch creates negative mass or changes the first moment.

The local candidate replaces only this support selection:

```python
mean_below = min(int(np.floor(mean_loss)), num_credits - 1)
mean_above = mean_below + 1
```

For `n >= 1` and admissible probabilities, these points remain adjacent and bracket the mean, including `mean_loss == n`. The original binomial recursion, variance expressions, integration, denominator guard and tolerances remain unchanged. No clipping or renormalization hides the failures.

## Recorded grid and restoration

The frozen principal grid contains **2,844 distinct input vectors**, with 1–125 credits:

| Check | Original source | Candidate | Original restored | Release 1.1.2 |
| --- | ---: | ---: | ---: | ---: |
| Vectors failing one or more declared invariants | 295 | 0 | 295 | 295 |
| Probability-range failures | 186 | 0 | 186 | 186 |
| Mean failures | 118 | 0 | 118 | 118 |
| Variance failures | 118 | 0 | 118 | 118 |
| Nonfinite outputs or total-mass failures | 0 | 0 | 0 | 0 |

Failure categories overlap. This is one support-selection defect, not 295 independent defects. The most negative principal-grid probability is `−0.13158250382396153`.

The grid consists of 1,353 small multiset portfolios, 81 explicitly grouped homogeneous controls, 250 deterministic controls, 460 dyadic/adjacent-float boundary vectors and 700 seeded heterogeneous-probability portfolios. Earlier groups also contain homogeneous cases; the group counts are disjoint because identical input vectors are deduplicated.

All **2,549 previously passing vectors still satisfy the invariants**. Their full distributions are not all unchanged: 870 change by more than `5e-12`, consistent with using different adjustment support points. The 331 explicitly grouped homogeneous/deterministic controls differ by at most `1.68e-14` per probability. **89 separate reversed-order checks per variant** also pass; they are not added to the main 2,844-vector count.

Original, restored and released raw main-grid outputs are byte-identical. All 233 current package files remain unchanged; the candidate alters exactly the target file. The existing upstream loss-distribution test passes on both original and candidate code. It covers nine factor-loading settings, but is not the full FinancePy suite.

The probability/mass tolerance is `5e-12`; mean tolerance is `5e-12 * max(1,n)`; second-moment and variance tolerance is `5e-12 * max(1,n*n)`. The same predeclared tolerances apply to every variant. Raw outputs, frozen inputs, scripts and checksums are retained.

## Effect through public credit-tranche functions

The additional run calls the actual `loss_dbn_hetero_adj_binomial` and `tranche_surv_prob_adj_binomial` functions, using four portfolio definitions, factor loadings `0`, `0.3`, `0.7`, and 2,000 integration steps: **12 separate scenarios per variant**.

The tranche function's returned quantity is `1 - expected tranche loss / tranche width`. It is an expected surviving fraction, not the literal probability that every borrower survives; it nevertheless must remain in `[0,1]` for the specified nonnegative loss model.

With four credits, probabilities `[0.01,0.01,0.5,0.99]`, zero recovery, zero factor loading and attachment/detachment `0.50/0.75`:

| Result | Returned tranche quantity |
| --- | ---: |
| Original / restored / release | 1.1004521990528198 |
| Candidate approximation | 0.9933384814020053 |
| Exact independent four-credit enumeration | approximately 0.9900995 |

The candidate fixes the invalid bound and moments; **the remaining difference from the exact four-credit distribution is approximation error**, not a claim of exact pricing. For the two-credit `[0.7,0.9]` full-portfolio case, original surviving fraction is `0.1750000417` versus the required `0.2`; candidate is `0.2000000246`.

Across the 12 scenarios, original code has three distribution-bound violations, three tranche-bound violations and four full-portfolio mean violations. Candidate has none under the declared integration tolerance of `2e-6` (mean tolerance scaled by `n`). Restoring the original code reproduces the same outputs. The approximately `2e-9` missing Gaussian tail mass and normal-CDF approximation were not repaired or presented as a new defect.

## Current-source replay and duplicate review

On 17 September 2026, the complete 792-file current-master archive was verified against Git blob identities. Fresh baseline, candidate and restored variants use 233 pristine package files, while the official 1.1.2 wheel contributes 219 package members verified against its RECORD. The frozen 2,844 input vectors, rational oracle and tolerances are unchanged. Every one of 12 raw grid/permutation/integration JSON files exactly reproduces the 15 September results. The existing official loss-distribution test passes on baseline and candidate. This is a new replay of the existing finding, not another discovery.

The refreshed bounded search covers 260 public issue/PR title-and-body records, eight target-path history entries, the 104-report canonical catalogue, GERO/HuggingFace catalogs and an owner-scoped Zenodo search. Existing own [issue 265](https://github.com/domokane/FinancePy/issues/265) is credited. The other term matches concern bond schedules/OAS, CDS integration dates and option-tree methods; no separate exact duplicate or intervening target fix was found in the reviewed material. Comments and private reports were not exhaustively searched. Details and raw responses are in review/.

The existing [English YouTube Short](https://youtube.com/shorts/weirECBnxW4) explains the original experiment and is not a new video discovery. No media binary is included in this archive. Maintainer issue 265 is open without comments as last checked; submission is not confirmation, merging or a new release. No PR for this support-selection change is claimed in this archival package.

## Reproduce and limitations

See README.md and run `python -B reproduce.py --output /path/to/new-directory`. The runner checks package hashes, extracts pristine source and official release members, applies and explicitly restores the support-selection patch, and executes all variants sequentially. Output must not already exist. The independent moment and short-portfolio convolution oracle uses exact rational arithmetic on the stored binary probabilities.

This is synthetic evidence for an implementation error. No bank deployment, investor loss, insurer use, complete trade valuation, performance improvement, full upstream test suite or upstream acceptance is established. Unequal loss ratios and very large portfolios beyond the recorded grid require separate assessment. Corrected general-portfolio probability shape remains approximate; no clipping or exact-general-pricing claim. Original issue attachment 37e3c3c1ab3d31bd57a24828c7dde76243ab2b4b3151aa4e6cd94ab1618a8ba8 remains an immutable historical artifact; this current-source package is a new archival edition, not a replacement of the earlier experiments.
