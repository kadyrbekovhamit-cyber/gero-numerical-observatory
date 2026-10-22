# Missing observations are not zero returns: a PyPortfolioOpt correction

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

AI-assisted investigation, test development and writing. This report is not independent human certification.

**Status:** reproduced within the scope below; [upstream PR #771](https://github.com/PyPortfolio/PyPortfolioOpt/pull/771) is open, not accepted or released as of 2026-10-22T16:54:32.307532+00:00.

## Finding and correction

Automatic shrinkage estimators currently replace missing returns with zero. `semicovariance` also converts NaNs to zero via `fmin` and divides every pair by the full row count. An unobserved return is therefore treated as an observed flat day.

Proposed policy:
- Ledoit–Wolf (all three targets) and OAS use the common complete-return sample, warn with the number of dropped rows, and require at least two complete observations. Constant-correlation covariance and higher moments use that same sample, retaining the existing ddof convention. Manual `shrunk_covariance(delta)` remains unchanged.
- Semicovariance divides each downside-product sum by that asset pair's joint-observation count, rejects pairs with no overlap, and retains the existing PSD-repair option. Actual zero returns still count as observations.

This makes a statistical choice explicit: complete-case shrinkage can discard much of the older history; pairwise estimates can need PSD repair. The docs state these limitations. Feedback is welcome on the policy/API before merge.

Six old integration goldens change because they included the zero-imputation result. We cross-checked the updated moments with pairwise means and sklearn, the BL posterior with its precision-form expression, and five optimized objective values with an independent scipy SLSQP solve (largest objective gap below 5e-13), before updating those goldens.

Validation: original initial regressions 14 failed / 1 passed; patched risk_models + 16 regressions + Black–Litterman/CLA/efficient-frontier modules: 146 passed / 4 skipped. Tests include internal/leading gaps, input immutability, insufficient overlap, observed zeros and off-diagonal joint counts. On the repository stock-price fixture at benchmark=0, BABA downside variance is .00560896 with the old denominator versus .04465234 using observed days; the long-only minimum-risk weight changes from .532475 to .081924. This is a fixture example, not observed investment losses or a universal multiplier.

Related prior discussion: #270 (common history recommendation), #294 (zero-fill criticism in a different estimator), #500/#514 (new-stock overweight symptoms). We do not claim that the general missing-data issue is new. No matching current patch found in the reviewed open PRs. Tested main a6638d2, Python 3.12 / numpy 2.2.6 / pandas 3.0.6. AI-assisted investigation and patch, one numerical worker.

## Pinned source and reproduction

- Upstream [PyPortfolio/PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt)
- Base commit: `a6638d2e06dae6f444fd022cfd4b3c528902a85b`
- Proposed commit: `8a61b0fa778e771b357472e1a8c63719d464249e`
- [Reviewed source](https://github.com/PyPortfolio/PyPortfolioOpt/blob/a6638d2e06dae6f444fd022cfd4b3c528902a85b/pypfopt/risk_models.py)
- [Patch and regression tests](https://github.com/PyPortfolio/PyPortfolioOpt/pull/771/files)

```text
ARCHIVE=/absolute/path/to/extracted/archive
git clone https://github.com/PyPortfolio/PyPortfolioOpt.git source
cd source
git checkout a6638d2e06dae6f444fd022cfd4b3c528902a85b
python -m pytest --import-mode=importlib "$ARCHIVE/test_missing_return_estimation.py" -q
# Expected baseline failures are described above.
git apply "$ARCHIVE/proposed.patch"
python -m pytest tests/test_missing_return_estimation.py -q
```

The PyPortfolioOpt package was imported normally. The module-level test counts above include existing tests, not just new counterexamples. No market data was downloaded during reproduction.

The archive includes the patch, test, test transcripts, upstream license and verification metadata. Run numerical work with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1. The source-level tests do not establish an effect on a real investor or production service.

## Interpretation and prior work

At benchmark=0 on the repository stock-price fixture, the BABA diagonal is 0.0056089601 under the old denominator and 0.0446523364 using observed days, a factor of 7.96089 in **downside variance**, not volatility. The optimized minimum-risk weight changes from 53.2475% to 8.1924%. Different estimators, periods or benchmarks give different numbers. Complete-case filtering is a proposed data policy, not proof that an estimate predicts future risk.

[Discussion #270](https://github.com/PyPortfolio/PyPortfolioOpt/issues/270) recommended a common history; [#294](https://github.com/PyPortfolio/PyPortfolioOpt/issues/294) objected to zero-fill in another estimator; [#500](https://github.com/PyPortfolio/PyPortfolioOpt/issues/500) and [#514](https://github.com/PyPortfolio/PyPortfolioOpt/issues/514) reported short-history overweight symptoms. We credit these prior discussions and do not claim discovery of the general missing-data problem.

## Search boundary and license

Public open/closed issue searches, related comments and current open PRs were checked on 22 October 2026. No matching fix was found within that search. This is not an exhaustive priority guarantee.

Report: CC BY 4.0. Original tests and harness: MIT. Upstream code remains under MIT. No private correspondence, credentials or customer data is included.
