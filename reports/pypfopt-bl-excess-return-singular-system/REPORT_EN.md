# Black–Litterman weights: excess returns and the singular fallback

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

AI-assisted investigation, test development and writing. This report is not independent human certification.

**Status:** reproduced within the scope below; [upstream PR #769](https://github.com/PyPortfolio/PyPortfolioOpt/pull/769) is open, not accepted or released as of 2026-10-22T16:54:08.155008+00:00.

## Finding and correction

With a market prior and views equal to that prior, `bl_weights()` should recover the market weights. For `cov=diag(.04,.09,.16)`, caps `[500,300,200]`, delta=2 and rf=.04, it returns `[.541353,.282707,.175940]` instead of `[.5,.3,.2]`: the inversion uses total rather than excess returns.

This retains the constructor's existing `risk_free_rate` kwarg and subtracts it only when computing implied weights. Explicit total-return priors can pass the same rate; zero remains the default for excess-return priors. Posterior returns, posterior-covariance convention and normalization are unchanged.

The same weight solver's singular fallback incorrectly uses the cached view-space matrix `self._A`. The patch uses its local asset-space matrix `delta*cov` instead. With duplicate A/C assets and one view it previously raises incompatible dimensions; with three views it silently returns equal thirds instead of `[.3,.4,.3]`.

Validation: the original six regressions gave 4 failures / 2 passes. The complete Black–Litterman module plus seven new cases passes: 28 passed. Tests include zero/nonzero rf, explicit/market priors, both singular shapes and unchanged handling of duplicate exact views. Related #541 introduced the fallback; #762 concerns a different posterior-covariance convention; #763/#764 concern raw-weight normalization, which this PR does not alter.

Tested against main a6638d2, Python 3.12, numpy 2.2.6, pandas 3.0.6; sequential single-thread numerical execution. AI-assisted investigation and patch; reproducible tests included. No production-loss claim.

## Pinned source and reproduction

- Upstream [PyPortfolio/PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt)
- Base commit: `a6638d2e06dae6f444fd022cfd4b3c528902a85b`
- Proposed commit: `98358abad21061d3a9299e3dd9d5df68e5df9b42`
- [Reviewed source](https://github.com/PyPortfolio/PyPortfolioOpt/blob/a6638d2e06dae6f444fd022cfd4b3c528902a85b/pypfopt/black_litterman.py)
- [Patch and regression tests](https://github.com/PyPortfolio/PyPortfolioOpt/pull/769/files)

```text
ARCHIVE=/absolute/path/to/extracted/archive
git clone https://github.com/PyPortfolio/PyPortfolioOpt.git source
cd source
git checkout a6638d2e06dae6f444fd022cfd4b3c528902a85b
python -m pytest --import-mode=importlib "$ARCHIVE/test_black_litterman_regressions.py" -q
# Expected baseline failures are described above.
git apply "$ARCHIVE/proposed.patch"
python -m pytest tests/test_black_litterman_regressions.py -q
```

The PyPortfolioOpt package was imported normally. The module-level test counts above include existing tests, not just new counterexamples. No market data was downloaded during reproduction.

The archive includes the patch, test, test transcripts, upstream license and verification metadata. Run numerical work with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1. The source-level tests do not establish an effect on a real investor or production service.

## Search boundary and license

Public open/closed issue searches, related comments and current open PRs were checked on 22 October 2026. No matching fix was found within that search. This is not an exhaustive priority guarantee.

Report: CC BY 4.0. Original tests and harness: MIT. Upstream code remains under MIT. No private correspondence, credentials or customer data is included.
