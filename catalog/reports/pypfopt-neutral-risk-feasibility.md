# Market-neutral portfolios rejected by a fully-invested risk bound

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

AI-assisted investigation, test development and writing. This report is not independent human certification.

**Status:** reproduced within the scope below; [upstream PR #770](https://github.com/PyPortfolio/PyPortfolioOpt/pull/770) is open, not accepted or released as of 2026-10-22T16:54:21.298883+00:00.

## Finding and correction

`efficient_risk(..., market_neutral=True)` currently applies the sum(weights)=1 global-minimum-volatility lower bound before solving the sum(weights)=0 problem. That rejects feasible low-risk neutral portfolios.

For returns `[.1,0]`, covariance `.04*I`, bounds `(-1,1)` and target volatility `.05`, weights `[.05/sqrt(.08),-.05/sqrt(.08)]` are feasible and analytically optimal. The current precheck rejects them because the fully-invested lower bound is sqrt(.02).

Apply that precheck only when `market_neutral=False`; the solver still evaluates all user constraints. Tests cover three low-risk targets, updating the target parameter, preservation of the fully-invested rejection and a genuinely infeasible neutral constraint.

Validation: original regressions 5 failed / 1 passed; patched efficient-frontier module plus regressions 77 passed / 4 skipped. Main a6638d2; Python 3.12, cvxpy 1.9.3. Related #569 discusses minimum-volatility rejections but does not supply this market-neutral counterexample; #362 concerns sector constraints, which this patch does not change.

AI-assisted investigation and patch; numerical work was sequential and single-threaded. This is a feasibility-check correction, not a claim about production portfolios.

## Pinned source and reproduction

- Upstream [PyPortfolio/PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt)
- Base commit: `a6638d2e06dae6f444fd022cfd4b3c528902a85b`
- Proposed commit: `9d9c063d5d8f46e44907e8c486386335340424a3`
- [Reviewed source](https://github.com/PyPortfolio/PyPortfolioOpt/blob/a6638d2e06dae6f444fd022cfd4b3c528902a85b/pypfopt/efficient_frontier/efficient_frontier.py)
- [Patch and regression tests](https://github.com/PyPortfolio/PyPortfolioOpt/pull/770/files)

```text
ARCHIVE=/absolute/path/to/extracted/archive
git clone https://github.com/PyPortfolio/PyPortfolioOpt.git source
cd source
git checkout a6638d2e06dae6f444fd022cfd4b3c528902a85b
python -m pytest --import-mode=importlib "$ARCHIVE/test_market_neutral_risk_regression.py" -q
# Expected baseline failures are described above.
git apply "$ARCHIVE/proposed.patch"
python -m pytest tests/test_market_neutral_risk_regression.py -q
```

The PyPortfolioOpt package was imported normally. The module-level test counts above include existing tests, not just new counterexamples. No market data was downloaded during reproduction.

The archive includes the patch, test, test transcripts, upstream license and verification metadata. Run numerical work with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1. The source-level tests do not establish an effect on a real investor or production service.

## Search boundary and license

Public open/closed issue searches, related comments and current open PRs were checked on 22 October 2026. No matching fix was found within that search. This is not an exhaustive priority guarantee.

Report: CC BY 4.0. Original tests and harness: MIT. Upstream code remains under MIT. No private correspondence, credentials or customer data is included.
