# Riskfolio Bayesian Black–Litterman: equilibrium weights use excess returns

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

AI-assisted investigation, test development and writing. This report is not independent human certification.

**Status:** reproduced within the scope below; [upstream PR #266](https://github.com/dcajasn/Riskfolio-Lib/pull/266) is open, not accepted or released as of 2026-10-22T16:54:43.377920+00:00.

## Status update before Zenodo deposit

The archived proposal is commit `bf6ee21bda17ef91a895726955c36d187bbc04aa`, which passes all four archived tests. The maintainer subsequently changed the PR head to `700fc19ee6955ed9d0e43623c500898ace22371d`; that version yields 2 passed / 2 failed on the same tests because it omits alpha from the weight formula. The intended alpha convention remains to be clarified. See [the status supplement](https://zenodo.org/records/23591085/files/STATUS_2026-10-23.md?download=1) for the exact distinction. The ZIP preserves the original 22 October evidence unchanged.

## Finding and correction

`black_litterman_bayesian` adds `rf` to its posterior return and then uses that total return in `inv(delta*cov) @ mu.T`. Its returned unconstrained equilibrium weights should instead satisfy `delta*cov @ w = mu.T - rf`, consistent with the excess-return inversion used by the adjacent BL routines.

The one-line change subtracts rf in the weight calculation. Returned expected returns and covariance stay unchanged; alpha remains part of expected excess returns when `const=True`.

Four synthetic tests check the first-order condition for rf=0 / .004 and with / without alpha. Before the patch the two nonzero-rate cases have residual .004 in every component (2 failed, 2 passed); after it all 4 pass.

Validation scope: master 7b1b26a, actual unmodified AST function bodies for the estimator and its historical-moment helpers, with real numpy 2.2.6 / pandas 3.0.6 / scipy 1.18.1. We did not load the full Riskfolio package or compile its extension. The included regression is a normal package-import pytest test for upstream CI. This concerns the directly returned weight vector; it does not establish a defect in the main Portfolio optimizer, which does not consume that vector.

AI-assisted investigation and patch, single-thread numerical execution. No measured production impact claimed.

## Pinned source and reproduction

- Upstream [dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)
- Base commit: `7b1b26ac9760974c12ecefe290a9d3bd37b66b7a`
- Proposed commit: `bf6ee21bda17ef91a895726955c36d187bbc04aa`
- [Reviewed source](https://github.com/dcajasn/Riskfolio-Lib/blob/7b1b26ac9760974c12ecefe290a9d3bd37b66b7a/riskfolio/src/ParamsEstimation.py)
- [Patch and regression tests](https://github.com/dcajasn/Riskfolio-Lib/pull/266/files)

```text
python run_riskfolio_isolated.py /path/to/Riskfolio-Lib /path/to/archive/test_bayesian_equilibrium_weights.py
```

The archived harness compiles the actual unchanged function bodies and their historical-moment helpers from the pinned source. It uses real numpy/pandas/scipy; no numerical stub substitutes their results. This is an isolated source-level run, not a full Riskfolio installation or compiled-extension test. Pass the archived test path as shown. Run before and after applying proposed.patch to the pinned repository; the patch also adds the test for normal upstream pytest execution.

The archive includes the patch, test, test transcripts, upstream license and verification metadata. Run numerical work with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1. The source-level tests do not establish an effect on a real investor or production service.

## Search boundary and license

Public open/closed issue searches, related comments and current open PRs were checked on 22 October 2026. No matching fix was found within that search. This is not an exhaustive priority guarantee.

Report: CC BY 4.0. Original tests and harness: MIT. Upstream code remains under BSD-3-Clause. No private correspondence, credentials or customer data is included.
