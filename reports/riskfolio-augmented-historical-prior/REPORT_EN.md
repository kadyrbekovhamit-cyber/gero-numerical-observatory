# Augmented Black–Litterman: historical prior must retain the joint state

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

AI-assisted investigation, test development and writing. This report is not independent human certification.

**Status:** reproduced within the scope below; [upstream PR #267](https://github.com/dcajasn/Riskfolio-Lib/pull/267) is open, not accepted or released as of 2026-10-22T16:54:56.540041+00:00.

## Finding and correction

With `eq=False`, `augmented_black_litterman` builds an (N+K)-dimensional augmented covariance, but selects an N-dimensional prior for asset-only views and a K-dimensional prior for factor-only views. With three assets / two factors these valid inputs raise `matmul` dimension errors (3 vs 5, or 2 vs 5).

Use the already constructed `mu_vstack - rf` in both branches, as the mixed-view branch already does. View selection changes the observation matrix, not the dimension of the joint asset-factor state.

The six regressions cover asset/factor/mixed views and both historical/equilibrium priors. They compare returned means, covariances and weights against an independent Gaussian-conditioning (gain-form) expression, rather than repeating the implementation's inverse-precision formula. Before: 2 failed / 4 passed. After: 6 passed.

Validation scope: master 7b1b26a, actual AST function bodies and historical mean/covariance helpers executed with real numpy/pandas/scipy; full package and compiled extensions were not loaded. Included tests use the normal package import for upstream CI. No claim about production deployments.

AI-assisted investigation and patch; one numerical worker.

## Pinned source and reproduction

- Upstream [dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)
- Base commit: `7b1b26ac9760974c12ecefe290a9d3bd37b66b7a`
- Proposed commit: `eea2e33eec41e6117bab4238db6134572e485f6a`
- [Reviewed source](https://github.com/dcajasn/Riskfolio-Lib/blob/7b1b26ac9760974c12ecefe290a9d3bd37b66b7a/riskfolio/src/ParamsEstimation.py)
- [Patch and regression tests](https://github.com/dcajasn/Riskfolio-Lib/pull/267/files)

```text
python run_riskfolio_isolated.py /path/to/Riskfolio-Lib /path/to/archive/test_augmented_historical_prior.py
```

The archived harness compiles the actual unchanged function bodies and their historical-moment helpers from the pinned source. It uses real numpy/pandas/scipy; no numerical stub substitutes their results. This is an isolated source-level run, not a full Riskfolio installation or compiled-extension test. Pass the archived test path as shown. Run before and after applying proposed.patch to the pinned repository; the patch also adds the test for normal upstream pytest execution.

The archive includes the patch, test, test transcripts, upstream license and verification metadata. Run numerical work with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1. The source-level tests do not establish an effect on a real investor or production service.

## Search boundary and license

Public open/closed issue searches, related comments and current open PRs were checked on 22 October 2026. No matching fix was found within that search. This is not an exhaustive priority guarantee.

Report: CC BY 4.0. Original tests and harness: MIT. Upstream code remains under BSD-3-Clause. No private correspondence, credentials or customer data is included.
