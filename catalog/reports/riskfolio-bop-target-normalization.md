# Riskfolio BOP: correcting the shrinkage target denominator

22 October 2026 · GERO Numerical Assurance · Xamit Kadirbekov

**REPRODUCED IN THE STATED SCOPE; FIX PROPOSED.** The upstream PR is open. Acceptance and a corrected release are not claimed.

The target coefficient differs from the cited equation. An exact rational regression passes after the one-line correction; full-package integration remains untested locally.

Upstream submission: [https://github.com/dcajasn/Riskfolio-Lib/pull/265](https://github.com/dcajasn/Riskfolio-Lib/pull/265)

`mean_vector(method="BOP")` divides the target coefficient by `mu.T @ Sigma_inv @ mu`. Equation (7) in Bodnar, Okhrin and Parolya instead divides by the target's quadratic form, `b.T @ Sigma_inv @ b`: https://arxiv.org/html/1610.09292v3#S3.E7 . This patch changes that denominator only.

The distinction is observable without a portfolio optimizer. Scaling a nonzero target by `c` must divide its coefficient by `c`, leaving the estimated mean unchanged. With the existing denominator the target contribution scales by `c**2`. The available targets `b1`, `b2`, and `b3` are all constant vectors, so they provide a direct regression case when their values are nonzero.

The test uses 20 observations, a sample mean `(1, 2, 3, 4, 5)/100`, and sample covariance `I/475`. Keeping the library's existing sample-covariance normalization, equation (6) gives `alpha = 17/57`; equations (7)-(8) then give the exact mean `(137, 154, 171, 188, 205)/5700`. The expected values are fixed rational numbers rather than a duplicate of the implementation. All three targets must produce that vector.

Validation at master `7b1b26ac9760974c12ecefe290a9d3bd37b66b7a`:

- Actual `mean_vector` function body executed in an isolated NumPy/pandas harness: **3 failures before; 3 passes after**.
- The submitted test uses the normal package import and belongs in the upstream suite.
- Full-package integration was **not run locally**: the lightweight environment cannot import the package because optional-to-this-function dependencies are missing (first failure: scikit-learn). The local harness extracts and executes the unchanged function AST; it does not replace its numerical operations.
- `git diff --check` passes.

This does not change covariance normalization or address singular/zero-target cases. Prepared with AI assistance; upstream integration review is requested.


## Scope and attribution

This is an AI-assisted numerical correctness investigation using synthetic inputs and, where stated, the upstream test fixtures. There is no evidence here of executed trades, client losses, a security exploit, or effects in a commercial deployment. Independent human review is pending. This report is CC BY 4.0; upstream code retains its original license.
