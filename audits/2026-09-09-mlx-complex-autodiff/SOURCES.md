# Evidence and prior-art ledger

| Source | Classification | Supported proposition |
|---|---|---|
| [Pinned primitives.cpp](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp) | Primary code | Seven VJPs delegate without conjugation; complex arccosh JVP uses rsqrt(z²−1) |
| `probe.py`, `probe-results.json` | Native official-wheel execution | Original complex errors and the real cosh loss counterexample |
| `native_regression.cpp`, `run-before.log`, `run-after.log` | Actual native C++ execution | Original 85/131 failing scenarios; patched 131/131 pass, under the recorded partial-build conditions |
| Explicit analytic formulas and finite differences | Independently specified mathematical checks | JVP/VJP adjoint relation; principal-branch arccosh sign; derivative of cosh |
| [Earlier issue #3765](https://github.com/ml-explore/mlx/issues/3765) and [PR #3766](https://github.com/ml-explore/mlx/pull/3766) | Prior public work by obchain | Same missing-conjugate class for square, sin, sinh, cosh, tan, tanh, log1p |
| [Commit af554062e789](https://github.com/ml-explore/mlx/commit/af554062e789ccf8d4913a0cb91384515d2a9f2f) | Recorded main history | Earlier complex-unary fix is represented in the inspected source history |
| [PR #3605](https://github.com/ml-explore/mlx/pull/3605) | Prior work | Complex exp/log conjugation; those operations are controls here |
| [PR #2178](https://github.com/ml-explore/mlx/pull/2178) | Prior work | Earlier complex multiplication, division and FFT corrections; coverage was not exhaustive |

## Duplicate search boundary

The original review used `repo:ml-explore/mlx` queries for complex/VJP,
cos/gradient, arccosh, conjugate/unary and arcsin/gradient and inspected the
last 50 primitives.cpp commits. It also reviewed #4227 and #4260, which
concern other derivative tests and complex variance respectively.

No exact report of this full seven-function selection or the arccosh sign
example was identified in that review. Search completeness and first
discovery are not established. A relevant prior PR was found via commit
history even when text search missed its word form, illustrating the limit.

On 2026-09-09, web retrieval of #3765/#3766 returned cached pages from two
months earlier, while `gh api repos/ml-explore/mlx/pulls/3766` returned 404.
The cached body still documents the earlier class and operation list;
its displayed Open status is not treated as current. The inspected main
history includes af554062e789 and matching changes. No claim is made to
have read all current comments or received maintainer confirmation.

No source above establishes effects on a trained model or deployed product.
