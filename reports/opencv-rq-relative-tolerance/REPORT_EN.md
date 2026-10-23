# OpenCV RQ decomposition: a scale-aware zero check

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO).

**Published:** 23 October 2026. **Status:** DRAFT proposed fix; not merged. Native OpenCV accuracy tests outstanding.

**Upstream PR:** https://github.com/opencv/opencv/pull/30485

## Scope and source

Repository: https://github.com/opencv/opencv. Reproduced source baseline: `4ffd42f66ae8e554819d86647753f2cec6380748`. Proposed patch snapshot: `2d6aa292c57eeca4d7cfcea11920e4712e245aa6`. OpenCV wheel reproductions use 4.14.0.94 and, where indicated, 5.0.0.93; these wheels are not the patched current development build.

## Finding, correction and validation

`RQDecomp3x3` asserts that eliminated entries are below an absolute `FLT_EPSILON`. For otherwise valid matrices with large magnitude, ordinary floating-point residuals can exceed this absolute threshold and trigger an exception. This also affects `decomposeProjectionMatrix`, even though multiplying a projective camera matrix by a nonzero scalar does not change the camera.

The patch scales the three zero checks by `max(1, ||M||_inf)` without changing the decomposition arithmetic. The regression covers factors 1e7, 1e9 and 1e12. The original small-magnitude thresholds are retained.

Validation: compiled the original and patched decomposition bodies in a small C++ harness. On 667 inputs, the original matched the 4.14.0 wheel, including 161 exceptions. The patched body produced no exceptions and matched all 506 previously successful outputs bit-for-bit. The maximum checked relative reconstruction/orthogonality/determinant residual was 1.22e-15. Existing degenerate-case fixtures were included. This was extracted-body validation, not the complete OpenCV module test suite; native CI remains outstanding.

Related #23733/#25182 concern small-scale matrices; #24330/#25050 concern degenerate inputs. This proposal addresses large-magnitude roundoff in the assertion checks, not those earlier cases.

Prepared by Xamit Kadirbekov and Daniyal Kadirbekov (GERO), with AI assistance in analysis and patch/test preparation. No production-device incident or downstream loss is claimed.

## Reproduction package

The accompanying ZIP contains the submitted source diff, source-level regression tests within that diff, selected research scripts and exact validation output. Apply `proposed-fix.diff` to the pinned upstream baseline. Follow that project's normal build instructions to run the added regression test. The archived logs identify the limited local execution scope; they are not evidence that the complete upstream suite passed. Script paths in historical build logs may refer to the local research workspace.

The mathematical checks use known synthetic geometry, round trips, or independent high-precision references. The samples intentionally exercise problematic coordinates: failure counts are **not** prevalence estimates for real deployments.

## Priority and impact boundaries

Public GitHub issues, pull requests and nearby changes were reviewed on 23 October 2026. References above identify related work where applicable. No exhaustive literature search or absolute priority claim is made. These are proposed implementation corrections using established numerical techniques, not new PnP theory.

No customer deployment, drone, robot, financial loss or safety incident was tested or established. Affected-library behavior alone does not show that any particular downstream product is affected. This is a numerical correctness report, not a security advisory.

AI assistance was used for analysis, code and writing. The authors prepared and checked this report; this is not an independent external certification. Report text: CC BY 4.0. Original reproduction code: MIT. Upstream code and modified code retain their upstream licenses (Apache-2.0).
