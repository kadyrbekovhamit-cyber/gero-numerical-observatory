# OpenCV RANSAC: preserve the camera-pose initial guess

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO).

**Published:** 23 October 2026. **Status:** DRAFT proposed fix; not merged. Native OpenCV accuracy tests outstanding.

**Upstream PR:** https://github.com/opencv/opencv/pull/30481

## Scope and source

Repository: https://github.com/opencv/opencv. Reproduced source baseline: `4ffd42f66ae8e554819d86647753f2cec6380748`. Proposed patch snapshot: `77daf924eb1e389e4feb2f8781bdc84b6ab5ccd0`. OpenCV wheel reproductions use 4.14.0.94 and, where indicated, 5.0.0.93; these wheels are not the patched current development build.

## Finding, correction and validation

When `useExtrinsicGuess=true`, the legacy `solvePnPRansac` path calls `_rvec.create(3, 1, CV_64F)` and `_tvec.create(...)` before reading the supplied guess. A float32 or 1×3 input is therefore reallocated and its initial values are lost. The final iterative solve can start from uninitialized values while RANSAC still reports success.

This change copies and converts valid supplied guesses before allocating outputs. It preserves the existing empty-guess fallback, including `Calib3d_SolvePnPRansac.double_support`. The regression test covers float32/float64 and row/column layouts, both a successful fit and the no-consensus path that exposes the retained initial guess.

Reproduced locally with opencv-python 4.14.0.94 and 5.0.0.93. For example, a requested rotation guess `(0.11, 0.22, 0.33)` returned as `(1.5, 2.5, 3.5)` after failure with a float32 input, while a float64 column preserved it. Values in invalidated buffers are allocation-dependent, so an exact erroneous pose is not a stable expectation.

Validation: the released-wheel reproduction and a Python mirror of the proposed regression cases were run locally. The native OpenCV test target has **not** been built or run locally; native CI validation remains outstanding. Related #12334 and closed/unmerged #28129 concern binding/input-output handling, but their proposed ordering still reads the guess after allocation. This PR concerns that remaining input-value loss. #22914 fixed overwriting the pose during RANSAC iterations; this is a separate reallocation-before-read problem. #8782 discusses which RANSAC stages use the guess.

Prepared by Xamit Kadirbekov and Daniyal Kadirbekov (GERO), with AI assistance in analysis and patch/test preparation.

## Reproduction package

The accompanying ZIP contains the submitted source diff, source-level regression tests within that diff, selected research scripts and exact validation output. Apply `proposed-fix.diff` to the pinned upstream baseline. Follow that project's normal build instructions to run the added regression test. The archived logs identify the limited local execution scope; they are not evidence that the complete upstream suite passed. Script paths in historical build logs may refer to the local research workspace.

The mathematical checks use known synthetic geometry, round trips, or independent high-precision references. The samples intentionally exercise problematic coordinates: failure counts are **not** prevalence estimates for real deployments.

## Priority and impact boundaries

Public GitHub issues, pull requests and nearby changes were reviewed on 23 October 2026. References above identify related work where applicable. No exhaustive literature search or absolute priority claim is made. These are proposed implementation corrections using established numerical techniques, not new PnP theory.

No customer deployment, drone, robot, financial loss or safety incident was tested or established. Affected-library behavior alone does not show that any particular downstream product is affected. This is a numerical correctness report, not a security advisory.

AI assistance was used for analysis, code and writing. The authors prepared and checked this report; this is not an independent external certification. Report text: CC BY 4.0. Original reproduction code: MIT. Upstream code and modified code retain their upstream licenses (Apache-2.0).
