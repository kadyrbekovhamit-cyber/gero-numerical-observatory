# OpenCV iterative PnP: solve near the object instead of a distant origin

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO).

**Published:** 23 October 2026. **Status:** DRAFT proposed fix; not merged. Native OpenCV accuracy tests outstanding.

**Upstream PR:** https://github.com/opencv/opencv/pull/30484

## Scope and source

Repository: https://github.com/opencv/opencv. Reproduced source baseline: `4ffd42f66ae8e554819d86647753f2cec6380748`. Proposed patch snapshot: `ec857112e683d48cdba7d57bcb12e2784944bb4f`. OpenCV wheel reproductions use 4.14.0.94 and, where indicated, 5.0.0.93; these wheels are not the patched current development build.

## Finding, correction and validation

With a distant object-frame origin, iterative PnP can stop at an image-inaccurate initial pose even when the same geometry converges after recentering. Large translation parameters interact poorly with the current refinement conditioning/stopping behavior.

This patch centers the object points before `findExtrinsicCameraParams2`, transforms the initial translation consistently when supplied, then restores the returned translation to the caller's frame. It leaves rotation coordinates unchanged and adds a regression with a distant origin and a close initial guess.

The regression geometry was reproduced on the 4.14.0 wheel: the unmodified call fails its accuracy criterion, while the coordinate-transformed wrapper passes. The same behavior was investigated on 5.0.0, but the detailed stopping-rule explanation was inspected only for the 4.x implementation. This PR targets 4.x.

Native OpenCV tests have not been built or run locally. The Python mirror validates the proposed coordinate transformation; actual C++ integration and the module's existing tests still need CI. This does not change the separate, intentional float32 conversion inside classic RANSAC.

Prepared by Xamit Kadirbekov and Daniyal Kadirbekov (GERO), with AI assistance in analysis and patch/test preparation. No production-device incident or downstream loss is claimed.

## Reproduction package

The accompanying ZIP contains the submitted source diff, source-level regression tests within that diff, selected research scripts and exact validation output. Apply `proposed-fix.diff` to the pinned upstream baseline. Follow that project's normal build instructions to run the added regression test. The archived logs identify the limited local execution scope; they are not evidence that the complete upstream suite passed. Script paths in historical build logs may refer to the local research workspace.

The mathematical checks use known synthetic geometry, round trips, or independent high-precision references. The samples intentionally exercise problematic coordinates: failure counts are **not** prevalence estimates for real deployments.

## Priority and impact boundaries

Public GitHub issues, pull requests and nearby changes were reviewed on 23 October 2026. References above identify related work where applicable. No exhaustive literature search or absolute priority claim is made. These are proposed implementation corrections using established numerical techniques, not new PnP theory.

No customer deployment, drone, robot, financial loss or safety incident was tested or established. Affected-library behavior alone does not show that any particular downstream product is affected. This is a numerical correctness report, not a security advisory.

AI assistance was used for analysis, code and writing. The authors prepared and checked this report; this is not an independent external certification. Report text: CC BY 4.0. Original reproduction code: MIT. Upstream code and modified code retain their upstream licenses (Apache-2.0).
