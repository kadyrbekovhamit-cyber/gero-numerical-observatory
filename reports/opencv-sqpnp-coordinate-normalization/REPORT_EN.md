# OpenCV SQPnP: conditioning across units and coordinate origins

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO).

**Published:** 23 October 2026. **Status:** DRAFT proposed fix; not merged. Native OpenCV accuracy tests outstanding.

**Upstream PR:** https://github.com/opencv/opencv/pull/30483

## Scope and source

Repository: https://github.com/opencv/opencv. Reproduced source baseline: `4ffd42f66ae8e554819d86647753f2cec6380748`. Proposed patch snapshot: `ad0bd0f0a0ec3c6733f9592590f0a9b03144f046`. OpenCV wheel reproductions use 4.14.0.94 and, where indicated, 5.0.0.93; these wheels are not the patched current development build.

## Finding, correction and validation

SQPnP builds its normal matrix from uncentered squared object coordinates and uses absolute rank thresholds. Consequently changing length units can cause an assertion or a wrong pose even though the image observations are unchanged. Large world-coordinate offsets can also lose precision through cancellation.

This patch solves in a centered, unit-RMS object frame and maps translations back with `t = scale * t_normalized - R * center`. The underlying SQPnP solver is unchanged. The regression tests cover changes of length unit and a distant world origin.

The released 4.14.0 wheel failed the same regression geometry, while the exact coordinate-transform wrapper passed. The standalone research witness included a small-scale scene returning success with an approximately 135-degree rotation error; the threshold depends on geometry and point count and is not a universal 0.5 mm limit. Native OpenCV tests have not been built or run locally; the wrapper validates the mathematical transformation, not every aspect of compiled integration. The test and change are submitted as draft pending native CI.

The proposal is numerical conditioning of an existing algorithm, not a claim of a new SQPnP algorithm. It does not repair degenerate input geometry.

Prepared by Xamit Kadirbekov and Daniyal Kadirbekov (GERO), with AI assistance in analysis and patch/test preparation. No production-device incident or downstream loss is claimed.

## Reproduction package

The accompanying ZIP contains the submitted source diff, source-level regression tests within that diff, selected research scripts and exact validation output. Apply `proposed-fix.diff` to the pinned upstream baseline. Follow that project's normal build instructions to run the added regression test. The archived logs identify the limited local execution scope; they are not evidence that the complete upstream suite passed. Script paths in historical build logs may refer to the local research workspace.

The mathematical checks use known synthetic geometry, round trips, or independent high-precision references. The samples intentionally exercise problematic coordinates: failure counts are **not** prevalence estimates for real deployments.

## Priority and impact boundaries

Public GitHub issues, pull requests and nearby changes were reviewed on 23 October 2026. References above identify related work where applicable. No exhaustive literature search or absolute priority claim is made. These are proposed implementation corrections using established numerical techniques, not new PnP theory.

No customer deployment, drone, robot, financial loss or safety incident was tested or established. Affected-library behavior alone does not show that any particular downstream product is affected. This is a numerical correctness report, not a security advisory.

AI assistance was used for analysis, code and writing. The authors prepared and checked this report; this is not an independent external certification. Report text: CC BY 4.0. Original reproduction code: MIT. Upstream code and modified code retain their upstream licenses (Apache-2.0).
