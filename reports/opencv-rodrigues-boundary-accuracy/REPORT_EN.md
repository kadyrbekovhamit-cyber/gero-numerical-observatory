# OpenCV Rodrigues: accuracy at small angles and near pi

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO).

**Published:** 23 October 2026. **Status:** DRAFT proposed fix; not merged. Native OpenCV accuracy tests outstanding.

**Upstream PR:** https://github.com/opencv/opencv/pull/30482

## Scope and source

Repository: https://github.com/opencv/opencv. Reproduced source baseline: `4ffd42f66ae8e554819d86647753f2cec6380748`. Proposed patch snapshot: `bf5ab2f63a43323d4e1a0bfcdb3500167f85db04`. OpenCV wheel reproductions use 4.14.0.94 and, where indicated, 5.0.0.93; these wheels are not the patched current development build.

## Finding, correction and validation

For matrix-to-vector `Rodrigues`, the `sin(theta) < 1e-5` branch maps every sufficiently small rotation to exactly zero. Near pi, its axis recovery can introduce components or choose the wrong sign for angles strictly below pi. For example, `(9e-6, 0, 0)` converted to a rotation matrix and back becomes zero on opencv-python 4.14.0.94.

The patch uses atan2 for the angle, a small-angle expansion near zero and the dominant column of the symmetric part near pi, with sign from the skew part. It also supplies the inverse-log Jacobian in the boundary branch. The regression checks round trips and tangent-direction derivatives. The sign ambiguity at exactly pi is not treated as a defect.

Validation on macOS arm64: compiled the changed C++ boundary block verbatim against OpenCV headers, with a 50-digit independent exponential-map reference. All 207 boundary matrices entered the compiled branch; maximum rotation error was 9.14e-16 rad. The proposed regression logic had 42 failed checks on the released implementation and none with the patched C++ block. A separate 3,000-case ordinary-angle comparison was also performed in the Python port. This is fragment-level validation, not a build of the complete patched library. Native accuracy tests remain outstanding.

Related #30402/#30403/#30424 repair IPPE's own rotation conversion; this patch addresses the shared Rodrigues inverse. #8813 concerns planar pose ambiguity rather than this small-angle round-trip witness.

Prepared by Xamit Kadirbekov and Daniyal Kadirbekov (GERO), with AI assistance in analysis and patch/test preparation. No production-device incident or downstream loss is claimed.

## Reproduction package

The accompanying ZIP contains the submitted source diff, source-level regression tests within that diff, selected research scripts and exact validation output. Apply `proposed-fix.diff` to the pinned upstream baseline. Follow that project's normal build instructions to run the added regression test. The archived logs identify the limited local execution scope; they are not evidence that the complete upstream suite passed. Script paths in historical build logs may refer to the local research workspace.

The mathematical checks use known synthetic geometry, round trips, or independent high-precision references. The samples intentionally exercise problematic coordinates: failure counts are **not** prevalence estimates for real deployments.

## Priority and impact boundaries

Public GitHub issues, pull requests and nearby changes were reviewed on 23 October 2026. References above identify related work where applicable. No exhaustive literature search or absolute priority claim is made. These are proposed implementation corrections using established numerical techniques, not new PnP theory.

No customer deployment, drone, robot, financial loss or safety incident was tested or established. Affected-library behavior alone does not show that any particular downstream product is affected. This is a numerical correctness report, not a security advisory.

AI assistance was used for analysis, code and writing. The authors prepared and checked this report; this is not an independent external certification. Report text: CC BY 4.0. Original reproduction code: MIT. Upstream code and modified code retain their upstream licenses (Apache-2.0).
