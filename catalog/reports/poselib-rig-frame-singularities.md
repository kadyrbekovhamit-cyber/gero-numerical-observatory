# PoseLib: valid camera geometry lost through coordinate singularities

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO).

**Published:** 23 October 2026. **Status:** OPEN proposed fix; not merged. Full-suite/cross-platform validation outstanding.

**Upstream PR:** https://github.com/PoseLib/PoseLib/pull/217

## Scope and source

Repository: https://github.com/PoseLib/PoseLib. Reproduced source baseline: `6a636b1d84273d5e3ba97676463476569611572a`. Proposed patch snapshot: `21682ed9c37f2b75534646cfdac5647a87b70907`. OpenCV wheel reproductions use 4.14.0.94 and, where indicated, 5.0.0.93; these wheels are not the patched current development build.

## Finding, correction and validation

The minimal solvers can lose solutions because of the chosen rig frame even when the geometric problem is well posed. In `gp3p`, the translation-elimination block is singular when rays 0 and 1 have the same x/z ratio. In `gp3p`, `gp4ps_kukelova`, `up2p`, `ugp2p` and `ugp3ps`, a bearing with z=0 makes the two selected cross-product equations dependent.

This patch detects those coordinate singularities and solves in a better-conditioned rig frame, then transforms the returned poses back. The upright solvers rotate only around the gravity/y axis. `gp3p` also selects the least-parallel first pair. The ordinary well-conditioned path is unchanged. This does not claim to solve physically degenerate geometry or the separate Cayley/Weierstrass 180-degree issue discussed in #4 and #7.

Added a regression test registered with the existing test runner. It creates exact observations from known 3D geometry and checks recovery of the known pose across generic, equal-column and z=0 cases. A small C++ executable compiled the actual five solver source files plus their dependencies and this test, before and after the patch:

- 6,000 solver calls across these cases, with 500 scenes per tested solver/configuration;
- original: 3,000 failures (500 equal-column gp3p and 2,500 z=0 failures);
- patched: 0 failures at the test's 1e-5 pose-error criterion;
- the 2,500 generic control calls passed before and after.

The full PoseLib test suite, performance benchmarking and cross-platform CI have not been run locally. Tested with Eigen 3.4.0 on macOS arm64, one compiler process at a time. No real device, production deployment or downstream safety impact is claimed.

Prior work: #7 already discusses coordinate rotation as a remedy for another parameterization singularity; this is not a claim to invent that technique. The specific selected-row and elimination-block failures above remain in the checked master commit 6a636b1.

Prepared by Xamit Kadirbekov and Daniyal Kadirbekov (GERO), with AI assistance in analysis and patch/test preparation.

### Upstream CI update — 23 October 2026

After the formatting-only commit `21682ed9c37f2b75534646cfdac5647a87b70907`, [upstream CI](https://github.com/PoseLib/PoseLib/actions/runs/39964249947) passed formatting, compilation and its solver benchmark. The workflow sets `WITH_BENCHMARK=ON` but does not enable `BUILD_TESTS`; this result therefore does not establish a complete unit-suite pass or a run of the new integrated regression test. A controlled before/after performance comparison also remains outstanding.


## Reproduction package

The accompanying ZIP contains the submitted source diff, source-level regression tests within that diff, selected research scripts and exact validation output. Apply `proposed-fix.diff` to the pinned upstream baseline. Follow that project's normal build instructions to run the added regression test. The archived logs identify the limited local execution scope; they are not evidence that the complete upstream suite passed. Script paths in historical build logs may refer to the local research workspace.

The mathematical checks use known synthetic geometry, round trips, or independent high-precision references. The samples intentionally exercise problematic coordinates: failure counts are **not** prevalence estimates for real deployments.

## Priority and impact boundaries

Public GitHub issues, pull requests and nearby changes were reviewed on 23 October 2026. References above identify related work where applicable. No exhaustive literature search or absolute priority claim is made. These are proposed implementation corrections using established numerical techniques, not new PnP theory.

No customer deployment, drone, robot, financial loss or safety incident was tested or established. Affected-library behavior alone does not show that any particular downstream product is affected. This is a numerical correctness report, not a security advisory.

AI assistance was used for analysis, code and writing. The authors prepared and checked this report; this is not an independent external certification. Report text: CC BY 4.0. Original reproduction code: MIT. Upstream code and modified code retain their upstream licenses (BSD-3-Clause).
