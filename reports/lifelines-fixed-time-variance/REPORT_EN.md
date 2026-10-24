# lifelines: keep fixed-time variance constant between observed events

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 24 October 2026.

**Status:** Open upstream PRs; not merged or released. Local regression checks and limitations documented.

A hand-calculated Greenwood regression exposes interpolation inconsistent with stepwise survival curves.

## Published proposals

- [CamDavidsonPilon PR #1709](https://github.com/CamDavidsonPilon/lifelines/pull/1709) — commit `3c59fcf90d38180e4f9d7ede1c53284b6d30cba1`, OPEN at submission.

## The discrepancy

The fixed-time survival comparison evaluates fitted Kaplan–Meier survival as a step function but interpolates the Greenwood cumulative sum linearly. Between observed times, that interpolation can change the statistic even though neither fitted survival curve changes.

The proposed helper selects the latest observed cumulative-variance value at or before the requested time, using a right-continuous step convention for nonparametric fits. The parametric path is unchanged. The regression computes the survival probabilities and Greenwood terms directly and checks times 5.0, 6.0 and 6.4 inside one unchanged interval.

## Validation

Base: lifelines master 7a8fc34a013ecd79fa405017b89e1697c1cc6e17; Python 3.12 and pandas 2.3.3. The new test fails on the original implementation and passes after the patch. The affected test_statistics.py module changes from 41 passed / two failed to 42 passed / one failed. The remaining test_StatisticalResult_can_be_printed failure is an existing presentation failure present before and after; it was not modified or hidden.

A full project test-suite pass is not claimed. Reports #1446, #1499 and #1543 concern documentation; the current targeted GitHub search did not find the same interpolation fix, but this is not an exhaustive novelty claim.

The archive contains the exact patch with hand-calculated regression, license and before/after validation metadata. Check out the base, apply the patch, install project dependencies and run lifelines/tests/test_statistics.py to reproduce in a compatible environment.

This test does not establish a measured percentage of missed clinical findings or errors in real studies. No patient data were used. The proposal is open for maintainer review, not an accepted release correction.

## Attribution and license

Investigation, proposed patches, tests and report preparation used Codex/ChatGPT assistance. Xamit Kadirbekov and Daniyal Kadirbekov are the report authors; upstream software remains credited to its developers. Report: CC BY 4.0. Original reproduction code: MIT; upstream patches retain their upstream licenses.
