# statsmodels: continuity correction and power inversion regressions

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 24 October 2026.

**Status:** Open upstream PRs; not merged or released. Local regression checks and limitations documented.

Two proposed corrections: avoid overshooting a continuity adjustment and reject false numerical roots.

## Published proposals

- [statsmodels PR #10551](https://github.com/statsmodels/statsmodels/pull/10551) — commit `1f3fb5a711293de4fd76600f41a3e92c7ca781a0`, OPEN at submission.
- [statsmodels PR #10552](https://github.com/statsmodels/statsmodels/pull/10552) — commit `fbeb847c68b767cee7fb89d8240447fd9788afaa`, OPEN at submission.

## Continuity correction

In StratifiedTable.test_null_odds(correction=True), subtracting one half unconditionally can increase the magnitude of a deviation that is already smaller than one half. The proposed correction applies the half-unit adjustment only when the absolute deviation reaches one half, matching R's mantelhaen.test implementation.

The supplied sparse synthetic example changes the asymptotic p-value from approximately 8.45e-7 to approximately 0.92. This is a numerical/rule discrepancy; the larger p-value does not establish that no association exists, and it does not establish that an asymptotic test is appropriate for such sparse observations. No clinical study or patient outcome was evaluated.

## Power and inversion

A one-sided TTestIndPower.solve_power request with nobs1=1237, alpha=0.05, power=0.8 and alternative='smaller' can return about +0.458 instead of about -0.100. Evaluating the requested power at that candidate does not recover 0.8. TTestPower.power(0.3, 1000, 0.05) can also return NaN in the checked SciPy environment.

The proposal uses noncentral-t symmetry only when CDFLIB returns NaN, and checks the equation residual before accepting a Brent result. A converged solver flag at a discontinuity is not proof that the requested equation was solved. New tests cover mirrored one-sided cases, the NaN discontinuity, and ordinary scalar/list inputs.

## Validation and limitations

Base: statsmodels main 4086205c69cc8224bf09d7f0fa1d66e90db20fce. Python 3.12, SciPy 1.18.1. Current pure-Python modules and test modules were overlaid on an existing statsmodels 0.15.0 binary installation; this was not a fresh Cython build.

The contingency-table module has 91 passing tests. The power module has 118 passing, two skipped and two expected failures. New regressions fail on the original implementation. Ruff passes, and coverage confirms every added executable source line is exercised. Full package testing was not performed. The existing warning test's xfail marker was removed because it now passes; no unrelated failure was hidden.

The human submitter reviewed/confirmed understanding of the final diffs before submission. The PRs disclose AI assistance. Related solver history includes #9884 and #9895; this report does not claim to originate the general noncentral-t numerical problem.

## References and reproduction

- [R mantelhaen.test source](https://github.com/wch/r-source/blob/trunk/src/library/stats/R/mantelhaen.test.R).
- Exact source and regression changes are in the PR commits and the accompanying archive. Check out the base, apply one patch, then run its affected test module using the project's test environment. The archive records the narrower local environment used here.

No actual trial plan, treatment decision or financial loss is inferred from these synthetic cases. Upstream review and independent integration testing remain necessary.

## Attribution and license

Investigation, proposed patches, tests and report preparation used Codex/ChatGPT assistance. Xamit Kadirbekov and Daniyal Kadirbekov are the report authors; upstream software remains credited to its developers. Report: CC BY 4.0. Original reproduction code: MIT; upstream patches retain their upstream licenses.
