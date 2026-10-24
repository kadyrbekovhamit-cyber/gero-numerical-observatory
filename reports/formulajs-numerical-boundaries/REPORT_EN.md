# Formula.js: seven proposed fixes for blanks, financial formulas and dates

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 24 October 2026.

**Status:** Open upstream PRs; not merged or released. Local regression checks and limitations documented.

Reproduced blank-cell, solver-convergence and calendar discrepancies, with independent PRs and tests.

## Published proposals

- [formulajs PR #258](https://github.com/formulajs/formulajs/pull/258) — commit `f7e850915c2e3b51ff4e4100bfdfd64e8a7503be`, OPEN at submission.
- [formulajs PR #259](https://github.com/formulajs/formulajs/pull/259) — commit `cfe160de0b97791bce1b83938da6bb5491112239`, OPEN at submission.
- [formulajs PR #260](https://github.com/formulajs/formulajs/pull/260) — commit `2abdf67ddd1d046385ecf85b9d36f3e38434e540`, OPEN at submission.
- [formulajs PR #261](https://github.com/formulajs/formulajs/pull/261) — commit `011d76480e3138aa2fea4e96219caa2fff294463`, OPEN at submission.
- [formulajs PR #262](https://github.com/formulajs/formulajs/pull/262) — commit `394e0947b699d03cca490dc8db55a888b49ab9b1`, OPEN at submission.
- [formulajs PR #263](https://github.com/formulajs/formulajs/pull/263) — commit `c40ec5f5e382d23eba3b5965da854d052597ae57`, OPEN at submission.
- [formulajs PR #264](https://github.com/formulajs/formulajs/pull/264) — commit `c6fc73239d0bb4f2d5a8ca839e52de1eafc0f887`, OPEN at submission.

## What changes, and why it matters

An empty cell is not always a numeric zero. For example, a referenced blank between 2, 6 and 8 changes a median from 6 to 4 if coerced to zero. The proposed blank-cell change ignores blank/null/undefined elements for relevant numeric and paired-series calculations while preserving explicit numeric zero and propagated errors. A missing value can otherwise alter a statistic or forecast silently. This does not claim that every spreadsheet engine treats every blank argument identically.

RATE can return a finite candidate that does not solve the cash-flow equation. In the supplied weekly-payment example, a bad default iteration implies roughly 164.84% nominal annual rate, versus the generating rate of 6.5%. The patch checks convergence and rejects invalid results with #NUM; it does not guarantee convergence from every starting guess. A suitable guess recovers the original weekly rate. Nominal annualization here means weekly rate multiplied by 52, not an effective annual return or a regulatory APR.

The remaining separate proposals concern DATEDIF month-end boundaries, WEEKNUM year-boundary handling, YEARFRAC calendar-day/DST and basis handling, NETWORKDAYS holiday/weekend behavior, and DISC/PRICEDISC day-count consistency. Each PR has a small regression and its own branch. These are seven change families, not a count of independent mathematical discoveries.

## Validation

The base is formulajs/formulajs master 42dbeb18b15f271b6e90d927cc972f0f06565dcf. Every new regression fails on the base. Each separate patched branch passes its complete existing Mocha test suite in both UTC and America/New_York: 571–573 tests pass, with three pre-existing pending tests. Changed-file Prettier checks pass. The repository's full lint/build pipeline was not run.

The evidence archive contains exact diffs, separate regression files, upstream licenses and validation metadata. To reproduce, check out the base, install the repository's declared dependencies, apply the relevant patch, and run its Mocha suite under each stated TZ. Do not apply unrelated PR patches when validating one proposal.

Reference checks used Microsoft's function documentation and LibreOffice's published day-count implementation. We did not run Microsoft Excel. The proposed US/NASD February changes and a separate DAYS360 allegation were excluded because the compatibility convention was not sufficiently established.

## Sources and boundaries

- [Microsoft MEDIAN](https://support.microsoft.com/en-us/excel/functions/median-function), [CORREL](https://support.microsoft.com/en-us/excel/functions/correl-function), [RATE](https://support.microsoft.com/en-us/excel/functions/rate-function).
- [LibreOffice financial implementation](https://github.com/LibreOffice/core/blob/master/scaddins/source/analysis/financial.cxx).
- PRs above include case-specific examples and related issue history; no claim is made that all community discussions were searched or that these are the first reports of any related concept.

No customer financial records were used, and no real monetary loss was measured. The archive's validation is scoped to these branches and dependency versions. Maintainer acceptance and release inclusion remain pending.

## Attribution and license

Investigation, proposed patches, tests and report preparation used Codex/ChatGPT assistance. Xamit Kadirbekov and Daniyal Kadirbekov are the report authors; upstream software remains credited to its developers. Report: CC BY 4.0. Original reproduction code: MIT; upstream patches retain their upstream licenses.
