# Luxon: skipped midnights and clamped month-end differences

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 25 October 2026.

**Status:** Open upstream PRs; not merged or released. Scoped checks and limitations documented.

Two independent proposals correct calendar counts and fractional month/quarter anchors.

## Source and proposed changes

Base: `f427515a38f6a671f8de663e6bcc040ed81f114e` in moment/luxon.

- [luxon-count PR #1830](https://github.com/moment/luxon/pull/1830) — proposed commit `7488b95fcca044f43bfddbe90a0f55918144c2b0`.
- [luxon-diff PR #1831](https://github.com/moment/luxon/pull/1831) — proposed commit `a56621f33284e90b7acdf31683133ac0a48e5a8a`.

## Count

Interval.count('days') returns 3 for an interval touching April 25–28, 2025 in Africa/Cairo. On the DST-start date, startOf('day') normalizes nonexistent midnight to 01:00. The boundary diff is then slightly below an integer and Math.floor loses a calendar date.

Round the diff for calendar units after both endpoints have been normalized to unit boundaries; retain flooring for elapsed-time units. The half-open end rule is unchanged. Seven regressions cover five midnight-DST zones, a Monday transition for weeks, and an exclusive end at a skipped-midnight boundary. Six fail on the base. This branch's full result is 1220 passed / the same 9 baseline failures.

Related open #1798 concerns endOf's next-unit boundary. This patch concerns Interval.count and does not change endOf.


Validation: the new regressions fail on upstream main f427515a38f6 (before the source change) and pass after it. Each PR was tested independently against that base with Jest --runInBand --no-cache under America/New_York. Nine pre-existing locale-formatting assertions also fail on the unchanged base in this Node/ICU environment; their exact failure names are unchanged after this patch. No full-suite pass is claimed. Changed-file Prettier and git diff --check pass.

## Diff

DateTime.diff(..., 'months') can decrease as the end instant moves forward after a clamped month end: starting at 2019-01-31T10:00Z, the result for Feb 28 at 23:00 exceeds the result for Mar 1 at 00:00. The fractional denominator uses cursor.plus(1 month), reaching Mar 28 instead of the original anchor plus two months (Mar 31).

Derive the next fractional anchor from the original earlier DateTime and accumulated units, as the overshoot branch already does. Two regressions cover monotonicity and independently counted month/quarter fractions (31 days and 91 days). Both fail on the base. This branch's full result is 1215 passed / the same 9 baseline failures.

This follows the original-anchor principle from #1340 / #1301 and does not claim month arithmetic is additive. The earlier fix changed the integer search; this patch fixes the remaining fractional denominator. #537 discusses the calendar semantics.


Validation: the new regressions fail on upstream main f427515a38f6 (before the source change) and pass after it. Each PR was tested independently against that base with Jest --runInBand --no-cache under America/New_York. Nine pre-existing locale-formatting assertions also fail on the unchanged base in this Node/ICU environment; their exact failure names are unchanged after this patch. No full-suite pass is claimed. Changed-file Prettier and git diff --check pass.

The Linux Foundation CLA check is pending; submission is not acceptance. The nine remaining locale-formatting failures were independently reproduced on the unchanged base and retain exactly the same test names.

## Reproduction and boundaries

The archive contains exact public commit patches, upstream licenses, and selected validation metadata. Check out the pinned base, apply each proposal independently, install the project’s declared test dependencies, and run the affected tests described above. The metadata records the actual narrower local environment; a full clean build or unlisted tests are not implied. Existing local paths in logs are descriptive, not required paths.

Novelty checking covered related open/closed GitHub issues and PRs and relevant source history on 25 October 2026. Related reports are acknowledged above; an exhaustive search of every community channel or first-discovery priority is not claimed. No real customer records were used, no deployed product was tested, and no monetary or clinical loss was measured.

## Attribution and licensing

Claude assisted the initial investigation and patch preparation. OpenAI Codex independently repeated the scoped before/after checks, checked the stated reference calculations, and helped prepare the proposed changes and this report. Research/report authors are Xamit Kadirbekov and Daniyal Kadirbekov. Independent human code or legal review is not claimed. Upstream projects retain credit for their software. Report: CC BY 4.0; original reproduction code: MIT; patches retain the upstream licenses.
