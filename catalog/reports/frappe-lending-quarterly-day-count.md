# Frappe Lending quarterly schedules used three days after the first repayment

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 7 October 2026

AI-assisted numerical investigation and report. Synthetic software examples;
no production accounts, customer losses or deployment frequency were measured.

## Finding

Frappe Lending `v16.6.0` and source commit
`605ae2a866aade812a4a147938b8a4f3b7aaa760` used the repayment frequency value
`3` as an interest day count after the first quarterly repayment. The first
period used its actual calendar length; later 90–92 day intervals used only
three days with a 365-day denominator.

A synthetic local Frappe site used a principal of 1,000,000, an annual rate of
12%, eight repayments, posting/disbursement on 1 January 2026, and a first
repayment on 1 April 2026. The relevant rows were:

| Payment date | Actual interval | Stored day count | Interest produced | Same opening balance with actual days / 365 |
|---|---:|---:|---:|---:|
| 2026-04-01 | 90 | 90 | 29,589.04 | 29,589.04 |
| 2026-07-01 | 91 | **3** | **874.98** | **26,541.05** |
| 2026-10-01 | 92 | **3** | **735.34** | **22,550.33** |

The remaining quarterly rows likewise stored `3`, while their calendar gaps
were 90–92 days. The site's eight rows totalled 32,776.59 of interest. Applying
the same day-based expression to the actual calendar gaps and the balances
already produced by the site gives 126,639.60. This comparison isolates the
day-count effect; it is not a reconstructed corrected schedule, because fixing
one row also changes later balances and payments.

An independent extracted-method replay reproduces the same unit mismatch. On an
isolated balance of 1,000,000, the unchanged `get_amounts` method returns 986.30
for 3 days; the 91-day `/365` calculation is 29,917.81.

## Cause

`get_non_monthly_days()` returned `3` for `Quarterly`.
`get_days_and_months()` then passed that value to `get_amounts()` as a number of
days while retaining the 365-day denominator. The value represented months in
one part of the calculation and days in another.

## Reproduction boundary

The [portable evidence](../../reports/frappe-lending-quarterly-day-count/)
executes the unchanged upstream methods from pinned source copies with minimal
framework doubles. It does not start a Frappe site or database. A separate
saved synthetic site record used Lending 16.6.0, Frappe 16.36.1 and ERPNext
16.37.0 and exercised the disbursement entry point while preserving the
quarterly frequency.

The bounded pre-report review covered the relevant source history and public
items #2, #100, #217 and #221. No applicable earlier fix was identified in that
scope. This is not a worldwide-priority claim and does not exclude private
reports.

## Upstream correction status

Frappe contributor Nihantra Patel implemented a calendar-day correction in
[PR #1483](https://github.com/frappe/lending/pull/1483). It carries the previous
payment date through schedule construction and uses the actual date difference.
The regression starts on a month end, checks consecutive calendar gaps and
checks a zero final balance.

- PR #1483 merged into `develop` on 5 October 2026 as
  `caaecb4482fdc60a035d83a3ef4b390f225b67d3`.
- Automatic backport [PR #1484](https://github.com/frappe/lending/pull/1484)
  merged into `version-16-hotfix` as
  `f340425d5bc73d2a0816c522a32fad0ee4171806`.
- The latest tagged release checked on 7 October 2026 was still
  [`v16.6.0`](https://github.com/frappe/lending/releases/tag/v16.6.0), published
  before those merges. A fixed official release was therefore not confirmed.

The pull request describes a user report with the same example numbers but does
not publicly identify its source. This report does not claim public attribution
for the upstream change.

## Evidence and limits

The evidence package contains the pinned source files and licence, the portable
replay and its saved result, the synthetic full-site record, the bounded source
review, and saved public metadata/diffs for PRs #1483 and #1484. No email
contents or private correspondence are included.

This is one day-count defect in a specific schedule path. It is not a finding
about Frappe Lending as a whole. No production loan, accounting posting,
customer outcome, financial loss, full upstream test suite, or fixed release
was measured here.

Original report: CC BY 4.0. Original GERO replay: MIT. Vendored Frappe Lending
source retains GPL-3.0.
