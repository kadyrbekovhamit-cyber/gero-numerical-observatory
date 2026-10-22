# Curo: fortnightly payments collapse onto monthly dates

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026. Investigation and writing were AI-assisted.

## Two independently checkable problems

In `andrewmurphy353/curo_python` revision `207715a730bb069784ceae7db75d8b0a97e9322b`, a fortnightly series beginning 22 January 2026 generates:

- Actual: 22 Jan, 22 Feb, 22 Feb, 22 Mar, 22 Mar, 22 Apr.
- Expected: 22 Jan, 5 Feb, 19 Feb, 5 Mar, 19 Mar, 2 Apr.

The source initially creates two-week intervals, then resets every date to the original day of the month. Weekly schedules are affected by the same adjustment. Repeated dates are an observable calendar error before solving any interest rate.

A separate error appears in `USAppendixJ.compute_factor`. For a fortnightly unit-period from 1 January to 2 February, 32 elapsed days equal **2 + 4/14** unit-periods. The code counts two complete 14-day periods but divides the four-day remainder by **15**. A semimonth and a fortnight are different conventions. [CFPB Appendix J, paragraph (b)(5)(iv)](https://www.consumerfinance.gov/rules-policy/regulations/1026/j/) defines a weekly multiple using the actual number of days per unit-period; the semimonth rule is separately stated in (iii). This is a check of the library's claimed convention, not a legal opinion about a lender.

## Fresh reproduction and a narrow candidate

On 22 October the source was downloaded at the revision above. Actual `SeriesPayment.to_cash_flows` and `USAppendixJ.compute_factor` were imported and run with pandas/SciPy. Only package-version metadata was substituted because the source checkout was not installed; no numerical or calendar methods were replaced.

Eleven independent checks cover weekly and fortnightly schedules starting on an ordinary date, a month-end and a leap day; three two-week day-count cases; and monthly/quarterly controls. **8 fail and 3 pass before the candidate; all 11 pass after.** Calendar expectations use Python standard-library date arithmetic; fractional periods use integer division and remainder by 14.

The candidate skips monthly day replacement for weekly/fortnightly series, on both posting and value dates, and changes the fortnight remainder denominator to 14. The executed schedule assertions cover posting dates; a complete value-date and solver regression suite remains necessary before upstream adoption. The two monthly/quarterly controls are unchanged. No full upstream suite or actual customer contract was tested.

## Disclosure and boundaries

[Issue 2](https://github.com/andrewmurphy353/curo_python/issues/2) was opened 5 October. On 22 October it remains open with no comments, and the repository head still matches the tested revision. The announced wait ended 20 October. No maintainer acceptance, released fix, measured APR error for a live loan, or customer loss is claimed.

The archive contains the independent harness, focused candidate patch and fresh before/after results. Upstream source and reproduction code are MIT licensed; report CC BY 4.0. An earlier report used the Calculator entry point; today's fresh verification is explicitly limited to the two methods above.
