# PyLoan replaces an explicit 100 payment with a 919.10 final payment

Xamit Kadirbekov · GERO Research · 18 September 2026

PyLoan 0.7.2 changes a user-specified annuity payment in a small synthetic loan schedule. For principal 1,000, annual interest 12%, two full monthly periods and an explicit payment of 100, the first payment is 100 but the last becomes **919.10**. The output reports no remaining principal. Applying the specified payment instead gives **100 paid and 819.10 remaining** at the second date.

This is a regression of the documented explicit-payment mode. It is not a claim that a bank charged a borrower or that a particular product deploys PyLoan.

## Measured chain: calculation → document → decision

**New evidence addendum, 18 September 2026.** This extends the same reported defect; it is not another independent bug.

We executed the official PyLoan 0.7.2 wheel, pinned current source, local candidate and restored original through a small published GERO adapter. The adapter writes a synthetic payment notice from the final schedule row, reads that serialized document, then checks whether its requested amount fits an example budget of **150.00 currency units**. The notice and budget rule are GERO demonstration code, not built-in PyLoan features or a tested bank integration.

| Measured output | Independent reference / candidate | Official release / current / restored |
|---|---:|---:|
| Requested amount in generated notice | 100.00 | **919.10** |
| Remaining principal shown by schedule | 819.10 | **0.00** |
| Budget check: request ≤ 150.00 | WITHIN_BUDGET | **EXCEEDS_BUDGET** |
| Requested cash above that budget | 0.00 | **769.10** |

The chain is: **the final-payment override → a generated notice requesting 819.10 more than specified → a reversed budget-check result**. The requested amount is **9.191 times** the configured payment. The candidate restores the document fields and decision to the independent reference; restoring the original branch reproduces both differences. The archive contains executable code, all five sets of generated JSON/HTML documents, and checksums.

These are measured consequences inside an explicitly synthetic workflow. The additional request represents accelerated principal repayment in this schedule, not a measured fee, extra interest charge or borrower loss. No payment was collected, no real person was assessed, and no production use or frequency was measured. The example budget is a stated demonstration input, not a credit-eligibility rule. The original 864-case results and remaining rounding/test limitations below remain unchanged.

## Contract and independent calculation

The [pinned quickstart](https://github.com/darius-lesch/pyloan/blob/73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1/docs/docsrc/source/quickstart.rst#L69) permits an explicit payment that does not fully amortize the loan within its term. The test uses the default 30E/360 ISDA convention and complete monthly periods, so interest is exactly 1% per period before currency rounding.

| Date | Opening balance | Interest | Specified payment | Expected remaining balance | PyLoan payment |
|---|---:|---:|---:|---:|---:|
| 1 February 2026 | 1,000.00 | 10.00 | 100.00 | 910.00 | 100.00 |
| 1 March 2026 | 910.00 | 9.10 | 100.00 | 819.10 | **919.10** |

The independent reference uses Decimal arithmetic: `next balance = opening balance + rounded interest - payment`. It caps an early final payment at the remaining debt plus interest; a legitimate final payment can be smaller than the supplied amount.

```python
from pyloan import Loan
loan = Loan(loan_amount=1000, interest_rate=12,
            loan_term=2, loan_term_period='M',
            start_date='2026-01-01', payment_end_of_month=False,
            payment_amount=100)
for p in loan.get_payment_schedule()[1:]:
    print(p.date.date(), p.payment_amount, p.loan_balance_amount)
```

## Versions, history and candidate

The official PyPI 0.7.2 wheel and current master `73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1` have byte-identical package modules. Develop and tag v0.7.2 retain the same branch. The official 0.7.0 wheel is a historical control.

[Issue 67](https://github.com/darius-lesch/pyloan/issues/67) concerned unwanted residual balances when the package automatically sizes an annuity payment. [PR 68](https://github.com/darius-lesch/pyloan/pull/68/files) intentionally added a final adjustment in 0.7.1; [PR 69](https://github.com/darius-lesch/pyloan/pull/69/files) retained it in 0.7.2. The adjustment also reaches explicitly sized annuities, whose documented behavior allows a positive residual. That extension is the subject of this report. The known automatic-payment fix is credited, not presented as a newly discovered mechanism.

The local candidate excludes an annuity with non-None `payment_amount` from this final override. It preserves the existing automatic-annuity, linear and interest-only branches. No patch acceptance or merged correction is claimed.

## Measured checks

The main grid has 864 explicit-annuity scenarios, varying principal, positive interest, payment frequency, term, payment size and two 30/360 conventions. These are ordinary regular payments without special payments, fees or grace periods; the supplied payments exceed period interest. The reference is specific to these full 30/360 periods, not a general day-count oracle. All inputs and complete outputs are archived.

| Executed implementation | Payment-cap violations / 864 |
|---|---:|
| Official PyPI 0.7.0 | 0 |
| Official PyPI 0.7.2 | 408 |
| Pinned current source | 408 |
| Local candidate | 0 |
| Original branch restored | 408 |

All 456 previously passing schedules are unchanged. Candidate output matches 0.7.0 across all 864 scenarios. The release/current/restored output files match byte-for-byte.

Sixty separate automatic-annuity, linear and interest-only controls retain identical schedules and pass final-balance checks. Five focused regression tests have two failures on original/restored code and all five pass with the candidate.

**Remaining differences are retained:** the exact Decimal recurrence disagrees with 24 candidate scenarios by up to 0.02, also present in 0.7.0. These are separate rounding differences, outside this correction. The broader exact-oracle mismatch count is 420 before and 24 after; this report does not claim that every calculated amount is exact.

The current upstream unittest suite runs 16 tests and produces the same three failure reports in original, candidate and restored states: two snapshot subtests and one February/March day-count test. Normalized logs match. The upstream suite is **not green**.

## Disclosure and limitations

A bounded review covered the current source, develop, release artifacts, all 69 public issues/PRs available before disclosure, six targeted all-state searches, relevant bodies, complete returned diffs and comments, plus the GERO catalog. No exact previous report or proposed guard for this explicit-payment regression was found. This is not a claim of exhaustive private or unindexed coverage.

Execution used Python 3.12.14 and python-dateutil 2.9.0.post0 on macOS arm64, one configured CPU worker. Evidence concerns synthetic library schedules. No bank deployment, borrower loss, production usage, performance or legal conclusion was measured. Test/report preparation was AI-assisted.

Vendor report sent before external publication: [https://github.com/darius-lesch/pyloan/issues/70](https://github.com/darius-lesch/pyloan/issues/70). No response or upstream acceptance is claimed.
