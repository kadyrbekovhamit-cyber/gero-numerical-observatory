**Describe the bug**

An explicitly specified annuity payment is replaced by a final balloon payment in PyLoan 0.7.2. The current quickstart permits an explicit payment that leaves a residual balance, but `get_payment_schedule()` forces the last non-interest-only principal repayment to the full remaining balance.

**To reproduce**

```python
from pyloan import Loan

loan = Loan(
    loan_amount=1000, interest_rate=12,
    loan_term=2, loan_term_period="M",
    start_date="2026-01-01", payment_end_of_month=False,
    payment_amount=100,
)
for p in loan.get_payment_schedule()[1:]:
    print(p.date.date(), p.payment_amount, p.interest_amount,
          p.principal_amount, p.loan_balance_amount)
```

Official PyPI 0.7.2 and current master print:
```text
2026-02-01 100.00 10.00 90.00 910.00
2026-03-01 919.10 9.10 910.00 0.00
```

**Expected behavior**

For these full monthly 30E/360 ISDA periods the interest rate per period is exactly 1%. With a specified gross payment of 100, the recurrence is remaining balance = opening balance + rounded interest - payment. The second row should be:
```text
2026-03-01 100.00 9.10 90.90 819.10
```

The [current pinned documentation](https://github.com/darius-lesch/pyloan/blob/73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1/docs/docsrc/source/quickstart.rst#L69) explicitly allows incomplete amortization when `payment_amount` is provided. A final payment may be smaller when the debt is cleared early; it should not silently become larger in this example.

**Environment and verification**

- macOS 15.5 arm64, Python 3.12.14; CPU, one configured worker.
- Official PyPI 0.7.2 wheel SHA256: `a0a24cb9b545d05d49e61186d7dded865511f95be20703031de79b7b0f0d16cb`. Its six package modules were checked byte-for-byte against master `73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1`. Develop and v0.7.2 have the same affected source.
- 864 explicit-annuity scenarios (ordinary regular payments, no special payments, fees or grace periods; specified payments exceed period interest): 408 payment-cap violations in release/current, 0 with the candidate guard, and 408 after restoring the original branch. All 456 previously passing schedules are unchanged. Candidate schedules match the official 0.7.0 wheel across all 864 scenarios.
- 60 separate automatic-annuity, linear and interest-only controls are unchanged; their final-balance checks pass. Five focused regression tests have 2 failures before/restoration and all 5 pass with the candidate.
- Limits retained: 24 scenarios differ from an exact Decimal recurrence by up to 0.02, also in 0.7.0; this candidate does not fix that separate rounding behavior. Upstream unittest discovery runs 16 tests and has the same 3 existing failure reports before, after and on restoration (two snapshot subtests and the February/March day-count test). This is not a claim of a green upstream suite.

**Additional context / candidate correction**

I reviewed #67, #68 and #69, including the diffs and comments. The final adjustment in #68 intentionally fixes automatically sized loans; this report concerns its effect on the separately documented explicit-payment mode. Version 0.7.0 preserves that mode, while 0.7.1 introduced the override. I found no exact previous report of this explicit-payment regression in the checked public history.

A focused candidate is to exclude explicit annuities from the final override, retaining the existing automatic, linear and interest-only behavior:

```python
if (is_last_payment and self.loan_type != LoanType.INTEREST_ONLY
        and not (self.loan_type == LoanType.ANNUITY
                 and self.payment_amount is not None)):
    principal_amount = balance_bop
```

I can help with a focused patch and regression tests. These are synthetic library checks, not evidence of bank deployment or borrower loss. Research, test preparation and this report were AI-assisted; no upstream acceptance is claimed.
