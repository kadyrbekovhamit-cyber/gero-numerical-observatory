# When gross-up does not return the requested net: two loan-calculator defects

Xamit Kadirbekov · GERO Research · 3 October 2026

**Status: reproduced locally; reported upstream; maintainer review pending.**

A gross-up routine should answer a simple question: how much principal is needed so that the requested amount remains after the modeled deductions? Applying the package’s own deductions to its answer provides a useful consistency check. This report records two separate formula errors under that contract.

[Official developer report](https://github.com/yanomateus/loan-calculator/issues/15) · [GERO article](https://www.gero.uz/research/articles/loan-calculator-iof-grossup-net-reconstruction.html)

## Summary

In loan-calculator 1.2.2, `IofGrossup` does not reconstruct the requested net
principal for two of its supported schedules when its result is composed with
the package's own `loan_iof` function:

1. Progressive Price uses the regressive amortization/tax pairing.
2. Constant amortization divides the average tax coefficient by an additional
   Price discount sum.

The examples use `start_date == reference_date`, zero grace period, increasing
equally spaced dates, and explicit synthetic fee inputs. This is an internal
API-consistency report, not a claim about current Brazilian tax law or actual
borrower losses. It does not depend on the separate irregular-date schedule
interpretation question.

## Minimal public-API reproduction

Environment: loan-calculator 1.2.2, Python 3.9.6, macOS arm64. The wheel has
SHA-256 `d3ebdfe35b751acc3716125db68d4470fab06835493e9c5be7797695673eb920`.
The relevant source is unchanged at master
`8c5a1a254e1fe1087fcb623438bd74b748022467`, checked 3 October 2026.

```python
from datetime import date, timedelta
from loan_calculator.loan import Loan
from loan_calculator.grossup.iof import IofGrossup
from loan_calculator.grossup.iof_tax import loan_iof

start = date(2026, 1, 1)
dates = [start + timedelta(days=n) for n in (1, 2)]

for schedule, net, daily_rate in [
    ("constant-amortization-schedule", 1021, 0),
    ("progressive-price-schedule", 764, 1),
]:
    # daily_rate=1 is deliberately extreme, solely for exact small arithmetic.
    annual_rate = (1 + daily_rate)**365 - 1
    base = Loan(net, annual_rate, start, dates,
                amortization_schedule_type=schedule)
    result = IofGrossup(base, start, 1/256, 0, 0).grossed_up_loan
    recovered = result.principal - loan_iof(
        result.principal, result.amortizations, [1, 2], 1/256, 0)
    print(schedule, "requested", net, "gross", result.principal,
          "amortizations", result.amortizations, "recovered", recovered)
```

Observed output:

```text
constant-amortization-schedule requested 1021 gross 1024.0 amortizations [512.0, 512.0] recovered 1018.0
progressive-price-schedule requested 764 gross 768.0 amortizations [256.0, 512.0] recovered 763.0
```

The expected recovered net is respectively 1021 and 764. These are not currency
rounding artifacts: the displayed deductions and amortizations are exact
binary-representable values. Under the same schedule/tax model the correct gross
principals are respectively `522752/509` and `586752/763`.

For an additional moderate-rate example, with net=10000, requested daily rate
0.0005 (converted to annual as above), days=[30,60], daily fee=0.000082,
complementary fee=0.0038 and service fee=0:

| Schedule | Returned gross | Recovered net | Expected gross (independent Decimal80 reference) |
|---|---:|---:|---:|
| Progressive | 10075.371613048566 | 9999.814159333664 | 10075.558857905752 |
| Constant | 10057.194454089487 | 9981.866067628358 | 10075.465234607208 |

The fees in this example are model inputs only. The reference uses the actual
`base.daily_interest_rate` after annual conversion.

## Cause and proposed correction

Let `v_i = (1+d)**(-n_i)`, `D = sum(v_i)`, and
`w_i = min(n_i * daily_iof_fee, 0.015)`. If `S` is gross principal, the schedule
and tax API together imply:

- Regressive: `A_i/S = v_i/D`, so `alpha = sum(w_i*v_i)/D`.
- Progressive: `A_i/S = v_(k+1-i)/D`, so `alpha = sum(w_i*v_(k+1-i))/D`.
- Constant: `A_i/S = 1/k`, so `alpha = sum(w_i)/k`.

In each case the net is `S*(1-alpha-complementary_fee-service_fee)`.

In `br_iof_progressive_price_grossup`, reversing the whole summand only changes
summation order; it does not reverse one side of the pairing. A minimal change is:

```diff
     iof_coef = sum(
-        float(min(n * d_iof, 0.015)) / (1 + d) ** n
-        for n in pmt_days[::-1]
+        float(min(n * d_iof, 0.015)) / (1 + d) ** discount_day
+        for n, discount_day in zip(pmt_days, reversed(pmt_days))
     )
```

In `br_iof_constant_amortization_grossup`, `iof_coef` already is the mean of
`w_i`; its return should be:

```diff
-    return p / (1 - (iof_coef / transport_coef) - c_iof - s_fee)
+    return p / (1 - iof_coef - c_iof - s_fee)
```

The now-unused constant-schedule discount calculation can be removed while
preserving the public signature. The displayed alpha formulas also need
correction: the regressive/progressive docstrings have mismatched pairings,
and the constant docstring repeats the extra divisor. The existing regressive
implementation should remain unchanged.

## Validation and limits

A separate-process original / proposed correction / restored-original check
covered 12 input configurations crossed with all three schedules (36 cases):
**14 failures / 0 failures / 14 failures**. Original and restored observations
were identical. The original failures were 5 progressive and 9 constant cases;
all 12 regressive controls passed. This is two mechanisms, not 14 distinct bugs.

The reference independently allocates amortizations with Decimal precision 80
and finds the gross amount by bisection of the net-cash-flow equation. Exact
rational calculations independently support the two small witnesses. Both the
public `Loan -> IofGrossup -> loan_iof` composition and the direct functions were
checked. Controls include zero daily tax, fee-only, zero interest, one payment,
all-capped tax weights, 12 payments, and changes of principal scale. Zero
interest and one payment do **not** generally mask the constant-schedule bug.

No security exploit, current tax-law correctness, actual deployment impact,
reward eligibility, worldwide priority, or support for all possible inputs is
claimed. The current public issues/PRs, releases and direct-function history
were searched; I did not find an exact existing report or correction. The
historical generic solver already used the net/gross composition identity,
so that identity itself is not presented as new.

This report and proposed correction were prepared with AI assistance. The
numerical results above were obtained by executing the released wheel and an
isolated temporary copy with the proposed corrections. The report remains
subject to maintainer review.

## Additional internal reproduction

A separate Python 3.12.13 run of the progressive and regressive Price example returned identical gross principals and amortization arrays. Feeding the corrected progressive gross principal back through the package schedule recovered the requested net to approximately 1e-12. This additional check covers the progressive defect and a regressive control, not the constant-amortization defect. Its Decimal calculations use decimal roundtrip representations of floats; they are not an exact binary64 conversion. It is an internal reproduction, not external peer review.

The author used AI assistance for investigation, code and writing. The results were generated by executed package calls. They do not demonstrate a fully autonomous Hunter or superiority over other models.
