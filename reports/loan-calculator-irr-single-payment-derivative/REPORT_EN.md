# A one-payment loan breaks IRR: an incorrect derivative in loan-calculator

**Xamit Kadirbekov / GERO · 3 October 2026**

**Reproduced locally; [reported to the maintainer in issue #16](https://github.com/yanomateus/loan-calculator/issues/16); review pending.** This is one IRR derivative-construction report, with a single-payment crash and a separately verified derivative mismatch. It is not a claim of two additional independent discoveries.

## The observable failure

`loan-calculator` 1.2.2 constructs a valid one-payment loan and its gross-up object, but reading the public `IofGrossup.irr` property raises `IndexError`. The derivative factory indexes `return_days[-2]`, which does not exist for one payment.

The error is separate from the earlier [gross-up report, issue #15](https://github.com/yanomateus/loan-calculator/issues/15). This example deliberately uses the regressive schedule and zero tax and fees. Its principal and payment schedule are consistent before IRR is requested.

```python
from datetime import date
from loan_calculator import Loan, IofGrossup

start = date(2026, 1, 1)
loan = Loan(
    10000, 0.12, start, [date(2026, 1, 31)],
    amortization_schedule_type="regressive-price-schedule",
)
grossup = IofGrossup(loan, start, 0, 0, 0)
print(grossup.grossed_up_loan.due_payments)
# [10093.582031654134]
print(grossup.irr)
# IndexError: list index out of range
```

There is a unique positive daily rate for the positive payment in this example. Solving `10000 = 10093.582031654134 / (1 + c)**30` gives approximately `0.000310537755655373`. With the proposed derivative correction, the public property returns `0.00031053775565537123`, and the reconstructed present value differs from the principal by zero at the reported float precision.

A still smaller public module example is:

```python
from loan_calculator.irr import approximate_irr
approximate_irr(100, [110], [1], 0.05)
# Expected: approximately 0.1; actual: IndexError
```

## The derivative also differs from the documented polynomial

For principal 100, payments `[60, 60]`, and days `[1, 2]`, the package's return polynomial is:

```text
f(c) = 100(1 + c)^2 - 60(1 + c) - 60
f'(c) = 200(1 + c) - 60
```

At `c = 0.5`, the exact derivative is **240**. The current factory returns **140**:

```python
from loan_calculator.irr import return_polynomial_derivative_factory
return_polynomial_derivative_factory(100, [60, 60], [1, 2])(0.5)
# 140.0; expected 240
```

The construction uses the final inter-payment interval instead of the final day for the leading coefficient, gives payment terms positive signs, and pairs coefficients with shifted exponents. A Newton iteration can nevertheless converge with this incorrect derivative. Accordingly, **this audit does not claim that every multi-payment IRR is wrong**: all 13 multi-payment IRR control cases in the bounded corpus pass both before and after the correction.

## Proposed correction and verification

The supplied patch differentiates the existing polynomial term by term: keep its signed coefficients, multiply each by its own exponent, reduce that exponent by one, and omit constant terms. It does not change gross-up, payment schedules, the Newton stopping policy, or the initial guess.

Tests run in three fresh processes: the unmodified wheel, the same wheel with only this patch, and a fresh unmodified wheel again. The original wheel is never edited.

| Check | Cases | Original failures | Proposed patch failures | Restored failures |
|---|---:|---:|---:|---:|
| Derivative values / construction | 15 | 15 | 0 | 15 |
| Direct `approximate_irr` calls | 8 | 4 | 0 | 4 |
| Public `Loan` → `IofGrossup.irr` | 12 | 3 | 0 | 3 |
| Total | 35 | 22 | 0 | 22 |

Baseline and restored observations match exactly. These are test failures, not 22 different bugs.

The derivative reference uses exact rational dual-number Horner evaluation of an expanded polynomial. Complex-step evaluation of the package's original return polynomial independently agrees with that reference. The rate reference uses Decimal80 bisection on discounted cash flows, without calling the package's Newton solver or derivative. Decimal inputs preserve the actual binary float arguments with `Decimal.from_float`.

Public API cases use a 10,000 principal, 12% annual rate, 1/2/12/36 monthly payments, zero tax, and service fees 0/1%/10%. No production borrowers or accounts were involved. The timed verification phase in each process took under 0.1 seconds on the test machine; this is a local observation, not a performance guarantee. One CPU worker used the existing shared compute lock; no GPU, network calculation, or paid model API was used.

## Version, source and novelty boundary

- Distribution: `loan-calculator` **1.2.2**, executed on **Python 3.12.13**, macOS arm64.
- Wheel SHA-256: `d3ebdfe35b751acc3716125db68d4470fab06835493e9c5be7797695673eb920`.
- Freshly checked upstream head: [`8c5a1a254e1fe1087fcb623438bd74b748022467`](https://github.com/yanomateus/loan-calculator/tree/8c5a1a254e1fe1087fcb623438bd74b748022467).
- The checked IRR implementation, public gross-up wrapper and Loan source match the release wheel byte for byte.
- [Official API documentation](https://loan-calculator.readthedocs.io/en/latest/loan_calculator.html#module-loan_calculator.irr) describes the polynomial and derivative; the pinned source docstring gives the same contract.
- The reviewed file history introduces the faulty derivative in commit [`6c8198532c0777522058f05303f14012949b85e2`](https://github.com/yanomateus/loan-calculator/commit/6c8198532c0777522058f05303f14012949b85e2). Later refactoring retains it.

The duplicate review covers current open/closed issues and PR bodies, available reviews and comments, relevant source history, branches/tags, release notes and the GERO catalog. PRs #3 and #8 concern earlier IRR implementation work; neither is a prior report of this failure or a current correction. No exact predecessor was found within that scope. This is **not proof of worldwide priority**. The dated source and novelty receipts state the coverage.

## Limits and attribution

The correction is proposed, not an accepted upstream release. The bounded tests do not establish correctness for every root configuration, negative rates, zero initial guesses, or solver non-convergence. No security exploit, actual customer loss, current tax-law correctness, bounty eligibility or payment is claimed.

Investigation, verification code and writing used AI assistance. This is not external peer review or evidence that Hunter can perform the full workflow autonomously.

Original report: CC BY 4.0. Original verifier and proposed patch: MIT. The bundled upstream wheel retains its original MIT license.
