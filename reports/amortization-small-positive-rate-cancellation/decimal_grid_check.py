from __future__ import annotations

from decimal import Decimal, getcontext
from itertools import product
from math import expm1, log1p


getcontext().prec = 80


def oracle(principal: int, annual_rate: str, periods: int, payments_per_year: int) -> Decimal:
    rate = Decimal(annual_rate) / Decimal(payments_per_year)
    if rate == 0:
        return Decimal(principal) / Decimal(periods)
    growth = (Decimal(1) + rate) ** periods
    return Decimal(principal) * rate * growth / (growth - Decimal(1))


def stable(principal: int, annual_rate: float, periods: int, payments_per_year: int) -> float:
    rate = annual_rate / payments_per_year
    if rate == 0:
        return principal / periods
    discount = -expm1(-periods * log1p(rate))
    return principal * rate / discount


principals = [1, 1000, 100000, 10_000_000]
rates = ["1e-15", "1e-12", "1e-9", "1e-6", "0.001", "0.01", "0.1", "1.0"]
periods = [1, 12, 60, 360, 1200]
frequencies = [1, 12, 24, 52]

checked = 0
cent_mismatches = []
max_abs_error = Decimal(0)

for principal, rate_text, period, frequency in product(principals, rates, periods, frequencies):
    expected = oracle(principal, rate_text, period, frequency)
    actual = Decimal.from_float(stable(principal, float(rate_text), period, frequency))
    error = abs(actual - expected)
    max_abs_error = max(max_abs_error, error)
    if round(actual, 2) != round(expected, 2):
        cent_mismatches.append((principal, rate_text, period, frequency, actual, expected))
    checked += 1

print(f"checked={checked}")
print(f"cent_mismatches={len(cent_mismatches)}")
print(f"max_abs_error={max_abs_error}")
for mismatch in cent_mismatches[:10]:
    print(mismatch)

raise SystemExit(bool(cent_mismatches))
