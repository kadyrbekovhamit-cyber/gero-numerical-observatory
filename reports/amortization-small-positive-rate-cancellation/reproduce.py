from __future__ import annotations

from decimal import Decimal, getcontext

from amortization.amount import calculate_amortization_amount


getcontext().prec = 80


def oracle(principal: int, annual_rate: str, periods: int, payments_per_year: int = 12) -> Decimal:
    rate = Decimal(annual_rate) / Decimal(payments_per_year)
    if rate == 0:
        return Decimal(principal) / Decimal(periods)
    growth = (Decimal(1) + rate) ** periods
    return Decimal(principal) * rate * growth / (growth - Decimal(1))


for rate_text in ("1e-15", "1e-12"):
    expected = oracle(100_000, rate_text, 360)
    try:
        actual: object = calculate_amortization_amount(100_000, float(rate_text), 360)
    except Exception as exc:
        actual = f"{type(exc).__name__}: {exc}"
    print(rate_text, "actual=", actual, "decimal_rounded=", round(expected, 2))
