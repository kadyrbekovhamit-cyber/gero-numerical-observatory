#!/usr/bin/env python3
"""Offline IRR audit. One process, shared compute lock, no extra dependencies."""
import argparse
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
import difflib
import fcntl
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
WHEEL = HERE / 'vendor/loan_calculator-1.2.2-py2.py3-none-any.whl'
if not WHEEL.exists():
    WHEEL = HERE.parent / 'research-20260930-loan-calculator' / WHEEL.name
EXPECTED_SHA = 'd3ebdfe35b751acc3716125db68d4470fab06835493e9c5be7797695673eb920'


def patched_source(source):
    start = source.index('    derivative_coefficients_vec = [')
    end = source.index('\n\ndef approximate_irr(', start)
    replacement = '''    last_day = return_days[-1]
    coefficients = [net_principal] + [-r for r in returns]
    exponents = [last_day - day for day in [0] + return_days]

    def return_polynomial_derivative(irr_):
        return sum(
            coefficient * exponent * (1 + irr_) ** (exponent - 1)
            for coefficient, exponent in zip(coefficients, exponents)
            if exponent != 0
        )

    return return_polynomial_derivative
'''
    return source[:start] + replacement + source[end:]


def fraction_derivative(principal, returns, days, rate):
    """Exact dual-number Horner evaluation of the expanded polynomial."""
    coefficients = [Fraction(0)] * (days[-1] + 1)
    coefficients[days[-1]] = Fraction(principal)
    for payment, day in zip(returns, days):
        coefficients[days[-1] - day] -= Fraction(payment)
    q, value, derivative = 1 + Fraction(rate), Fraction(0), Fraction(0)
    for coefficient in reversed(coefficients):
        derivative, value = derivative * q + value, value * q + coefficient
    return derivative


def cashflow_oracle(principal, returns, days):
    """Independent monotone discounted-cash-flow bisection, Decimal80."""
    with localcontext() as context:
        context.prec = 80
        principal = Decimal.from_float(float(principal))
        payments = [Decimal.from_float(float(r)) for r in returns]

        def npv(rate):
            return sum(r / (1 + rate) ** d for r, d in zip(payments, days)) - principal

        low, high = Decimal(0), Decimal(2)
        assert npv(low) >= 0 and npv(high) <= 0
        for _ in range(240):
            middle = (low + high) / 2
            if npv(middle) > 0:
                low = middle
            else:
                high = middle
        root = (low + high) / 2
        return str(root), str(npv(root))


def evaluate_irr(name, principal, payments, days, calculate):
    expected, reference_residual = cashflow_oracle(principal, payments, days)
    try:
        actual = calculate()
        residual = math.fsum(r / (1 + actual) ** d for r, d in zip(payments, days)) - principal
        error = None
        passed = math.isfinite(actual) and abs(actual - float(expected)) < 1e-10 and abs(residual) < max(1e-7, abs(principal) * 1e-10)
    except (IndexError, ZeroDivisionError, OverflowError, ValueError) as exc:
        actual, residual, passed = None, None, False
        error = type(exc).__name__ + ': ' + str(exc)
    return dict(name=name, principal=principal, payments=payments, days=days,
                expected_decimal80=expected, reference_npv_residual=reference_residual,
                actual=actual, npv_residual=residual, error=error, passed=passed)


def execute():
    from loan_calculator import Loan, IofGrossup
    from loan_calculator.irr import approximate_irr, return_polynomial_derivative_factory, return_polynomial_factory
    derivatives = []
    # Small exact polynomial cases, with points exactly representable in binary.
    for days, returns in [([1], [110]), ([30], [110]), ([1, 2], [60, 60]),
                          ([2, 5], [40, 80]), ([1, 3, 6], [30, 40, 50])]:
        for rate in [0, Fraction(1, 8), Fraction(1, 2)]:
            expected = fraction_derivative(100, returns, days, rate)
            try:
                actual = return_polynomial_derivative_factory(100, returns, days)(float(rate))
                error = None
                passed = abs(actual - float(expected)) <= max(1e-10, abs(float(expected)) * 1e-12)
            except IndexError as exc:
                actual, passed, error = None, False, type(exc).__name__ + ': ' + str(exc)
            # An independent complex-step check of the ORIGINAL polynomial.
            complex_step = return_polynomial_factory(100, returns, days)(float(rate) + 1e-20j).imag / 1e-20
            assert abs(complex_step - float(expected)) <= max(1e-10, abs(float(expected)) * 1e-12)
            derivatives.append(dict(principal=100, returns=returns, days=days, rate=str(rate),
                                    actual=actual, expected_fraction=str(expected), complex_step=complex_step,
                                    error=error, passed=passed))
    direct = []
    for principal, returns, days, guess in [
        (100, [110], [1], .05), (100, [110], [30], .001),
        (100, [110], [365], .0001), (1000, [1050], [90], .0003),
        (100, [60, 60], [1, 2], .05), (100, [55, 55], [30, 60], .001),
        (100, [30, 40, 50], [30, 60, 90], .001),
        (100, [30, 40, 50], [20, 55, 110], .001),
    ]:
        direct.append(evaluate_irr('direct_' + str(len(direct)), principal, returns, days,
                      lambda p=principal, r=returns, d=days, g=guess: approximate_irr(p, r, d, g)))
    public = []
    start = date(2026, 1, 1)
    for periods in [1, 2, 12, 36]:
        for fee in [0, .01, .1]:
            days = [30 * (i + 1) for i in range(periods)]
            loan = Loan(10000, .12, start, [start + timedelta(days=d) for d in days],
                        amortization_schedule_type='regressive-price-schedule')
            grossup = IofGrossup(loan, start, 0, 0, fee)
            # Regressive + zero tax deliberately isolates this from issue #15.
            assert abs(grossup.grossed_up_principal * (1 - fee) - loan.principal) < 1e-8
            row = evaluate_irr('public_' + str(periods) + '_' + str(fee), loan.principal,
                               grossup.grossed_up_loan.due_payments, days, lambda: grossup.irr)
            row.update(annual_rate=.12, service_fee=fee, daily_tax=0, complementary_tax=0,
                       schedule='regressive-price-schedule', grace_period=0,
                       initial_daily_rate=loan.daily_interest_rate)
            public.append(row)
    groups = dict(derivative=derivatives, direct_irr=direct, public_loan=public)
    return dict(groups=groups, summary={name:dict(total=len(rows), failures=sum(not r['passed'] for r in rows))
                                       for name, rows in groups.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['baseline', 'patched', 'restored'])
    parser.add_argument('--lock', type=Path, default=Path.home() / 'Library/Application Support/GERO/compute.lock')
    args = parser.parse_args()
    assert hashlib.sha256(WHEEL.read_bytes()).hexdigest() == EXPECTED_SHA
    # Local GERO runs use the existing lock. Readers may supply --lock /tmp/irr.lock.
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    if not args.lock.exists():
        args.lock.touch()
    with args.lock.open('rb') as lock, tempfile.TemporaryDirectory(prefix='gero-irr-') as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(tmp)
        with zipfile.ZipFile(WHEEL) as archive:
            for member in archive.namelist():
                if member.endswith('.py') and member.startswith('loan_calculator/'):
                    target = root / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
        target = root / 'loan_calculator/irr.py'
        original = target.read_text()
        if args.mode == 'patched':
            changed = patched_source(original)
            target.write_text(changed)
            (HERE / 'proposed-fix.patch').write_text(''.join(difflib.unified_diff(
                original.splitlines(True), changed.splitlines(True),
                fromfile='a/loan_calculator/irr.py', tofile='b/loan_calculator/irr.py')))
        sys.path.insert(0, str(root))
        started = time.perf_counter()
        result = execute()
        result.update(mode=args.mode, python=platform.python_version(), wall_seconds=time.perf_counter() - started,
                      wheel_sha256=EXPECTED_SHA, irr_source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      no_network=True, cpu_workers=1, gpu=False,
                      limits=['Positive single-disbursement cash flows; not all IRR root configurations.',
                              'No zero initial guess, negative-rate, or non-convergence policy is repaired.',
                              'Full API uses regressive schedules and zero tax, avoiding the earlier gross-up defects.',
                              'Synthetic financial inputs; no claim of current tax-law correctness or real losses.'])
        (HERE / (args.mode + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(mode=args.mode, summary=result['summary'], seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
