#!/usr/bin/env python3
"""Offline public-API regression, with a Decimal80 cash-flow oracle.

One fresh process per baseline/patched/restored mode. Original wheel stays
unchanged. Fees are explicit synthetic model inputs, not current tax advice.
"""
import argparse
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, localcontext
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
WHEEL = HERE / 'vendor/loan_calculator-1.2.2-py2.py3-none-any.whl'
LOCK = Path(os.environ.get('GERO_COMPUTE_LOCK', str(Path(tempfile.gettempdir()) / 'gero-loan-grossup.lock')))
SCHEDULES = ('progressive_price_schedule', 'regressive_price_schedule', 'constant_amortization_schedule')


def patch_source(source):
    old = '''float(min(n * d_iof, 0.015)) / (1 + d) ** n
        for n in pmt_days[::-1]'''
    new = '''float(min(n * d_iof, 0.015)) / (1 + d) ** discount_day
        for n, discount_day in zip(pmt_days, reversed(pmt_days))'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    start = source.index('def br_iof_constant_amortization_grossup(')
    first, constant = source[:start], source[start:]
    old = 'return p / (1 - (iof_coef / transport_coef) - c_iof - s_fee)'
    assert constant.count(old) == 1
    return first + constant.replace(old, 'return p / (1 - iof_coef - c_iof - s_fee)')


def oracle(net, daily, days, daily_fee, complementary, fee, schedule):
    """Value the model's own cash flows, then invert them by bisection.

    No package calls and no grossup formula are used in this reference.
    Decimal inputs describe the actual binary float values sent to the API.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        cv = lambda x: Decimal.from_float(float(x))
        n, q, f, c, g = cv(net), 1 + cv(daily), cv(daily_fee), cv(complementary), cv(fee)
        weights = [min(Decimal(day) * f, Decimal('0.015')) for day in days]
        if schedule == 'constant_amortization_schedule':
            fractions = [Decimal(1) / len(days)] * len(days)
        else:
            discounts = [q ** (-day) for day in days]
            total = sum(discounts)
            fractions = [v / total for v in discounts]
            if schedule == 'progressive_price_schedule':
                fractions.reverse()
        def remaining(principal):
            payments = [principal * share for share in fractions]
            return principal - sum(a * w for a, w in zip(payments, weights)) - principal * c - principal * g
        low, high = n, n * 2
        assert remaining(low) <= n <= remaining(high)
        for _ in range(240):
            mid = (low + high) / 2
            if remaining(mid) < n:
                low = mid
            else:
                high = mid
        gross = (low + high) / 2
        return gross, fractions, remaining(gross) - n


def scenarios():
    # Each case is crossed with three schedule types. Equal-spaced main
    # examples avoid the separate irregular-date contract dispute.
    cases = [
        ('two_equal_periods', 10000, .0005, [30, 60], .000082, .0038, 0),
        ('twelve_equal_periods', 10000, .0005, list(range(30, 361, 30)), .000082, .0038, .02),
        ('zero_interest', 10000, 0, [30, 60], .000082, .0038, 0),
        ('single_payment', 10000, .0005, [30], .000082, .0038, 0),
        ('both_capped', 10000, .0005, [200, 400], .000082, .0038, 0),
        ('zero_tax', 10000, .0005, [30, 60], 0, 0, 0),
        ('fee_only', 10000, .0005, [30, 60], 0, 0, .02),
        ('zero_daily_tax', 10000, .0005, [30, 60], 0, .0038, .02),
        ('small_principal', 1, .0005, [30, 60], .000082, .0038, 0),
        ('large_principal', 1e6, .0005, [30, 60], .000082, .0038, 0),
        ('constant_rational_witness', 1021, 0, [1, 2], 1/256, 0, 0),
        ('progressive_rational_witness', 764, 1, [1, 2], 1/256, 0, 0),
    ]
    for case in cases:
        for schedule in SCHEDULES:
            yield case + (schedule,)


def execute(mode, source_root, source_hash):
    sys.path.insert(0, str(source_root))
    from loan_calculator.loan import Loan
    from loan_calculator.grossup.iof import IofGrossup
    from loan_calculator.grossup.iof_tax import loan_iof
    from loan_calculator.grossup import functions
    rows = []
    reference = date(2026, 1, 1)
    for name, net, daily, days, tax, comp, fee, schedule in scenarios():
        annual = (1 + daily) ** 365 - 1
        base = Loan(net, annual, reference, [reference + timedelta(days=d) for d in days],
                    amortization_schedule_type=schedule.replace('_', '-'))
        converted_daily = base.daily_interest_rate
        gross = IofGrossup(base, reference, tax, comp, fee).grossed_up_loan
        withheld = loan_iof(gross.principal, gross.amortizations, days, tax, comp)
        reconstructed_net = gross.principal - withheld - gross.principal * fee
        expected, shares, oracle_residual = oracle(net, converted_daily, days, tax, comp, fee, schedule)
        tolerance = max(1e-9, abs(net) * 1e-11)
        # Independently check that the reference allocates the same schedule
        # as the public package before using it to judge grossup.
        allocation_error = max(abs(a - gross.principal * float(s)) for a, s in zip(gross.amortizations, shares))
        assert allocation_error <= tolerance
        direct = getattr(functions, 'br_iof_' + schedule.replace('_schedule', '') + '_grossup')(
            net, converted_daily, tax, comp, days, fee)
        assert abs(direct - gross.principal) <= tolerance
        residual = reconstructed_net - net
        rows.append({'name': name, 'schedule': schedule,
                     'input': {'net': net, 'requested_daily_rate': daily, 'actual_daily_rate': converted_daily,
                               'annual_rate': annual, 'return_days': days, 'daily_fee': tax,
                               'complementary_fee': comp, 'service_fee': fee, 'grace_period': 0,
                               'start_equals_reference': True},
                     'gross': gross.principal, 'amortizations': gross.amortizations,
                     'reconstructed_net': reconstructed_net, 'residual': residual,
                     'reference_gross_decimal80': str(expected), 'reference_net_residual': str(oracle_residual),
                     'tolerance': tolerance, 'allocation_error': allocation_error,
                     'passed': abs(residual) <= tolerance and abs(gross.principal - float(expected)) <= tolerance})
    # Exact rational witnesses are independent of the Decimal reference.
    exact = {'progressive': {'baseline_gross': '768', 'baseline_amortizations': ['256', '512'],
                            'baseline_net': str(Fraction(768) - 256 * Fraction(1, 256) - 512 * Fraction(1, 128)),
                            'expected_gross': str(Fraction(764) / (1 - Fraction(5, 768)))},
             'constant': {'baseline_gross': '1024', 'baseline_amortizations': ['512', '512'],
                          'baseline_net': str(Fraction(1024) - 512 * Fraction(1, 256) - 512 * Fraction(1, 128)),
                          'expected_gross': str(Fraction(1021) / (1 - Fraction(3, 512)))}}
    return {'mode': mode, 'python': sys.version.split()[0], 'source_functions_sha256': source_hash,
            'wheel_sha256': hashlib.sha256(WHEEL.read_bytes()).hexdigest(), 'cases': rows,
            'summary': {'total': len(rows), 'failures': sum(not r['passed'] for r in rows),
                        'failures_by_schedule': dict(Counter(r['schedule'] for r in rows if not r['passed']))},
            'exact_rational_witnesses': exact,
            'limits': ['Explicit synthetic model inputs; no statement of current tax law.',
                       'Only start_date=reference_date and grace_period=0 are covered.',
                       'All tested return dates have equal intervals; irregular-date dispute is excluded.',
                       'No security impact, actual borrower loss, bounty eligibility, or worldwide priority established.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'patched', 'restored'))
    args = parser.parse_args()
    # Advisory lock shared with the installed Hunter and other numerical jobs.
    with LOCK.open('a+b') as lock, tempfile.TemporaryDirectory(prefix='gero-grossup-') as temp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(temp)
        with zipfile.ZipFile(WHEEL) as z:
            for name in z.namelist():
                p = Path(name)
                if p.parts[0] == 'loan_calculator' and not p.is_absolute() and '..' not in p.parts and not name.endswith('/'):
                    out = root / p
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
        function_file = root / 'loan_calculator/grossup/functions.py'
        if args.mode == 'patched':
            function_file.write_text(patch_source(function_file.read_text()))
        result = execute(args.mode, root, hashlib.sha256(function_file.read_bytes()).hexdigest())
        (HERE / ('replay-' + args.mode + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
