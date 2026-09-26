"""Post-hoc checks of four v1 failures. Not new confirmation evidence."""
import json
from pathlib import Path

import mpmath as mp

from lab.black_scholes import _discounted_legs
from lab.corpus import arguments
from lab.metrics import rounding_class, text
from lab.reference import evaluate, integral_log_otm

ROOT = Path(__file__).resolve().parents[1]
IDS = ('confirmation-0158','confirmation-0197','confirmation-0222','confirmation-0314')


def main():
    base = json.loads((ROOT/'evidence/benchmark-v1.json').read_text())
    output = ROOT/'evidence/diagnostics-v1.json'
    if output.exists():
        raise FileExistsError('Preserve post-hoc evidence')
    rows = []
    for row in base['rows']:
        if row['id'] not in IDS:
            continue
        args = arguments(row)
        a, b = evaluate(*args,dps=260), evaluate(*args,dps=360)
        integral = integral_log_otm(*args,dps=120)
        _, _, candidate_m = _discounted_legs(*args[:5])
        with mp.workdps(360):
            relative = abs(a.otm-b.otm)/b.otm
            log_change = abs(a.log_otm-b.log_otm)
            log_integral = abs(integral-b.log_otm)
            normalized = b.otm / mp.sqrt(mp.mpf(args[0])*mp.mpf(args[1]))
            result = {
                'id':row['id'], 'precision_levels':[260,360],
                'price_relative_change':text(relative), 'log_absolute_change':text(log_change),
                'integral_dps':120, 'integral_log_absolute_difference':text(log_integral),
                'reference_otm':text(b.otm), 'reference_moneyness':text(b.m),
                'candidate_moneyness_hex':candidate_m.hex(),
                'normalized_price_exact_inputs':text(normalized),
                'normalized_rounding_class':rounding_class(normalized),
                'final_price_rounding_class':rounding_class(b.otm),
                'passed':bool(relative<mp.mpf('1e-200') and log_integral<mp.mpf('1e-90')),
            }
        if not result['passed']:
            raise ArithmeticError('Diagnostic disagreement: '+row['id'])
        rows.append(result)
    report = {'status':'post-hoc diagnostic, not an additional holdout', 'rows':rows}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
