"""Frozen-candidate audit: exact inputs, checked oracle, pinned native baseline."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import mpmath as mp

from lab.black_scholes import naive_prices, stable_detailed, stable_prices
from lab.corpus import arguments
from lab.metrics import log_error, price_error, round_binary64, rounding_class, text
from lab.reference import checked, integral_log_otm

ROOT = Path(__file__).resolve().parents[1]
CORPUS_SHA256 = '2867930dc4cd8f0a280450dd48d566eb37144323d3d6bffd937f247a2cd5b296'
INTEGRAL_INDICES = (0, 17, 58, 89, 116, 117, 145, 174, 212, 230, 256, 280, 300, 315)
METHODS = ('direct', 'candidate_v0', 'jackel_native')


def safe(value):
    return value if math.isfinite(value) else str(value)


def native_batch(cases):
    mappings, lines = [], []
    for case in cases:
        s, k, t, r, q, sigma = arguments(case)
        forward, discount = s * math.exp((r-q)*t), math.exp(-r*t)
        mappings.append((forward, discount))
        lines.append(' '.join(repr(x) for x in (forward, k, sigma, t)))
    result = subprocess.run([str(ROOT/'build/jackel_cli')],
                            input='\n'.join(lines)+'\n', text=True,
                            capture_output=True, check=True)
    values = [tuple(float(x) for x in line.split()) for line in result.stdout.splitlines()]
    if len(values) != len(cases) or any(len(v) != 2 for v in values):
        raise RuntimeError('Native output rows do not match frozen inputs')
    return mappings, values


def comparison(left, right, reference):
    if not math.isfinite(left) or not math.isfinite(right):
        return 'not_comparable'
    with mp.workdps(260):
        a, b = abs(mp.mpf(left)-reference), abs(mp.mpf(right)-reference)
        return 'candidate_closer' if a < b else 'other_closer' if a > b else 'equal_error'


def summarize(rows):
    result = {'cases': len(rows), 'rounding_classes': dict(Counter(r['rounding_class'] for r in rows)),
              'oracle_failures': sum(not r['convergence']['passed'] for r in rows),
              'oracle_precision_increased': sum(r['convergence']['precision_increased'] for r in rows),
              'integral_checks': sum('integral' in r for r in rows),
              'integral_failures': sum(not r['integral']['passed'] for r in rows if 'integral' in r),
              'candidate_clips': sum(bool(r['candidate_diagnostics'].get('clipped')) for r in rows),
              'candidate_otm_orientation_mismatches': sum(r['candidate_log']['otm_option'] != r['otm_option'] for r in rows),
              'methods': {}}
    nonzero = [r for r in rows if r['rounding_class'] in ('normal','subnormal')]
    for other in ('direct','jackel_native'):
        result['candidate_vs_'+other] = dict(Counter(r['comparisons'][other] for r in rows))
        result['candidate_vs_'+other+'_nonzero_rounded'] = dict(Counter(r['comparisons'][other] for r in nonzero))
    with mp.workdps(180):
        for method in METHODS:
            eligible = [r for r in nonzero if r['methods'][method]['finite']]
            worst = max(eligible, key=lambda r: mp.mpf(r['methods'][method]['relative']), default=None)
            result['methods'][method] = {
                'nonfinite_otm': sum(not r['methods'][method]['finite'] for r in rows),
                'negative_otm': sum(r['methods'][method]['finite'] and r['methods'][method]['value'] < 0 for r in rows),
                'unexpected_zero': sum(r['methods'][method]['value'] == 0 for r in nonzero),
                'rounded_reference_matches': sum(r['methods'][method]['rounded_ulp_distance'] == 0 for r in rows),
                'rounded_reference_matches_nonzero': sum(r['methods'][method]['rounded_ulp_distance'] == 0 for r in nonzero),
                'max_relative_nonzero': worst['methods'][method]['relative'] if worst else None,
                'max_relative_case': worst['id'] if worst else None,
                'max_rounded_ulp_distance_nonzero': max((r['methods'][method]['rounded_ulp_distance'] for r in eligible), default=None),
            }
        finite_log = [r for r in rows if r['candidate_log']['finite']]
        comparable_log = [r for r in finite_log if r['candidate_log']['comparable_same_leg']]
        worst_log = max(comparable_log, key=lambda r: mp.mpf(r['candidate_log']['real_ulp_error']), default=None)
        result['candidate_log'] = {
            'finite': len(finite_log),
            'comparable_same_leg': len(comparable_log),
            'excluded_orientation_mismatches': sum(not r['candidate_log']['comparable_same_leg'] for r in rows),
            'max_real_ulp_error': worst_log['candidate_log']['real_ulp_error'] if worst_log else None,
            'max_real_ulp_case': worst_log['id'] if worst_log else None,
        }
    return result


def run():
    corpus = ROOT/'evidence/corpus-v1.json'
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != CORPUS_SHA256:
        raise RuntimeError('Frozen corpus hash mismatch')
    cases = json.loads(corpus.read_text())['cases']
    build = json.loads((ROOT/'evidence/jackel-build-v1.json').read_text())
    if hashlib.sha256((ROOT/'build/jackel_cli').read_bytes()).hexdigest() != build['binary_sha256']:
        raise RuntimeError('Native binary differs from build manifest')
    for name, digest in build['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Vendored source changed after build: '+name)
    if not all(build['floating_environment'].values()):
        raise RuntimeError('Unsupported floating-point environment')
    mappings, native = native_batch(cases)
    rows = []
    for index, case in enumerate(cases):
        args = arguments(case)
        ref, convergence = checked(*args)
        if not convergence['passed']:
            raise ArithmeticError('Oracle convergence failed: '+case['id'])
        direct, diagnostics = naive_prices(*args), {}
        candidate = stable_prices(*args, diagnostics=diagnostics)
        detailed = stable_detailed(*args)
        fwd, discount = mappings[index]
        native_value = native[index][0 if ref.option == 'call' else 1] * discount
        values = {'direct': getattr(direct, ref.option),
                  'candidate_v0': getattr(candidate, ref.option), 'jackel_native': native_value}
        row = {**case, 'otm_option': ref.option, 'reference_otm': text(ref.otm),
               'reference_log_otm': text(ref.log_otm), 'reference_round_binary64_hex': round_binary64(ref.otm).hex(),
               'reference_actual_log_moneyness': text(ref.m), 'reference_total_vol': text(ref.total_vol),
               'rounding_class': rounding_class(ref.otm), 'convergence': convergence,
               'methods': {name: price_error(value, ref.otm, dps=ref.dps) for name, value in values.items()},
               'comparisons': {name: comparison(values['candidate_v0'], values[name], ref.otm)
                               for name in ('direct','jackel_native')},
               'candidate_diagnostics': {k: safe(v) if isinstance(v,float) else v for k,v in diagnostics.items()},
               'candidate_log': {**log_error(detailed.log_otm_value, ref.log_otm, dps=ref.dps),
                                 'otm_option': detailed.otm_option},
               'both_legs': {'direct': [safe(direct.call),safe(direct.put)],
                             'candidate_v0': [safe(candidate.call),safe(candidate.put)],
                             'jackel_native': [safe(x*discount) for x in native[index]]},
               'jackel_adapter': {'forward_binary64_hex': fwd.hex(), 'discount_binary64_hex': discount.hex()}}
        same_leg = detailed.otm_option == ref.option
        row['candidate_log']['comparable_same_leg'] = same_leg
        if not same_leg:
            # A wrong-side interface output is a contract failure, not a
            # same-leg numerical error. Preserve it separately for diagnosis.
            row['candidate_log']['cross_leg_contract_discrepancy'] = {
                key: row['candidate_log'][key] for key in ('absolute','real_ulp_error')}
            row['candidate_log']['absolute'] = None
            row['candidate_log']['real_ulp_error'] = None
        if args[3] != 0 or args[4] != 0:
            mapped, mapped_check = checked(fwd, args[1], args[2], 0., 0., args[5])
            if not mapped_check['passed']:
                raise ArithmeticError('Mapped oracle convergence failed')
            with mp.workdps(max(ref.dps,mapped.dps)):
                mapped_value = getattr(mapped, ref.option)*mp.mpf(discount)
                delta = abs(mapped_value-ref.otm)
                row['jackel_adapter'].update({
                    'mapped_reference': text(mapped_value), 'mapping_absolute_error': text(delta),
                    'mapping_relative_error': text(delta/ref.otm),
                    'evaluation_after_mapping': price_error(native_value,mapped_value),
                    'mapped_oracle_convergence': mapped_check,
                })
        if index in INTEGRAL_INDICES:
            integral = integral_log_otm(*args, dps=90)
            with mp.workdps(ref.dps):
                difference = abs(integral-ref.log_otm)
                row['integral'] = {'dps': 90, 'log_value': text(integral),
                                   'log_absolute_difference': text(difference),
                                   'passed': bool(difference <= mp.mpf('1e-50'))}
            if not row['integral']['passed']:
                raise ArithmeticError('Positive-integral cross-check failed: '+case['id'])
        rows.append(row)
    files = list((ROOT/'lab').glob('*.py')) + list((ROOT/'tests').glob('*.py')) + [ROOT/'EXPERIMENT_V1.md',ROOT/'lab/jackel_cli.cpp']
    return {
        'schema': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'contract': 'Exact binary64 BSM inputs; OTM price and log price; historical unmodified Jaeckel baseline',
        'limitations': ['Empirical converged reference, not certified interval arithmetic',
                        'Synthetic corpus; no market/model-fit/hedging claims',
                        'Historical mirrored Jaeckel revision; not verified latest upstream',
                        'End-to-end price comparison, not a pure normalized-kernel comparison',
                        'Confirmation data now consumed; no tuning in this experiment'],
        'measurement_revision': 'Same-leg log errors exclude orientation mismatches after internal review; ordinary candidate and all price comparisons unchanged',
        'environment': {'python': sys.version, 'mpmath': mp.__version__, 'platform': platform.platform()},
        'corpus_sha256': CORPUS_SHA256, 'code_sha256': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        'native_build': build, 'integral_indices': INTEGRAL_INDICES,
        'summary': summarize(rows),
        'by_partition': {name: summarize([r for r in rows if r['partition'] == name]) for name in ('development','confirmation')},
        'by_family': {name: summarize([r for r in rows if r['family'] == name]) for name in sorted({r['family'] for r in rows})},
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/benchmark-v1.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing evidence; choose a new --output path')
    report = run()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'summary':report['summary'],'by_partition':report['by_partition']},indent=2))


if __name__ == '__main__':
    main()
