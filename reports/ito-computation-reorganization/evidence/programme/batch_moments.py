#!/usr/bin/env python3
"""Serial many-query reorganization of the SAME discrete Ito left sums.

Known linearity/moment aggregation, no novelty or improved statistical error.
All preparation belongs to the cached algorithm's measured end-to-end cost.
"""
from decimal import Decimal, localcontext
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
import tracemalloc

ROOT = Path(__file__).resolve().parent


def horner(c, x):
    result = c[-1]
    for a in reversed(c[:-1]):
        result = result*x+a
    return result


def direct(nodes, coeffs):
    return [math.fsum(horner(c, x)*(y-x) for x, y in zip(nodes, islice(nodes, 1, None))) for c in coeffs]


def moments(nodes, degree):
    """Single pass; compensated accumulation; O(D) working memory."""
    totals, compensations = [0.]*(degree+1), [0.]*(degree+1)
    for x, y in zip(nodes, islice(nodes, 1, None)):
        increment, value = y-x, 1.
        for d in range(degree+1):
            term = increment*value
            updated = totals[d]+term
            if abs(totals[d]) >= abs(term):
                compensations[d] += (totals[d]-updated)+term
            else:
                compensations[d] += (term-updated)+totals[d]
            totals[d] = updated
            value *= x
    return [x+y for x, y in zip(totals, compensations)]


def cached(nodes, coeffs):
    m = moments(nodes, max(map(len, coeffs))-1)
    return [math.fsum(a*b for a, b in zip(c, m)) for c in coeffs]


def table_baseline(nodes, coeffs):
    """Known shared-feature baseline, but does not aggregate the path first."""
    degree = max(map(len, coeffs))-1
    rows = []
    for x, y in zip(nodes, islice(nodes, 1, None)):
        powers = [1.]
        for _ in range(degree):
            powers.append(powers[-1]*x)
        rows.append([(y-x)*v for v in powers])
    return [math.fsum(math.fsum(a*b for a, b in zip(c, row)) for row in rows) for c in coeffs]


def reference_decimal(nodes, coeffs):
    with localcontext() as ctx:
        ctx.prec = 70
        xnodes = list(map(Decimal.from_float, nodes))
        out = []
        for c in coeffs:
            dc = list(map(Decimal.from_float, c))
            out.append(sum((horner(dc, x)*(y-x) for x, y in zip(xnodes, xnodes[1:])), Decimal(0)))
        return out


def run():
    run_started = time.perf_counter()
    seeds = (17, 43, 101, 271, 577)
    methods = {'direct_horner': direct, 'shared_feature_table': table_baseline, 'streamed_moment_cache': cached}
    records = []
    for n, degree in ((64, 4), (256, 8)):
        for count in (1, 8, 32):
            per_seed = []
            for seed in seeds:
                rng = random.Random(seed)
                started = time.perf_counter_ns()
                nodes = [0.]
                for _ in range(n):
                    nodes.append(nodes[-1]+rng.gauss(0, math.sqrt(1/n)))
                coeffs = [[rng.uniform(-1,1)/math.factorial(d+1) for d in range(degree+1)] for _ in range(count)]
                setup_ns = time.perf_counter_ns()-started
                outputs, times = {}, {name: [] for name in methods}
                # All methods rebuild own precomputations each timed call.
                for repetition in range(5):
                    order = list(methods)
                    if repetition%2:
                        order.reverse()
                    for name in order:
                        started = time.perf_counter_ns()
                        outputs[name] = methods[name](nodes, coeffs)
                        times[name].append(time.perf_counter_ns()-started)
                if not all(math.isfinite(value) for out in outputs.values() for value in out):
                    raise AssertionError(('nonfinite output', n, degree, count, seed))
                error = max(abs(x-y) for left in methods for right in methods
                            for x,y in zip(outputs[left], outputs[right]))
                if not math.isfinite(error) or error > 2e-10:
                    raise AssertionError(('algebraic mismatch',n,degree,count,seed,error))
                ref = reference_decimal(nodes, coeffs) if seed == seeds[0] else None
                decimal_errors = None if ref is None else {
                    name: float(max(abs(Decimal.from_float(v)-r) for v,r in zip(out, ref)))
                    for name,out in outputs.items()}
                peak = {}
                if seed == seeds[0]:
                    for name, method in methods.items():
                        tracemalloc.start()
                        method(nodes, coeffs)
                        peak[name] = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                stats = {name: {'samples_ns': v, 'median_ns': statistics.median(v), 'min_ns': min(v), 'max_ns': max(v),
                                'end_to_end_with_common_input_setup_ns': setup_ns+statistics.median(v)}
                         for name,v in times.items()}
                per_seed.append({'seed':seed,'common_input_setup_ns':setup_ns,'times':stats,
                                 'max_pairwise_absolute_difference':error,'decimal70_max_absolute_errors':decimal_errors,
                                 'separate_peak_traced_bytes':peak,
                                 'speed_ratio_direct_over_cache':stats['direct_horner']['median_ns']/stats['streamed_moment_cache']['median_ns'],
                                 'accounted_ratio_with_common_input_setup':stats['direct_horner']['end_to_end_with_common_input_setup_ns']/stats['streamed_moment_cache']['end_to_end_with_common_input_setup_ns']})
            ratios = [x['speed_ratio_direct_over_cache'] for x in per_seed]
            # Approximate t interval across independent seed cases, not universal timing evidence.
            center = statistics.mean(ratios)
            half = 2.776445105*statistics.stdev(ratios)/math.sqrt(len(ratios))
            full_ratios = [x['accounted_ratio_with_common_input_setup'] for x in per_seed]
            full_center = statistics.mean(full_ratios)
            full_half = 2.776445105*statistics.stdev(full_ratios)/math.sqrt(len(full_ratios))
            records.append({'n':n,'polynomial_degree':degree,'queries':count,
                'integrand_evaluations_direct':n*count,'polynomial_basis_terms_cache':n*(degree+1),
                'query_dot_terms_cache':count*(degree+1),'ratio_mean_across_seeds':center,
                'ratio_approx95_t_interval':[center-half,center+half],
                'accounted_ratio_with_common_input_setup_mean':full_center,
                'accounted_ratio_with_common_input_setup_approx95_t_interval':[full_center-full_half,full_center+full_half],
                'seeds':per_seed})

    # Explicit ill-conditioned monomial-basis stress test, NOT a Brownian sample.
    nodes = [1.+.001*i/16 for i in range(17)]
    c = [float(math.comb(8,d)*(-1)**(8-d)) for d in range(9)]
    reference = reference_decimal(nodes,[c])[0]
    values = {name:method(nodes,[c])[0] for name,method in methods.items()}
    values['known_factored_polynomial'] = math.fsum((x-1.)**8*(y-x) for x,y in zip(nodes,nodes[1:]))
    if not reference.is_finite() or not all(math.isfinite(v) for v in values.values()):
        raise AssertionError('nonfinite stability output')
    stability = {'description':'deterministic cancellation stress f=(x-1)^8 near1, not Brownian data',
                 'decimal70_reference':str(reference),
                 'methods':{name:{'value':value,'absolute_error':float(abs(Decimal.from_float(value)-reference)),
                                  'relative_error':float(abs((Decimal.from_float(value)-reference)/reference))}
                            for name,value in values.items()}}
    result = {'status':'PASS','workers':1,'python':platform.python_version(),'platform':platform.platform(),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'total_elapsed_seconds_before_json_write':time.perf_counter()-run_started,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,
        'stability_negative_control':stability,
        'limitations':['same finite left sums; no reduction in discretization or Monte Carlo error',
            'ordinary linearity/batched moment aggregation, not a new method relative to that known analogue',
            'wall time depends on Python implementation and current machine load',
            'all cache preparation included in timed calls; common input generation separately listed',
            'setup-inclusive times are accounted sums of separate measurements, not contiguous runs',
            'tracemalloc measures traced Python allocations, not total process memory or RSS',
            'monomial coordinates can be unstable, see explicit negative control',
            'timing intervals across five seeds are descriptive approximate t intervals']}
    (ROOT/'batch-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','summary':[{'n':x['n'],'D':x['polynomial_degree'],'queries':x['queries'],
        'speed_ratio_direct_over_cache_mean':x['ratio_mean_across_seeds'],'interval':x['ratio_approx95_t_interval']} for x in records],
        'stability':stability},indent=2))


if __name__ == '__main__':
    run()
