"""Independent float32 rsqrt directional-derivative screen; no MLX Python import."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parent


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(b):
    return struct.unpack('<f', struct.pack('<I', int(b)))[0]


def exact(x):
    n, d = x.as_integer_ratio()
    return mp.mpf(n) / d


def rounded_reference(x):
    """Round the high-precision value directly to binary32, ties to even."""
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    a = abs(x)
    exponent = int(mp.floor(mp.log(a, 2)))
    quantum = mp.mpf(2) ** max(exponent - 23, -149)
    rounded = mp.nint(a / quantum) * quantum
    # The rounded number is now exactly representable in binary32 (and binary64).
    return f32(sign * rounded)


def generate(dps):
    mp.mp.dps = dps
    cases = {}
    def add(x, v, kind):
        x, v = f32(x), f32(v)
        if math.isfinite(x) and x > 0 and math.isfinite(v):
            cases.setdefault((bits(x), bits(v)), kind)

    for e in range(-126, 127, 3):
        x = math.ldexp(1., e)
        for scale in [0., .125, -.125, 1., -1., 8., -8.]:
            try:
                add(x, x * scale, 'proportional_cotangent')
            except OverflowError:
                pass
    for x in [1e-38, 1e-35, 1e-30, 1e-27, 1e-10, .25, 1., 4., 1e10, 1e27, 1e30, 1e32, 1e35, 1e38]:
        for scale in [0., 1., -1.]:
            add(x, f32(x) * scale, 'decimal_proportional')
    for x in [.0001, .001, .01, .1, .5, 1., 2., 10., 100., 1000., 10000.]:
        for v in [0., 1., -1., .125, -8.]:
            add(x, v, 'ordinary_control')
    rng = random.Random(20260917)
    # Positive normal primals and normal, signed cotangents; retain only finite
    # nonzero-normal expected derivatives, avoiding undefined/overflow targets.
    for _ in range(2400):
        x = value(rng.randrange(0x00800000, 0x7f7fffff))
        v = value(rng.randrange(0x00800000, 0x7f7fffff))
        if rng.randrange(2):
            v = -v
        expected = -exact(v) / (2 * exact(x) ** mp.mpf('1.5'))
        if mp.mpf(2) ** -126 <= abs(expected) <= mp.mpf(2) ** 126:
            add(x, v, 'random_normal_finite_target')
    rows = []
    for i, ((xb, vb), kind) in enumerate(cases.items()):
        x, v = value(xb), value(vb)
        xx, vv = exact(x), exact(v)
        derivative = -vv / (2 * xx ** mp.mpf('1.5'))
        forward = 1 / mp.sqrt(xx)
        rows.append(dict(id=i, x_bits=xb, v_bits=vb, kind=kind,
                         expected=rounded_reference(derivative), forward=rounded_reference(forward)))
    return rows


def prepare():
    rows = generate(100)
    assert rows == generate(150), 'Oracle rounding changed with precision'
    (ROOT / 'inputs.txt').write_text(''.join(f"{r['id']} {r['x_bits']} {r['v_bits']}\n" for r in rows))
    (ROOT / 'oracle.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def run(executable, label):
    rows = json.loads((ROOT / 'oracle.json').read_text())
    outdir = ROOT / 'results'
    outdir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    summaries = {}
    base = None
    for layout in ['flat', 'row', 'column']:
        p = subprocess.run([str(executable), str(ROOT / 'inputs.txt'), layout],
                           text=True, capture_output=True, check=True, env=env)
        (outdir / f'{label}-{layout}.csv').write_text(p.stdout)
        (outdir / f'{label}-{layout}.stderr').write_text(p.stderr)
        observed = list(csv.DictReader(p.stdout.splitlines()))
        assert len(observed) == len(rows)
        failures = []
        controls = 0
        for r, got in zip(rows, observed):
            assert int(got['id']) == r['id']
            assert int(got['x_bits']) == r['x_bits'] and int(got['v_bits']) == r['v_bits']
            wrong = []
            for field in ['jvp', 'vjp', 'forward']:
                expected = r['forward' if field == 'forward' else 'expected']
                actual = value(got[field + '_bits'])
                tol = 2e-6 * abs(expected) + 2 * 2.**-149
                if not math.isfinite(actual) or abs(actual - expected) > tol:
                    wrong.append(field)
            if wrong:
                failures.append(dict(id=r['id'], kind=r['kind'], x=value(r['x_bits']),
                                     v=value(r['v_bits']), expected=r['expected'],
                                     jvp=str(value(got['jvp_bits'])), vjp=str(value(got['vjp_bits'])),
                                     failed_fields=wrong))
            if r['kind'] == 'ordinary_control':
                controls += 1
        if base is None:
            base = p.stdout
        summaries[layout] = dict(rows=len(rows), failed_rows=len(failures),
                                 jvp_failures=sum('jvp' in x['failed_fields'] for x in failures),
                                 vjp_failures=sum('vjp' in x['failed_fields'] for x in failures),
                                 forward_failures=sum('forward' in x['failed_fields'] for x in failures),
                                 ordinary_controls=controls,
                                 ordinary_control_failures=sum(x['kind']=='ordinary_control' for x in failures),
                                 same_as_flat=p.stdout == base)
        (outdir / f'{label}-{layout}-failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    report = dict(label=label, binary_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                  source_pin='59d600b5e64c238427d0f8d897ab7c682ef4d3d2',
                  device='CPU', dtype='float32', oracle_precision=[100,150],
                  layouts=summaries)
    (outdir / f'{label}-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        print('Prepared', len(prepare()), 'exact-binary-input oracle rows')
    elif sys.argv[1] == 'run':
        run(Path(sys.argv[2]).resolve(), sys.argv[3])
