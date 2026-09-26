"""Fixed fixtures for v1. Binary64 values are stored as exact hex strings."""

import hashlib
import json
import math
import random
from pathlib import Path


FIELDS = ("s", "k", "t", "r", "q", "sigma")
MONEYNESS = (-100., -50., -20., -10., -2., -0.1, 0., 0.1, 2., 10., 20., 50., 100.)
TOTAL_VOL = (1e-10, 1e-8, 1e-5, 1e-3, 0.05, 0.2, 1., 3., 10.)


def build():
    cases = []

    def add(partition, family, s, k, sigma, t=1., r=0., q=0.):
        values = (s, k, t, r, q, sigma)
        cases.append({
            "id": f"{partition}-{len(cases):04d}",
            "partition": partition,
            "family": family,
            "binary64_hex": {name: float(x).hex() for name, x in zip(FIELDS, values)},
        })

    for m in MONEYNESS:
        for v in TOTAL_VOL:
            add("development", "legacy_normalized_grid", math.exp(m), 1., v)

    rng = random.Random(20260926)
    for _ in range(96):
        # Predefined mixture: central, cancellation, and far-tail regions.
        v = 10. ** rng.uniform(-10., 1.)
        m = rng.choice((-1., 1.)) * v * 10. ** rng.uniform(-3., 2.)
        m = min(300., max(-300., m))
        add("confirmation", "new_normalized_sample", math.exp(m), 1., v)

    for scale in (1e-200, 1e-50, 1., 1e50, 1e200):
        for ratio in (0.5, math.nextafter(1., 0.), 1., math.nextafter(1., math.inf), 2.):
            for v in (1e-8, 0.2, 2.):
                add("confirmation", "monetary_scale", ratio * scale, scale, v)

    for s, k in ((100., 100.), (80., 100.), (120., 100.)):
        for t in (1e-6, 0.25, 5.):
            for r, q in ((-0.02, 0.03), (0.1, -0.01)):
                add("confirmation", "bsm_nonzero_carry", s, k, 0.2, t, r, q)

    # Subnormal rounding boundary diagnostics: not used to tune the candidate.
    for m in (-37., -37.5, -38., -38.5, -39., 37., 37.5, 38., 38.5, 39.):
        add("confirmation", "underflow_boundary", math.exp(m), 1., 1.)

    return {"schema": 1, "seed": 20260926, "candidate": "frozen-v0",
            "input_contract": "exact binary64 values, all six BSM inputs",
            "cases": cases}


def arguments(case):
    return tuple(float.fromhex(case["binary64_hex"][name]) for name in FIELDS)


def freeze(path):
    path = Path(path)
    data = (json.dumps(build(), indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError("Refusing to overwrite different frozen fixtures")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    print(freeze("evidence/corpus-v1.json"))
