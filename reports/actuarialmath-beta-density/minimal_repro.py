"""Actual Beta API calls; expected values follow from exact rational survival."""
from pathlib import Path
from fractions import Fraction
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', help='Optional extracted source or wheel root containing src/')
args = parser.parse_args()
if args.source:
    sys.path.insert(0, str(Path(args.source).resolve() / 'src'))
import actuarialmath
from actuarialmath import Beta

if args.source:
    assert Path(actuarialmath.__file__).resolve().is_relative_to(Path(args.source).resolve())
life = Beta(omega=70, alpha=2).set_interest(i=0)
benefit, term = 100000, 20
survival = Fraction(1, 2) ** 2
print(json.dumps({
    'actual': {
        'survival_to_20': life.p_r(30, t=term),
        'density_at_20': life.f_r(30, t=term),
        'whole_life_value': life.whole_life_insurance(30, b=benefit, discrete=False),
        'term_value': life.term_insurance(30, t=term, b=benefit, discrete=False),
        'equivalent_direct_value': life.A_x(30, t=term, benefit=lambda age, t: benefit, discrete=False),
    },
    'expected': {
        'survival_to_20': float(survival),
        'density_at_20': float(Fraction(2, 40) * Fraction(1, 2)),
        'whole_life_value': benefit,
        'term_value': float(benefit * (1 - survival)),
        'equivalent_direct_value': float(benefit * (1 - survival)),
    },
}, indent=2))
