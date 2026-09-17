"""Execute FinancePy's real BondZero public methods on a one-year bond."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / ("cache-" + variant))
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache-mpl")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache-xdg")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / variant))

from financepy.products.bonds.bond_zero import BondZero
assert Path(sys.modules[BondZero.__module__].__file__).is_relative_to(ROOT / variant)
from financepy.utils.date import Date

settle = Date(1, 1, 2025)
bond = BondZero(settle, Date(1, 1, 2026), 95.0)
y = 0.05
price = float(bond.dirty_price_from_ytm(settle, y))
dv01 = float(bond.dv01(settle, y))
duration = float(bond.modified_duration(settle, y))
print(json.dumps({
    "variant": variant,
    "loaded_module": sys.modules[BondZero.__module__].__file__,
    "price_per_100": price,
    "dv01_per_100": dv01,
    "modified_duration": duration,
    "analytic_modified_duration": 1 / 1.05,
    "exact_one_bp_forward_secant": 1 / 1.0501,
    "duration_implied_one_bp_price_change": duration * price * 0.0001,
    "actual_one_bp_price_change": dv01,
    "ratio_of_implied_to_actual": duration * price * 0.0001 / dv01,
}, indent=2))
