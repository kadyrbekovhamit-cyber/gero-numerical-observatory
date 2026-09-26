"""Run the frozen v0 accuracy comparison and write a JSON report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mpmath as mp

from lab.black_scholes import naive_prices, stable_detailed, stable_prices
from lab.reference import prices as reference_prices


MONEYNESS = (
    -100.0,
    -50.0,
    -20.0,
    -10.0,
    -2.0,
    -0.1,
    0.0,
    0.1,
    2.0,
    10.0,
    20.0,
    50.0,
    100.0,
)
TOTAL_VOL = (1e-10, 1e-8, 1e-5, 1e-3, 0.05, 0.2, 1.0, 3.0, 10.0)
MIN_SUBNORMAL = mp.mpf(float.fromhex("0x0.0000000000001p-1022"))


def _error(value: float, reference: mp.mpf) -> dict[str, float | None]:
    absolute = float(abs(mp.mpf(value) - reference))
    relative = None if reference == 0 else float(abs(mp.mpf(value) / reference - 1))
    return {"absolute": absolute, "relative": relative}


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    wins = {"stable": 0, "naive": 0, "tie": 0}
    representable_wins = {"stable": 0, "naive": 0, "tie": 0}
    violations = {"stable": 0, "naive": 0}
    representability = {"float64_representable": 0, "below_float64": 0}
    log_errors: list[float] = []
    log_ulp_errors: list[float] = []
    below_float_log_errors: list[float] = []
    below_float_log_ulp_errors: list[float] = []

    # Use T=1, r=q=0 and encode log-forward-moneyness in spot.  This isolates
    # the normalized two-dimensional kernel without a market-data claim.
    for m in MONEYNESS:
        strike = 1.0
        spot = math.exp(m)
        for total_vol in TOTAL_VOL:
            ref_call, ref_put = reference_prices(
                spot, strike, 1.0, 0.0, 0.0, total_vol, dps=100
            )
            naive = naive_prices(spot, strike, 1.0, 0.0, 0.0, total_vol)
            stable = stable_prices(spot, strike, 1.0, 0.0, 0.0, total_vol)
            detailed = stable_detailed(
                spot, strike, 1.0, 0.0, 0.0, total_vol
            )
            option = "call" if m <= 0 else "put"
            ref = ref_call if option == "call" else ref_put
            representable = ref == 0 or abs(ref) >= MIN_SUBNORMAL
            representability[
                "float64_representable" if representable else "below_float64"
            ] += 1
            naive_value = getattr(naive, option)
            stable_value = getattr(stable, option)
            reference_log = float(mp.log(ref)) if ref > 0 else -math.inf
            log_error = (
                abs(detailed.log_otm_value - reference_log)
                if math.isfinite(reference_log)
                and math.isfinite(detailed.log_otm_value)
                else None
            )
            if log_error is not None:
                log_errors.append(log_error)
                log_ulp_errors.append(log_error / math.ulp(reference_log))
                if not representable:
                    below_float_log_errors.append(log_error)
                    below_float_log_ulp_errors.append(
                        log_error / math.ulp(reference_log)
                    )
            naive_error = _error(naive_value, ref)
            stable_error = _error(stable_value, ref)
            n_abs = float(naive_error["absolute"])
            s_abs = float(stable_error["absolute"])
            if s_abs < n_abs:
                wins["stable"] += 1
                if representable:
                    representable_wins["stable"] += 1
            elif n_abs < s_abs:
                wins["naive"] += 1
                if representable:
                    representable_wins["naive"] += 1
            else:
                wins["tie"] += 1
                if representable:
                    representable_wins["tie"] += 1
            for name, pair in (("naive", naive), ("stable", stable)):
                if pair.call < 0 or pair.put < 0 or not all(
                    math.isfinite(x) for x in (pair.call, pair.put, pair.parity)
                ):
                    violations[name] += 1
            rows.append(
                {
                    "log_forward_moneyness": m,
                    "total_volatility": total_vol,
                    "otm_option": option,
                    "reference": mp.nstr(ref, 25),
                    "reference_float64_representable": representable,
                    "naive": naive_value,
                    "stable": stable_value,
                    "stable_log_otm_value": detailed.log_otm_value,
                    "reference_log_otm_value": reference_log,
                    "log_absolute_error": log_error,
                    "naive_error": naive_error,
                    "stable_error": stable_error,
                }
            )

    return {
        "contract": {
            "precision_digits": 100,
            "spot": "exp(log_forward_moneyness)",
            "strike": 1.0,
            "time": 1.0,
            "rate": 0.0,
            "dividend_yield": 0.0,
            "evaluated_leg": "OTM call for m<=0, OTM put for m>0",
        },
        "cases": len(rows),
        "absolute_error_wins": wins,
        "representable_absolute_error_wins": representable_wins,
        "reference_representability": representability,
        "log_contract": {
            "finite_comparisons": len(log_errors),
            "max_absolute_log_error": max(log_errors, default=None),
            "max_log_error_in_float64_ulps": max(log_ulp_errors, default=None),
            "below_float64_comparisons": len(below_float_log_errors),
            "below_float64_max_absolute_log_error": max(
                below_float_log_errors, default=None
            ),
            "below_float64_max_log_error_in_float64_ulps": max(
                below_float_log_ulp_errors, default=None
            ),
        },
        "finite_or_nonnegative_violations": violations,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
