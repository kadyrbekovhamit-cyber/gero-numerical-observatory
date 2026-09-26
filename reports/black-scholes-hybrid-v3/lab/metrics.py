"""Error measurement; preserve high precision until JSON serialization."""

import math
import struct
import sys

import mpmath as mp


def text(x):
    return mp.nstr(x, 35)


def round_binary64(value):
    """Nearest-even conversion with an explicit subnormal path.

    Rounding a full mantissa to 53 bits and subsequently underflowing can
    double-round near subnormal boundaries. Round in subnormal units first.
    """
    with mp.workdps(180):
        if not mp.isfinite(value) or abs(value) >= mp.mpf(sys.float_info.min):
            return float(value)
        units = abs(value) * mp.power(2, 1074)
        integer = int(mp.floor(units))
        fraction = units - integer
        if fraction > mp.mpf("0.5") or (fraction == mp.mpf("0.5") and integer % 2):
            integer += 1
        rounded = math.ldexp(float(integer), -1074)
        return -rounded if value < 0 else rounded


def rounding_class(reference):
    if reference == 0:
        return "exact_zero"
    rounded = round_binary64(reference)
    if not math.isfinite(rounded):
        return "overflow"
    if rounded == 0:
        return "rounds_to_zero"
    if abs(rounded) < sys.float_info.min:
        return "subnormal"
    return "normal"


def float_rank(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)


def price_error(value, reference, dps=180):
    with mp.workdps(dps):
        rounded = round_binary64(reference)
        if not math.isfinite(value):
            return {"finite": False, "value": str(value), "absolute": "inf",
                    "relative": "inf", "real_ulp_error": "inf", "rounded_ulp_distance": None}
        delta = abs(mp.mpf(value) - reference)
        return {
            "finite": True, "value": value,
            "absolute": text(delta),
            "relative": text(delta / abs(reference)) if reference else None,
            "real_ulp_error": text(delta / mp.mpf(math.ulp(rounded))) if math.isfinite(rounded) else None,
            "rounded_ulp_distance": abs(float_rank(value) - float_rank(rounded)) if math.isfinite(rounded) else None,
        }


def log_error(value, reference_log, dps=180):
    with mp.workdps(dps):
        if not math.isfinite(value):
            return {"finite": False, "value": str(value), "absolute": "inf", "real_ulp_error": "inf"}
        delta = abs(mp.mpf(value) - reference_log)
        return {"finite": True, "value": value, "absolute": text(delta),
                "real_ulp_error": text(delta / mp.mpf(math.ulp(float(reference_log))))}
