from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess

import mpmath as mp


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs.txt"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def make_inputs() -> list[dict]:
    rows: list[dict] = []
    idx = 0
    exponents = list(range(-120, 121, 8))
    for sx in (-1.0, 1.0):
        for xv in exponents:
            x = f32(sx * math.ldexp(1.0, xv))
            for sv in (-1.0, 1.0):
                for vv in exponents:
                    v = f32(sv * math.ldexp(1.0, vv))
                    rows.append({"id": idx, "x": x, "v": v})
                    idx += 1
    for x in (0.0, -0.0, 1.0, -1.0):
        rows.append({"id": idx, "x": f32(x), "v": 0.0})
        idx += 1
    for x in (-10.0, -3.0, -1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5, 3.0, 10.0):
        for v in (-3.0, -1.0, 0.25, 1.0, 3.0):
            rows.append({"id": idx, "x": f32(x), "v": f32(v)})
            idx += 1
    for x in (-math.inf, math.inf, math.nan):
        for v in (-1.0, 0.0, 1.0, math.inf, math.nan):
            rows.append({"id": idx, "x": f32(x), "v": f32(v)})
            idx += 1
    INPUTS.write_text("".join(f"{r['id']} {bits(r['x'])} {bits(r['v'])}\n" for r in rows))
    return rows


def expected(x: float, v: float) -> float:
    mp.mp.dps = 160
    if math.isnan(x) or math.isnan(v) or (math.isinf(x) and math.isinf(v)):
        return math.nan
    if math.isinf(x):
        return math.copysign(0.0, v)
    value = mp.mpf(float(v)) / (mp.mpf(1) + mp.mpf(float(x)) ** 2)
    return f32(float(value))


def read_csv(text: str) -> list[dict]:
    out = []
    for row in csv.DictReader(text.splitlines()):
        out.append({k: int(v) for k, v in row.items()})
    return out


def bad(actual: float, target: float) -> bool:
    if not math.isfinite(target):
        return False
    if not math.isfinite(actual):
        return True
    if target == 0.0:
        return abs(actual) > 1e-37
    return abs((actual - target) / target) > 8e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("label")
    args = parser.parse_args()
    source = make_inputs()
    by_id = {r["id"]: r for r in source}
    summaries = {}
    all_bits = {}
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    for layout in ("flat", "row", "column"):
        run = subprocess.run([str(args.binary), str(INPUTS), layout], check=True, text=True, capture_output=True)
        (evidence / f"{args.label}-{layout}.csv").write_text(run.stdout)
        rows = read_csv(run.stdout)
        failures = []
        for row in rows:
            src = by_id[row["id"]]
            target = expected(src["x"], src["v"])
            jvp = from_bits(row["jvp_bits"])
            vjp = from_bits(row["vjp_bits"])
            if bad(jvp, target) or bad(vjp, target):
                failures.append({
                    "id": row["id"], "x": src["x"], "v": src["v"],
                    "expected": target, "jvp": jvp, "vjp": vjp,
                    "jvp_bits": row["jvp_bits"], "vjp_bits": row["vjp_bits"],
                })
            all_bits.setdefault(row["id"], {})[layout] = (row["jvp_bits"], row["vjp_bits"])
        (evidence / f"{args.label}-{layout}-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        summaries[layout] = {"rows": len(rows), "failed_rows": len(failures)}
    layout_disagreements = sum(1 for layouts in all_bits.values() if len(set(layouts.values())) != 1)
    result = {
        "label": args.label,
        "source_pin": "59d600b5e64c238427d0f8d897ab7c682ef4d3d2",
        "operation": "arctan JVP and VJP",
        "oracle": "160-decimal evaluation of v/(1+x^2), rounded to float32",
        "summaries": summaries,
        "layout_disagreements": layout_disagreements,
        "gpu": False,
    }
    (evidence / f"{args.label}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
