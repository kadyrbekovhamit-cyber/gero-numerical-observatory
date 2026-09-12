"""Run the actual native executable and save machine-readable observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    env[key] = "1"
r = subprocess.run([str(a.binary.resolve())], env=env, text=True,
                   capture_output=True, timeout=60)
(a.output / "run.log").write_text(r.stdout + r.stderr)
rows = [json.loads(s[5:]) for s in r.stdout.splitlines() if s.startswith("CASE ")]
summary = [json.loads(s[8:]) for s in r.stdout.splitlines() if s.startswith("SUMMARY ")]
if len(summary) != 1 or summary[0]["checks"] != len(rows):
    raise SystemExit("Incomplete native execution; inspect run.log")
(a.output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
(a.output / "summary.json").write_text(json.dumps({"returncode": r.returncode, **summary[0]}, indent=2) + "\n")
print(json.dumps(summary[0]))
raise SystemExit(r.returncode)
