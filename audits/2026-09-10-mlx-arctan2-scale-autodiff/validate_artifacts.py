"""Validate recorded evidence without executing the numerical suite again."""
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from build_and_test import method, patch
ROOT = Path(__file__).resolve().parent
checks = []
start = time.process_time()
def check(label, ok):
    checks.append(dict(label=label, passed=bool(ok)))
def read(name):
    return json.loads((ROOT/name).read_text())
def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)
before, after = read("run-before.json"), read("run-after.json")
check("1368 cases each side", len(before) == len(after) == 1368)
check("772 baseline failures", sum(not x["passed"] for x in before) == 772)
check("all final cases pass", all(x["passed"] for x in after))
check("same cases references tolerances", all(
    (a["label"], a["dtype"], a["expected"], a["tolerance"]) ==
    (b["label"], b["dtype"], b["expected"], b["tolerance"])
    for a, b in zip(before, after)))
check("all references finite", all(finite(v) for x in after for v in x["expected"]))
check("all final outputs finite", all(finite(v) for x in after for v in x["actual"]))
check("dtype coverage", collections.Counter(x["dtype"] for x in after) ==
      {"float16": 294, "bfloat16": 300, "float32": 387, "float64": 387})
check("recompute final comparisons", all(
    len(x["actual"]) == len(x["expected"]) and all(
        abs(a) <= x["tolerance"] if e == 0 else abs(a/e-1) <= x["tolerance"]
        for a, e in zip(x["actual"], x["expected"])) for x in after))
check("one-sided and joint JVP present", all(any(x["label"].endswith(suffix)
    for x in after) for suffix in ("/JVP-y", "/JVP-x", "/JVP-both")))
check("mixed Hessian checked", sum(x["label"].startswith("mixed-Hessian") for x in after) == 6)
check("radial invariant checked", sum(x["label"].endswith("/radial-JVP") for x in after) == 84)
check("angular invariant checked", sum(x["label"].endswith("/angular-JVP") for x in after) == 84)
wheel = read("wheel-reproduction.json")
check("wheel CPU and version", wheel["version"] == "0.32.2" and wheel["device"] == "Device(cpu, 0)")
check("wheel four dtypes", len({x["dtype"] for x in wheel["cases"]}) == 4)
check("wheel first-order finite differences", len(wheel["compositions"]) == 6 and all(
    abs(x["finite_difference"]-.5) < 5e-4 for x in wheel["compositions"]))
check("wheel scale-dependent failure", all(
    x["gradient"] == ("Infinity" if x["scale"] < 1 else 0) and
    x["hessian"] == x["third"] == "NaN"
    for x in wheel["compositions"] if x["scale"] != 1))
meta = read("source-metadata.json")
check("source fetch no errors", not meta["errors"])
for record in meta["files"]:
    saved = ROOT/("upstream-"+record["path"].replace("/", "_"))
    check("source hash "+record["path"],
          hashlib.sha256(saved.read_bytes()).hexdigest() == record["sha256"])
native = read("native-source-metadata.json")
check("target methods identical to public main", len(native["target_methods_identical"]) == 2 and
      all(native["target_methods_identical"].values()))
check("archive matches recorded dependency", read("publication-rerun.json")["archive_sha256"] == native["dependencies"][0]["sha256"])
source = (ROOT/"upstream-mlx_primitives.cpp").read_text()
patched = patch(source)
restored = patched.replace((ROOT/"scaled_partials.cpp.inc").read_text()+"\n", "", 1)
for name in ("jvp", "vjp"):
    restored = restored.replace(method(restored, name), method(source, name), 1)
check("only target methods and helpers changed", restored == source)
with tempfile.TemporaryDirectory(prefix="mlx-atan2-patch-") as temp:
    target = Path(temp)/"mlx/primitives.cpp"
    target.parent.mkdir()
    target.write_text(source)
    result = subprocess.run(["patch", "-p1", "-i", str(ROOT/"arctan2-scale-autodiff.patch")],
        cwd=temp, capture_output=True, text=True, timeout=5)
    check("patch applies to saved current main", result.returncode == 0)
    check("applied patch content", target.read_text() == patched)
search = read("duplicate-search.json")
check("four complete duplicate searches", len(search) == 4 and all(
    "result" in x and not x["result"]["incomplete_results"] and
    len(x["result"]["items"]) == x["result"]["total_count"] for x in search))
check("11 unique public results", len({x["number"] for q in search for x in q["result"]["items"]}) == 11)
initial = read("initial-normalization/run-after.json")
check("initial 1350 series retained", len(initial) == 1350 and all(x["passed"] for x in initial))
initial_extended = read("initial-normalization/extended-run-after.json")
initial_before = read("initial-normalization/extended-run-before.json")
check("initial six real regressions retained", len(initial_extended) == 1362 and
    sum(not x["passed"] for x in initial_extended) == 6 and
    all(a["passed"] for a, b in zip(initial_before, initial_extended) if not b["passed"]))
second = read("weighted-two-order/extended-run-after.json")
check("second six remaining failures retained", len(second) == 1368 and
    sum(not x["passed"] for x in second) == 6 and
    all(x["label"].startswith("anisotropic/small-weight") for x in second if not x["passed"]))
check("expected build return codes", [x["returncode"] for x in read("build-results.json")] == [0,0,0,1,0,0])
out = dict(checks=checks, failures=sum(not x["passed"] for x in checks),
           cpu_seconds=time.process_time()-start)
(ROOT/"artifact-validation.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
raise SystemExit(bool(out["failures"]))
