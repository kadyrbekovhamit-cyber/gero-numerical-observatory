"""Check saved evidence and apply the patch only in a temporary directory."""
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
def check(label, condition):
    checks.append(dict(label=label, passed=bool(condition)))
def read(name):
    return json.loads((ROOT/name).read_text())
def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

before, after = read("run-before.json"), read("run-after.json")
check("518 cases on each side", len(before) == len(after) == 518)
check("188 baseline failures", sum(not x["passed"] for x in before) == 188)
check("no failures after", all(x["passed"] for x in after))
check("identical cases and references", all(
    (a["label"], a["dtype"], a["expected"], a["tolerance"]) ==
    (b["label"], b["dtype"], b["expected"], b["tolerance"])
    for a, b in zip(before, after)))
check("all references finite", all(finite(v) for x in after for v in x["expected"]))
check("all patched outputs finite", all(finite(v) for x in after for v in x["actual"]))
check("dtype coverage", collections.Counter(x["dtype"] for x in after) ==
      {"float32": 195, "float64": 195, "float16": 64, "bfloat16": 64})
check("recomputed patched pass flags", all(
    len(x["actual"]) == len(x["expected"]) and all(
        abs(a) <= x["tolerance"] if e == 0 else abs(a/e-1) <= x["tolerance"]
        for a, e in zip(x["actual"], x["expected"])) for x in after))
wheel = read("wheel-reproduction.json")
check("wheel CPU version", wheel["version"] == "0.32.2" and wheel["device"] == "Device(cpu, 0)")
check("wheel dtype coverage", len({x["dtype"] for x in wheel["cases"]}) == 4)
check("wheel four composite counterexamples", len(wheel["compositions"]) == 4 and all(
    x["gradient"] == 0 and x["hessian"] == 0 and x["third"] == "NaN" and
    abs(x["finite_difference"]-1) < 5e-4 for x in wheel["compositions"]))
meta = read("source-metadata.json")
check("public fetch without errors", not meta["errors"])
for record in meta["files"]:
    saved = ROOT / ("upstream-" + record["path"].replace("/", "_"))
    check("hash "+record["path"], hashlib.sha256(saved.read_bytes()).hexdigest() == record["sha256"])
native = read("native-source-metadata.json")
check("four methods match public main", len(native["target_methods_identical"]) == 4 and
      all(native["target_methods_identical"].values()))
for dep in native["dependencies"]:
    check("dependency "+Path(dep["path"]).name,
          hashlib.sha256(Path(dep["path"]).read_bytes()).hexdigest() == dep["sha256"])
source = (ROOT/"upstream-mlx_primitives.cpp").read_text()
patched = patch(source)
for cls in ("ArcSinh", "ArcCosh"):
    old, new = method(source, cls, "jvp"), method(patched, cls, "jvp")
    check(cls+" original complex JVP body preserved",
          old[old.index("  array one"):] == new[new.index("  array one"):])
    check(cls+" real-only guard", "if (!issubdtype(primals[0].dtype(), complexfloating))" in new)
    check(cls+" VJP unchanged", method(source, cls, "vjp") == method(patched, cls, "vjp"))
with tempfile.TemporaryDirectory(prefix="mlx-hyperbolic-patch-") as temp:
    target = Path(temp)/"mlx/primitives.cpp"
    target.parent.mkdir()
    target.write_text(source)
    applied = subprocess.run(["patch", "-p1", "-i", str(ROOT/"inverse-hyperbolic-real.patch")],
        cwd=temp, text=True, capture_output=True, timeout=5)
    check("patch applies to saved current main", applied.returncode == 0)
    check("applied patch exact content", target.read_text() == patched)
queries = read("duplicate-search.json")
check("four complete duplicate searches", len(queries) == 4 and all(
    "result" in x and not x["result"]["incomplete_results"] and
    len(x["result"]["items"]) == x["result"]["total_count"] for x in queries))
build = read("build-results.json")
check("build return codes", [x["returncode"] for x in build] == [0, 0, 0, 1, 0, 0])
check("initial harness evidence retained",
      len(read("initial-harness-attempt/run-after.json")) == 390 and
      sum(not x["passed"] for x in read("initial-harness-attempt/run-after.json")) == 31)
out = dict(checks=checks, failures=sum(not x["passed"] for x in checks),
           cpu_seconds=time.process_time()-start)
(ROOT/"artifact-check-results.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
raise SystemExit(bool(out["failures"]))
