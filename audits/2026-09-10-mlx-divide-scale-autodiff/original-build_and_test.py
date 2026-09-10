"""Sequential CPU-only translation-unit test; no working-tree edits."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time
ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "mlx-logcumsumexp-hessian-2026-09-10"
REPO = ROOT.parent.parent / "audit-targets/current-stack-2026-09-07/mlx"
BUILD = ROOT.parent / "current-stack-2026-09-07-round3/build-native"
ARCHIVE = BUILD / "mlx-build/libmlx.a"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
records = []
def method(source, name):
    begin = source.index("std::vector<array> Divide::" + name + "(")
    return source[begin:source.index("\n}", begin)+2]
def patch(source):
    old_vjp, old_jvp = method(source, "vjp"), method(source, "jvp")
    needle = "    } else {\n"
    vjp_guard = """    } else if (!issubdtype(primals[0].dtype(), complexfloating)) {
      vjps.push_back(real_divide_denominator_partial(
          cotangents[0], primals[0], primals[1], stream()));
    } else {
"""
    jvp_guard = """    } else if (!issubdtype(primals[0].dtype(), complexfloating)) {
      return real_divide_denominator_partial(
          tangents[i], primals[0], primals[1], stream());
    } else {
"""
    assert old_vjp.count(needle) == old_jvp.count(needle) == 1
    vjp = old_vjp.replace(needle, vjp_guard, 1)
    jvp = old_jvp.replace(needle, jvp_guard, 1)
    source = source.replace(method(source, "vjp"), vjp, 1)
    source = source.replace(method(source, "jvp"), jvp, 1)
    return source.replace("std::vector<array> Divide::vjp(",
        (ROOT/"weighted_partial.cpp.inc").read_text()+"\nstd::vector<array> Divide::vjp(", 1)
def run(cmd, label, expected=0):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    r = subprocess.run(cmd, cwd=BUILD, text=True, capture_output=True, timeout=60)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT/(label+".log")).write_text(r.stdout+r.stderr)
    records.append(dict(label=label, command=cmd, returncode=r.returncode,
        wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT/"build-results.json").write_text(json.dumps(records, indent=2)+"\n")
    rows = [json.loads(line[5:]) for line in r.stdout.splitlines() if line.startswith("CASE ")]
    if rows:
        (ROOT/(label+".json")).write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    print(label, r.returncode,
          *[line for line in r.stdout.splitlines() if line.startswith("SUMMARY")], flush=True)
    if r.returncode != expected:
        print((r.stdout+r.stderr)[-5000:], flush=True)
        raise SystemExit(1)
def main():
    baseline = subprocess.run(["git", "show", "HEAD:mlx/primitives.cpp"],
        cwd=REPO, text=True, capture_output=True, check=True).stdout
    assert baseline == (PREVIOUS/"baseline-primitives.cpp").read_text()
    source = (ROOT/"upstream-mlx_primitives.cpp").read_text()
    (ROOT/"baseline-primitives.cpp").write_text(baseline)
    (ROOT/"patched-baseline-primitives.cpp").write_text(patch(baseline))
    (ROOT/"divide-scale-autodiff.patch").write_text("".join(difflib.unified_diff(
        source.splitlines(True), patch(source).splitlines(True),
        fromfile="a/mlx/primitives.cpp", tofile="b/mlx/primitives.cpp")))
    db = json.loads((BUILD/"compile_commands.json").read_text())
    entry = next(x for x in db if x["file"].endswith("/mlx/primitives.cpp"))
    args = shlex.split(entry["command"])
    base = ["-O0" if x == "-O3" else x for x in args[:args.index("-o")]]
    run(base+["-o", str(ROOT/"test.o"), "-c", str(ROOT/"native_regression.cpp")], "compile-test")
    run(base+["-o", str(ROOT/"after.o"), "-c", str(ROOT/"patched-baseline-primitives.cpp")], "compile-patch")
    for label, obj, expected in (("before", PREVIOUS/"before.o", 1), ("after", ROOT/"after.o", 0)):
        binary = ROOT/("native-"+label)
        run(["/usr/bin/c++", str(ROOT/"test.o"), str(obj), str(ARCHIVE),
             "-framework", "Accelerate", "-o", str(binary)], "link-"+label)
        run([str(binary)], "run-"+label, expected)
    meta = dict(
        baseline_revision=subprocess.run(["git", "rev-parse", "HEAD"],
            cwd=REPO, text=True, capture_output=True, check=True).stdout.strip(),
        target_methods_identical={name: method(source, name) == method(baseline, name)
                                  for name in ("jvp", "vjp")},
        dependencies=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in (ARCHIVE, PREVIOUS/"before.o")],
        scope="Primitives TU override before a prebuilt CPU archive; not a full current-main build.")
    (ROOT/"native-source-metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
if __name__ == "__main__":
    main()
