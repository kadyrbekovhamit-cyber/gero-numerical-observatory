"""One translation unit at a time; no repository edits or full build."""
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
REPO = Path(os.environ["MLX_SOURCE_ROOT"]).resolve()
BUILD = Path(os.environ["MLX_CPU_BUILD"]).resolve()
ARCHIVE = BUILD/"mlx-build/libmlx.a"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
records = []
def run(cmd, label, expected=0):
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    r = subprocess.run(cmd, cwd=BUILD, text=True, capture_output=True, timeout=60)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT/(label+".log")).write_text(r.stdout+r.stderr)
    records.append(dict(label=label, command=cmd, returncode=r.returncode,
        wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT/"build-results.json").write_text(json.dumps(records, indent=2)+"\n")
    rows = [json.loads(x[5:]) for x in r.stdout.splitlines() if x.startswith("CASE ")]
    if rows: (ROOT/(label+".json")).write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    print(label, r.returncode, *[x for x in r.stdout.splitlines() if x.startswith("SUMMARY")], flush=True)
    if r.returncode != expected:
        print((r.stdout+r.stderr)[-4000:]); raise SystemExit(1)
def patch(source):
    old = """  if constexpr (is_complex<T>) {
    return Simd<T, 1>{std::exp(in.value)};
  } else {"""
    new = """  if constexpr (is_complex<T>) {
    return Simd<T, 1>{std::exp(in.value)};
  } else if constexpr (std::is_same_v<T, double>) {
    Simd<T, N> out;
    for (int i = 0; i < N; ++i) {
      out[i] = std::exp(in[i]);
    }
    return out;
  } else {"""
    assert source.count(old) == 1
    return source.replace(old, new)
def main():
    path = "mlx/backend/cpu/simd/math.h"
    source = subprocess.run(["git", "show", "HEAD:"+path], cwd=REPO, text=True,
                            capture_output=True, check=True).stdout
    assert source == (REPO/path).read_text()
    (ROOT/"baseline-math.h").write_text(source)
    header = ROOT/"overlay"/path
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(patch(source))
    upstream = ROOT/"upstream-mlx_backend_cpu_simd_math.h"
    patch_base = upstream.read_text() if upstream.exists() else source
    (ROOT/"float64-exp.patch").write_text("".join(difflib.unified_diff(
        patch_base.splitlines(True), patch(patch_base).splitlines(True),
        fromfile="a/"+path, tofile="b/"+path)))
    db = json.loads((BUILD/"compile_commands.json").read_text())
    entry = next(x for x in db if x["file"].endswith("/mlx/backend/cpu/unary.cpp"))
    args = shlex.split(entry["command"])
    base = ["-O0" if a == "-O3" else a for a in args[:args.index("-o")]]
    run(base+["-o", str(ROOT/"test.o"), "-c", str(ROOT/"native_regression.cpp")], "compile-test")
    for label in ("before", "after"):
        cmd = base[:1]+(["-I"+str(ROOT/"overlay")] if label == "after" else [])+base[1:]
        run(cmd+["-o", str(ROOT/(label+".o")), "-c", entry["file"]], "compile-"+label)
        binary = ROOT/("native-"+label)
        run(["/usr/bin/c++", str(ROOT/"test.o"), str(ROOT/(label+".o")), str(ARCHIVE),
             "-framework", "Accelerate", "-o", str(binary)], "link-"+label)
        run([str(binary)], "run-"+label, 1 if label == "before" else 0)
    meta = dict(baseline_revision=subprocess.run(["git","rev-parse","HEAD"], cwd=REPO,
        capture_output=True,text=True,check=True).stdout.strip(),
        header_sha256=hashlib.sha256(source.encode()).hexdigest(),
        archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        scope="CPU archive with isolated unary.cpp override; other TUs using the header are not rebuilt.")
    (ROOT/"native-source-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__ == "__main__": main()
