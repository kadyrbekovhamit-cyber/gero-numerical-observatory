"""Sequential isolated native prototype; reuse the existing CPU archive."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT/"higher-order"
EXP_FIX = ROOT/"known-float64-exp"
REPO = Path(os.environ["MLX_SOURCE_ROOT"]).resolve()
BUILD = Path(os.environ["MLX_CPU_BUILD"]).resolve()
ARCHIVE = BUILD/"mlx-build/libmlx.a"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
records = []


def patch(source):
    start = source.index("  } else if (reduce_type_ == Scan::LogAddExp) {",
                         source.index("std::vector<array> Scan::vjp("))
    end = source.index("  } else if (reduce_type_ == Scan::Prod) {", start)
    source = source[:start] + """  } else if (reduce_type_ == Scan::LogAddExp) {
    return {stable_log_scan_vjp(
        primals[0], cotangents[0], axis_, reverse_, inclusive_, stream())};
""" + source[end:]
    return source.replace("std::vector<array> Scan::vjp(",
        (ROOT/"stable_scan_vjp.cpp.inc").read_text() + "\nstd::vector<array> Scan::vjp(", 1)


def method(source):
    start = source.index("std::vector<array> Scan::vjp(")
    return source[start:source.index("\n}", start)+2]


def run(command, label, expected=None):
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run(command, cwd=BUILD, text=True, capture_output=True, timeout=60)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT/(label+".log")).write_text(result.stdout+result.stderr)
    records.append(dict(label=label, command=command, returncode=result.returncode,
        wall_seconds=time.monotonic()-started,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT/"build-results.json").write_text(json.dumps(records, indent=2)+"\n")
    cases = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("CASE ")]
    if cases:
        (ROOT/(label+".json")).write_text(json.dumps(cases, indent=2, allow_nan=False)+"\n")
    summary = [line for line in result.stdout.splitlines() if line.startswith(("SUMMARY", "EVIDENCE"))]
    print(label, result.returncode, *summary, sep="\n", flush=True)
    if expected is not None and result.returncode != expected:
        print((result.stdout+result.stderr)[-5000:], flush=True)
        raise SystemExit(1)
    return result.returncode


def main():
    for name in ("build-results.json", "run-after.json", "run-after.log"):
        saved = ROOT/("initial-"+name)
        if (ROOT/name).exists() and not saved.exists():
            shutil.copy2(ROOT/name, saved)
    source = subprocess.run(["git", "show", "HEAD:mlx/primitives.cpp"], cwd=REPO,
        text=True, capture_output=True, check=True).stdout
    # The pristine object was built from this exact saved translation unit.
    assert source == (PREVIOUS/"baseline-primitives.cpp").read_text()
    upstream = (ROOT/"upstream-mlx_primitives.cpp").read_text() if (
        ROOT/"upstream-mlx_primitives.cpp").exists() else (
        PREVIOUS/"upstream-mlx_primitives.cpp").read_text()
    (ROOT/"baseline-primitives.cpp").write_text(source)
    (ROOT/"patched-baseline-primitives.cpp").write_text(patch(source))
    (ROOT/"stable-logcumsumexp-vjp-prototype.patch").write_text("".join(
        difflib.unified_diff(upstream.splitlines(True), patch(upstream).splitlines(True),
                            fromfile="a/mlx/primitives.cpp", tofile="b/mlx/primitives.cpp")))
    metadata = dict(baseline_revision=subprocess.run(["git", "rev-parse", "HEAD"],
        cwd=REPO, text=True, capture_output=True, check=True).stdout.strip(),
        baseline_sha256=hashlib.sha256(source.encode()).hexdigest(),
        upstream_scan_vjp_identical=method(source)==method(upstream),
        archive=str(ARCHIVE), archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        baseline_object=str(PREVIOUS/"before.o"),
        baseline_object_sha256=hashlib.sha256((PREVIOUS/"before.o").read_bytes()).hexdigest(),
        scope="Small CPU-only archive with pristine/patched translation-unit override; not a full current-main build.")
    (ROOT/"native-source-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    entry = next(item for item in json.loads((BUILD/"compile_commands.json").read_text())
                 if item["file"].endswith("/mlx/primitives.cpp"))
    args = shlex.split(entry["command"])
    base = ["-O0" if arg == "-O3" else arg for arg in args[:args.index("-o")]]
    test_object = ROOT/"native_regression.o"
    run(base+["-o", str(test_object), "-c", str(ROOT/"native_regression.cpp")], "compile-test", 0)
    after_object = ROOT/"after.o"
    run(base+["-o", str(after_object), "-c", str(ROOT/"patched-baseline-primitives.cpp")],
        "compile-prototype", 0)
    stages = (
        ("before", PREVIOUS/"before.o", EXP_FIX/"before.o", 1),
        ("scan-only", after_object, EXP_FIX/"before.o", 1),
        ("exp-only", PREVIOUS/"before.o", EXP_FIX/"after.o", 1),
        ("combined", after_object, EXP_FIX/"after.o", 0),
    )
    for label, obj, unary, expected in stages:
        binary = ROOT/("native-"+label)
        run(["/usr/bin/c++", str(test_object), str(obj), str(unary), str(ARCHIVE),
             "-framework", "Accelerate", "-o", str(binary)], "link-"+label, 0)
        run([str(binary)], "run-"+label, expected)
    # The changed VJP also needs the previous zero-cotangent higher-order checks.
    binary = ROOT/"higher-order-after"
    run(["/usr/bin/c++", str(PREVIOUS/"native_regression.o"), str(after_object),
         str(EXP_FIX/"after.o"), str(ARCHIVE),
         "-framework", "Accelerate", "-o", str(binary)], "link-higher-order", 0)
    run([str(binary)], "run-higher-order", 0)


if __name__ == "__main__":
    main()
