"""Compile sequentially at O0 and reuse the existing CPU-only archive."""
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
REPO = Path(os.environ["MLX_SOURCE_ROOT"])
BUILD = Path(os.environ["MLX_CPU_BUILD"])
ARCHIVE = BUILD / "mlx-build/libmlx.a"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))


def method(source):
    start = source.index("std::vector<array> BlockMaskedMM::vjp(")
    return source[start:source.index("\n}", start)+2]


def patch(source):
    old = method(source)
    anchor = """      auto C = block_masked_mm(
          primals[0],
          primals[1],
          block_size_,
          primals[2],
          lhs_mask,
          rhs_mask,
          stream());"""
    assert old.count(anchor) == 1
    return source.replace(old, old.replace(anchor, anchor.replace("primals[2]", "std::nullopt")), 1)


records = []


def run(command, label):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    p = subprocess.run(command, cwd=BUILD, capture_output=True, text=True, timeout=60)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT / (label+".log")).write_text(p.stdout+p.stderr)
    records.append(dict(label=label, command=command, returncode=p.returncode,
                        wall_seconds=time.monotonic()-started,
                        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT / "build-results.json").write_text(json.dumps(records, indent=2)+"\n")
    print(label, p.returncode, round(records[-1]["wall_seconds"],4), flush=True)
    if label.startswith("run-"):
        print("\n".join(x for x in p.stdout.splitlines() if x.startswith(("SUMMARY","EVIDENCE"))), flush=True)
    elif p.returncode:
        print(p.stderr[-6000:], flush=True)
        raise SystemExit(p.returncode)
    return p


def main():
    baseline = subprocess.run(["git","show","ce916dbbcaa88e433b6fd1e60a17f766d49c27fe:mlx/primitives.cpp"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout
    (ROOT/"baseline-primitives.cpp").write_text(baseline)
    (ROOT/"patched-baseline-primitives.cpp").write_text(patch(baseline))
    meta_path = ROOT/"source-metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    upstream_path = ROOT/"upstream-mlx_primitives.cpp"
    source = upstream_path.read_text() if upstream_path.exists() else baseline
    (ROOT/"output-mask-vjp.patch").write_text("".join(difflib.unified_diff(
        source.splitlines(True), patch(source).splitlines(True),
        fromfile="a/mlx/primitives.cpp", tofile="b/mlx/primitives.cpp")))
    (ROOT/"BlockMaskedMM-baseline-to-upstream.diff").write_text("".join(difflib.unified_diff(
        method(baseline).splitlines(True),method(source).splitlines(True))))
    meta.update(baseline_revision=subprocess.run(["git","rev-parse","ce916dbbcaa88e433b6fd1e60a17f766d49c27fe"],cwd=REPO,
                    capture_output=True,text=True,check=True).stdout.strip(),
                archive=str(ARCHIVE),archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                build="Isolated pristine/patched baseline primitives.cpp linked before existing CPU archive; not a clean current-main build.",
                patch_scope="Only replace the output mask by nullopt in its own VJP branch.",
                upstream_method_identical=method(baseline)==method(source) if upstream_path.exists() else None)
    meta_path.write_text(json.dumps(meta,indent=2)+"\n")
    entry = next(c for c in json.loads((BUILD/"compile_commands.json").read_text())
                 if c["file"].endswith("/mlx/primitives.cpp"))
    args = shlex.split(entry["command"])
    base = ["-O0" if a=="-O3" else a for a in args[:args.index("-o")]]
    test = ROOT/"native_regression.o"
    run(base+["-o",str(test),"-c",str(ROOT/"native_regression.cpp")],"compile-test")
    for label, name in (("before","baseline-primitives.cpp"),("after","patched-baseline-primitives.cpp")):
        obj,binary = ROOT/(label+".o"),ROOT/("native-"+label)
        run(base+["-o",str(obj),"-c",str(ROOT/name)],"compile-"+label)
        run(["/usr/bin/c++",str(test),str(obj),str(ARCHIVE),"-framework","Accelerate","-o",str(binary)],"link-"+label)
        result = run([str(binary)],"run-"+label)
        assert result.returncode == (1 if label=="before" else 0)


if __name__ == "__main__":
    main()
