"""Sequential isolated native test; no complete MLX build."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time
ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ["MLX_SOURCE_ROOT"])
BUILD=Path(os.environ["MLX_CPU_BUILD"])
ARCHIVE=BUILD/"mlx-build/libmlx.a"
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
resource.setrlimit(resource.RLIMIT_CPU,(120,120))

def method(source):
    begin=source.index("std::vector<array> Scan::vjp(")
    return source[begin:source.index("\n}",begin)+2]

def patch(source):
    start=source.index("  } else if (reduce_type_ == Scan::LogAddExp) {",source.index("std::vector<array> Scan::vjp("))
    end=source.index("  } else if (reduce_type_ == Scan::Prod) {",start)
    replacement="""  } else if (reduce_type_ == Scan::LogAddExp) {
    return {log_scan_product(
        primals[0], cotangents[0], axis_, reverse_, inclusive_, true, stream())};
"""
    source=source[:start]+replacement+source[end:]
    source=source.replace("#include <numeric>","#include <limits>\n#include <numeric>",1)
    source=source.replace('#include "mlx/utils.h"','#include "mlx/transforms.h"\n#include "mlx/utils.h"',1)
    return source.replace("std::vector<array> Scan::vjp(",
        (ROOT/"logscan_product.cpp.inc").read_text()+"\nstd::vector<array> Scan::vjp(",1)

records=[]
def run(command,label):
    start=time.monotonic()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    r=subprocess.run(command,cwd=BUILD,capture_output=True,text=True,timeout=60)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT/(label+".log")).write_text(r.stdout+r.stderr)
    records.append(dict(label=label,command=command,returncode=r.returncode,
        wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT/"build-results.json").write_text(json.dumps(records,indent=2)+"\n")
    print(label,r.returncode,flush=True)
    if label.startswith("run-"):
        print("\n".join(x for x in r.stdout.splitlines() if x.startswith(("SUMMARY","EVIDENCE"))),flush=True)
    elif r.returncode:
        print(r.stderr[-5000:],flush=True)
        raise SystemExit(r.returncode)
    return r.returncode

def main():
    source=subprocess.run(["git","show","ce916dbbcaa88e433b6fd1e60a17f766d49c27fe:mlx/primitives.cpp"],cwd=REPO,
                          capture_output=True,text=True,check=True).stdout
    upstream_path=ROOT/"upstream-mlx_primitives.cpp"
    upstream=upstream_path.read_text() if upstream_path.exists() else source
    (ROOT/"baseline-primitives.cpp").write_text(source)
    (ROOT/"patched-baseline-primitives.cpp").write_text(patch(source))
    (ROOT/"logcumsumexp-higher-order-prototype.patch").write_text("".join(difflib.unified_diff(
        upstream.splitlines(True),patch(upstream).splitlines(True),
        fromfile="a/mlx/primitives.cpp",tofile="b/mlx/primitives.cpp")))
    meta_path=ROOT/"source-metadata.json"
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(baseline_revision=subprocess.run(["git","rev-parse","ce916dbbcaa88e433b6fd1e60a17f766d49c27fe"],cwd=REPO,
        capture_output=True,text=True,check=True).stdout.strip(),
        archive=str(ARCHIVE),archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        upstream_scan_vjp_identical=method(source)==method(upstream) if upstream_path.exists() else None,
        build="Pristine/patched baseline primitives.cpp linked before an existing CPU archive; not a full current-main rebuild.",
        patch_status="Research prototype with explicit reverse rules for Jacobian products; not performance-qualified.")
    meta_path.write_text(json.dumps(meta,indent=2)+"\n")
    entry=next(x for x in json.loads((BUILD/"compile_commands.json").read_text())
        if x["file"].endswith("/mlx/primitives.cpp"))
    args=shlex.split(entry["command"])
    base=["-O0" if x=="-O3" else x for x in args[:args.index("-o")]]
    obj=ROOT/"native_regression.o"
    run(base+["-o",str(obj),"-c",str(ROOT/"native_regression.cpp")],"compile-test")
    for label,name in (("before","baseline-primitives.cpp"),("after","patched-baseline-primitives.cpp")):
        part=ROOT/(label+".o")
        binary=ROOT/("native-"+label)
        run(base+["-o",str(part),"-c",str(ROOT/name)],"compile-"+label)
        run(["/usr/bin/c++",str(obj),str(part),str(ARCHIVE),"-framework","Accelerate",
             "-o",str(binary)],"link-"+label)
        code=run([str(binary)],"run-"+label)
        assert code==(1 if label=="before" else 0)

if __name__=="__main__":main()
