"""Reuse the previous isolated QMM objects. No complete MLX rebuild."""
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time
ROOT=Path(__file__).resolve().parent
PREV=Path(os.environ["QMM_PREVIOUS_OBJECTS"])
BUILD=Path(os.environ["MLX_CPU_BUILD"])
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
resource.setrlimit(resource.RLIMIT_CPU,(60,60))
records=[]
def run(cmd,label):
    start=time.monotonic()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    r=subprocess.run(cmd,cwd=BUILD,capture_output=True,text=True,timeout=30)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT/(label+".log")).write_text(r.stdout+r.stderr)
    records.append(dict(label=label,command=cmd,returncode=r.returncode,
        wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT/"mixed-build-results.json").write_text(json.dumps(records,indent=2)+"\n")
    print(label,r.returncode,flush=True)
    print(r.stdout if label.startswith("run") else r.stderr[-4000:],flush=True)
    return r.returncode
entry=next(x for x in json.loads((BUILD/"compile_commands.json").read_text())
           if x["file"].endswith("/mlx/primitives.cpp"))
args=shlex.split(entry["command"])
base=["-O0" if x=="-O3" else x for x in args[:args.index("-o")]]
obj=ROOT/"mixed_regression.o"
assert run(base+["-o",str(obj),"-c",str(ROOT/"mixed_regression.cpp")],"compile-mixed")==0
for label in ("before","after"):
    binary=ROOT/("mixed-"+label)
    assert run(["/usr/bin/c++",str(obj),str(PREV/(label+".o")),
        str(BUILD/"mlx-build/libmlx.a"),"-framework","Accelerate","-o",str(binary)],
        "link-"+label)==0
    assert run([str(binary)],"run-"+label)==(1 if label=="before" else 0)
