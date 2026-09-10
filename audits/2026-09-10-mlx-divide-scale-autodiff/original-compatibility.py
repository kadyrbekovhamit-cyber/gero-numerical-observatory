"""Sequential compatibility checks against two prior local patches."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time
ROOT=Path(__file__).resolve().parent
BUILD=ROOT.parent/"current-stack-2026-09-07-round3/build-native"
ARCHIVE=BUILD/"mlx-build/libmlx.a"
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
resource.setrlimit(resource.RLIMIT_CPU,(120,120))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
source=(ROOT/"baseline-primitives.cpp").read_text()
packages=[
    ("hyperbolic",ROOT.parent/"mlx-inverse-hyperbolic-overflow-2026-09-10",518),
    ("arctan2",ROOT.parent/"mlx-arctan2-scale-autodiff-2026-09-10",1368)]
for name,path,count in packages:
    source=load("compat_"+name,path/"build_and_test.py").patch(source)
source=load("compat_divide",ROOT/"build_and_test.py").patch(source)
(ROOT/"combined-primitives.cpp").write_text(source)
records=[]
def run(cmd,label):
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
    r=subprocess.run(cmd,cwd=BUILD,capture_output=True,text=True,timeout=60)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT/(label+".log")).write_text(r.stdout+r.stderr)
    records.append(dict(label=label,command=cmd,returncode=r.returncode,
        wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT/"compatibility-build-results.json").write_text(json.dumps(records,indent=2)+"\n")
    cases=[json.loads(line[5:]) for line in r.stdout.splitlines() if line.startswith("CASE ")]
    if cases:(ROOT/(label+".json")).write_text(json.dumps(cases,indent=2)+"\n")
    print(label,r.returncode,*[x for x in r.stdout.splitlines() if x.startswith("SUMMARY")],flush=True)
    if r.returncode:
        print((r.stdout+r.stderr)[-3000:]);raise SystemExit(1)
    return cases
db=json.loads((BUILD/"compile_commands.json").read_text())
entry=next(x for x in db if x["file"].endswith("/mlx/primitives.cpp"))
args=shlex.split(entry["command"])
base=["-O0" if x=="-O3" else x for x in args[:args.index("-o")]]
run(base+["-o",str(ROOT/"combined.o"),"-c",str(ROOT/"combined-primitives.cpp")],"compat-compile")
dependencies=[]
for name,path,count in packages:
    obj=path/"test.o";binary=ROOT/("compat-"+name)
    dependencies.append(dict(suite=name,checks=count,path=str(obj),
        sha256=hashlib.sha256(obj.read_bytes()).hexdigest()))
    run(["/usr/bin/c++",str(obj),str(ROOT/"combined.o"),str(ARCHIVE),
         "-framework","Accelerate","-o",str(binary)],"compat-link-"+name)
    rows=run([str(binary)],"compat-run-"+name)
    assert len(rows)==count and all(x["passed"] for x in rows)
(ROOT/"compatibility-metadata.json").write_text(json.dumps(dict(
    dependencies=dependencies,
    scope="Three local patches together on the same baseline TU; prebuilt CPU archive."),
    indent=2)+"\n")
