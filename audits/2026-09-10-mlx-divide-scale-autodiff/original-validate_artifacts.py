"""Validate saved evidence and apply the patch only in a temporary directory."""
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from build_and_test import method, patch
ROOT=Path(__file__).resolve().parent
checks=[]
start=time.process_time()
def check(label,ok):checks.append(dict(label=label,passed=bool(ok)))
def read(name):return json.loads((ROOT/name).read_text())
def finite(x):return isinstance(x,(float,int)) and math.isfinite(x)
before,after=read("run-before.json"),read("run-after.json")
check("1328 cases each side",len(before)==len(after)==1328)
check("600 baseline mismatches",sum(not x["passed"] for x in before)==600)
check("all final cases pass",all(x["passed"] for x in after))
check("matching cases references tolerances",all(
    (a["label"],a["dtype"],a["expected"],a["tolerance"])==
    (b["label"],b["dtype"],b["expected"],b["tolerance"]) for a,b in zip(before,after)))
check("all references finite",all(finite(v) for x in after for v in x["expected"]))
check("all outputs finite",all(finite(v) for x in after for v in x["actual"]))
check("dtype coverage",collections.Counter(x["dtype"] for x in after)==
      {"float16":280,"bfloat16":280,"float32":376,"float64":376,"complex64":16})
check("recomputed final comparisons",all(
    len(x["actual"])==len(x["expected"]) and all(
        abs(a)<=x["tolerance"] if e==0 else abs(a/e-1)<=x["tolerance"]
        for a,e in zip(x["actual"],x["expected"])) for x in after))
check("complex controls pass both versions",all(x["passed"] for x in before+after if x["dtype"]=="complex64"))
check("zero-gradient higher order coverage",sum(x["label"].startswith("zero-gradient/") for x in after)==36)
check("weighted exponent cases",sum(x["label"].startswith("weighted/") for x in after)==112)
check("mixed Hessian coverage",sum(x["label"].startswith("mixed-Hessian/") for x in after)==6)
wheel=read("wheel-reproduction.json")
check("wheel version and CPU",wheel["version"]=="0.32.2" and wheel["device"]=="Device(cpu, 0)")
check("wheel direct values",len(wheel["cases"])==12 and all(x["value"]==1 for x in wheel["cases"]))
check("wheel finite difference evidence",len(wheel["compositions"])==6 and all(
    abs(x["finite_difference"]+1)<3e-4 for x in wheel["compositions"]))
check("wheel scale-dependent gradient failure",all(
    x["gradient"]==("-Infinity" if x["scale"]<1 else 0) and
    x["hessian"]==x["third"]=="NaN"
    for x in wheel["compositions"] if x["scale"]!=1))
meta=read("source-metadata.json")
check("public fetch without errors",not meta["errors"])
for record in meta["files"]:
    path=ROOT/("upstream-"+record["path"].replace("/","_"))
    check("source hash "+record["path"],hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"])
native=read("native-source-metadata.json")
check("JVP and VJP match public main",len(native["target_methods_identical"])==2 and
      all(native["target_methods_identical"].values()))
for dep in native["dependencies"]:
    check("dependency "+Path(dep["path"]).name,
          hashlib.sha256(Path(dep["path"]).read_bytes()).hexdigest()==dep["sha256"])
source=(ROOT/"upstream-mlx_primitives.cpp").read_text()
patched=patch(source)
restored=patched.replace((ROOT/"weighted_partial.cpp.inc").read_text()+"\n","",1)
for name in ("vjp","jvp"):
    old,new=method(source,name),method(patched,name)
    begin=new.index("    } else if (!issubdtype(primals[0].dtype(), complexfloating))")
    end=new.index("    } else {",begin)
    check(name+" original branches preserved",new[:begin]+new[end:]==old)
    restored=restored.replace(method(restored,name),old,1)
check("only target methods and helper changed",restored==source)
with tempfile.TemporaryDirectory(prefix="mlx-divide-patch-") as temp:
    target=Path(temp)/"mlx/primitives.cpp";target.parent.mkdir();target.write_text(source)
    result=subprocess.run(["patch","-p1","-i",str(ROOT/"divide-scale-autodiff.patch")],
        cwd=temp,capture_output=True,text=True,timeout=5)
    check("patch applies to saved current main",result.returncode==0)
    check("applied patch exact content",target.read_text()==patched)
search=read("duplicate-search.json")
check("four complete duplicate searches",len(search)==4 and all(
    "result" in q and not q["result"]["incomplete_results"] and
    len(q["result"]["items"])==q["result"]["total_count"] for q in search))
check("18 unique public results",len({x["number"] for q in search for x in q["result"]["items"]})==18)
check("native return codes",[x["returncode"] for x in read("build-results.json")]==[0,0,0,1,0,0])
for name,count in (("hyperbolic",518),("arctan2",1368)):
    cases=read("compat-run-"+name+".json")
    check(name+" compatibility cases",len(cases)==count and all(x["passed"] for x in cases))
for dep in read("compatibility-metadata.json")["dependencies"]:
    check("compatibility object "+dep["suite"],
          hashlib.sha256(Path(dep["path"]).read_bytes()).hexdigest()==dep["sha256"])
check("compatibility commands succeeded",len(read("compatibility-build-results.json"))==5 and
      all(x["returncode"]==0 for x in read("compatibility-build-results.json")))
out=dict(checks=checks,failures=sum(not x["passed"] for x in checks),
         cpu_seconds=time.process_time()-start)
(ROOT/"artifact-check-results.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
raise SystemExit(bool(out["failures"]))
