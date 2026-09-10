"""Check stored test outcomes, upstream provenance and patch applicability."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parent
EXP = ROOT/"known-float64-exp"
records = []
def check(name, condition):
    records.append(dict(check=name, passed=bool(condition)))
def data(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
for folder, cases in ((ROOT, {"before":348,"scan-only":120,"exp-only":401,"combined":0}),
                      (EXP, {"before":34,"after":0})):
    expected_n = 1122 if folder == ROOT else 47
    for label, expected_fails in cases.items():
        rows = data(folder/("run-"+label+".json"))
        check(folder.name+"/"+label, len(rows)==expected_n and
              sum(not r["passed"] for r in rows)==expected_fails)
    for file in folder.glob("*.json"):
        data(file)
    for match in re.findall(r"\]\(([^)]+)\)", (folder/"README.md").read_text()):
        if not match.startswith(("https://","http://")):
            check("link:"+folder.name+"/"+match, (folder/match.split("#")[0]).exists())
    meta = data(folder/"source-metadata.json")
    check("fetch:"+folder.name, not meta["errors"])
    for item in meta["files"]:
        target = ROOT if item["path"] == "mlx/primitives.cpp" else EXP
        saved = target/("upstream-"+item["path"].replace("/","_"))
        check("sha256:"+item["path"], hashlib.sha256(saved.read_bytes()).hexdigest()==item["sha256"])
    check("duplicate-search:"+folder.name, all("result" in x for x in data(folder/"duplicate-search.json")))
check("float64-only-residual", all("-f64" in x["label"] for x in
    data(ROOT/"run-scan-only.json") if not x["passed"]))
check("higher-orders-477", "SUMMARY scenarios=186 checks=477 failures=0 failed_scenarios=0"
      in (ROOT/"run-higher-order.log").read_text())
check("original-failed-attempt-preserved", data(ROOT/"initial-run-after.json")==data(ROOT/"run-scan-only.json"))
check("upstream-exp-header-identical", (EXP/"baseline-math.h").read_bytes()==
      (EXP/"upstream-mlx_backend_cpu_simd_math.h").read_bytes())
spec = importlib.util.spec_from_file_location("scanbuild", ROOT/"build_and_test.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
check("upstream-Scan-vjp-identical", module.method((ROOT/"higher-order/baseline-primitives.cpp").read_text()) ==
      module.method((ROOT/"upstream-mlx_primitives.cpp").read_text()))
for folder, source_path, patch_name in (
    (ROOT,"mlx/primitives.cpp","stable-logcumsumexp-vjp-prototype.patch"),
    (EXP,"mlx/backend/cpu/simd/math.h","float64-exp.patch")):
    with tempfile.TemporaryDirectory(prefix="mlx-audit-patch-") as directory:
        target = Path(directory)/source_path
        target.parent.mkdir(parents=True)
        target.write_bytes((folder/("upstream-"+source_path.replace("/","_"))).read_bytes())
        p = subprocess.run(["/usr/bin/patch","--dry-run","-p1","-i",str(folder/patch_name)],
                           cwd=directory,text=True,capture_output=True,timeout=10)
        check("patch-applies:"+patch_name,p.returncode==0)
        (folder/"patch-dry-run.log").write_text(p.stdout+p.stderr)
summary = dict(checks=len(records),failures=sum(not r["passed"] for r in records),results=records)
(ROOT/"artifact-validation.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
raise SystemExit(bool(summary["failures"]))
