"""Check saved evidence and links without repeating numerical tests."""
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
rows=[]
def check(name,condition):rows.append(dict(name=name,passed=bool(condition)))
def read_json(path):
    def invalid(x):raise ValueError("Nonstandard JSON constant: "+x)
    return json.loads(path.read_text(),parse_constant=invalid)
def main():
    meta=read_json(ROOT/"source-metadata.json")
    wheel=read_json(ROOT/"wheel-reproduction.json")
    build=read_json(ROOT/"build-results.json")
    check("version",wheel["version"]=="0.32.2")
    check("cpu",wheel["device"]=="Device(cpu, 0)")
    one=wheel["singleton"]
    check("singleton_measured",one["second"]==0 and one["value"]==0 and one["gradient"]==0)
    check("singleton_finite_difference",one["finite_curvature"]==one["expected_second"]==1)
    two=wheel["two_element"]
    check("two_element_measured",two["hessian"]==[[0,0],[0,0]])
    check("two_element_expected",two["expected_hessian"]==[[1.25,.25],[.25,.25]])
    check("two_element_finite_difference",all(abs(a-b)<1e-5 for r,s in
        zip(two["finite_hessian"],two["expected_hessian"]) for a,b in zip(r,s)))
    check("direct_cotangent_nan",all(x=="NaN" for r in
        two["derivative_of_vjp_wrt_cotangent"] for x in r))
    check("source_errors",not meta["errors"])
    for item in meta["files"]:
        p=ROOT/("upstream-"+item["path"].replace("/","_"))
        check("sha256_"+item["path"],hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"])
    def method(text):
        start=text.index("std::vector<array> Scan::vjp(")
        return text[start:text.index("\n}",start)+2]
    check("upstream_method",method((ROOT/"baseline-primitives.cpp").read_text())==
        method((ROOT/"upstream-mlx_primitives.cpp").read_text()))
    check("helper_embedded",(ROOT/"logscan_product.cpp.inc").read_text() in
        (ROOT/"patched-baseline-primitives.cpp").read_text())
    for label,fail,scenes in (("before",189,59),("after",0,0)):
        log=(ROOT/("run-"+label+".log")).read_text()
        match=re.search(r"SUMMARY scenarios=(\d+) checks=(\d+) failures=(\d+) failed_scenarios=(\d+)",log)
        check(label+"_summary",match and tuple(map(int,match.groups()))==(186,477,fail,scenes))
        lines=[s for s in log.splitlines() if s.startswith("FAIL")]
        check(label+"_failure_count",len(lines)==fail)
        check(label+"_no_exceptions","exception=" not in log)
        check(label+"_first_order_passed",not any("/gradient " in s for s in lines))
        check(label+"_singleton","EVIDENCE singleton second="+("0" if label=="before" else "1")+" finite=1" in log)
    after=(ROOT/"run-after.log").read_text()
    check("third_order_one","EVIDENCE third n=1 actual=1 expected=1" in after)
    check("third_order_two","EVIDENCE third n=2 actual=2 expected=2" in after)
    check("build_count",len(build)==7)
    check("build_codes",all(r["returncode"]==(1 if r["label"]=="run-before" else 0) for r in build))
    searches=read_json(ROOT/"duplicate-search.json")
    check("search_success",len(searches)==3 and all("result" in r for r in searches))
    check("search_pages_complete",all(r["result"]["total_count"]==len(r["result"]["items"]) for r in searches))
    scout=ROOT/"gather-qmm-followup"
    probe=read_json(scout/"probe-results.json")
    check("corrected_transpose_oracle",all(r["expected"]==(1 if r["transpose"] else 32)
        for r in probe["qmm"] if "expected" in r))
    for label,fails in (("before",4),("after",0)):
        log=(scout/("run-"+label+".log")).read_text()
        check("mixed_"+label,"SUMMARY checks=24 failures="+str(fails) in log)
    for root in (ROOT,scout):
        for link in re.findall(r"\]\(([^)]+)\)",(root/"README.md").read_text()):
            if not link.startswith("http"):check(root.name+"/link/"+link,(root/link).exists())
    result=dict(checks=len(rows),failures=sum(not r["passed"] for r in rows),details=rows)
    (ROOT/"artifact-validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="details"}))
    for r in rows:
        if not r["passed"]:print(r)
    raise SystemExit(bool(result["failures"]))
if __name__=="__main__":main()
