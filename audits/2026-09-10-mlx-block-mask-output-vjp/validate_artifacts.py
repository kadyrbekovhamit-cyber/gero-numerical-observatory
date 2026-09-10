"""Check consistency of saved evidence without rerunning the numerical suite."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
checks = []


def require(name, condition):
    checks.append(dict(name=name, passed=bool(condition)))


def method(source):
    start = source.index("std::vector<array> BlockMaskedMM::vjp(")
    return source[start:source.index("\n}",start)+2]


def main():
    meta = json.loads((ROOT/"source-metadata.json").read_text())
    wheel = json.loads((ROOT/"wheel-reproduction.json").read_text())
    build = json.loads((ROOT/"build-results.json").read_text())
    require("wheel_version", wheel["version"]=="0.32.2")
    require("wheel_cpu", wheel["device"]=="Device(cpu, 0)")
    for row in wheel["rows"]:
        require("forward_"+str(row["mask"]), row["forward"]==12*row["mask"])
        require("incorrect_gradient_"+str(row["mask"]), row["vjp"]==12*row["mask"])
        require("finite_difference_"+str(row["mask"]), row["finite_difference"]==12)
    require("training_stuck", wheel["training"]["actual_next"]==18)
    require("training_correct_reference", wheel["training"]["expected_next"]==0)
    require("training_gradient", wheel["training"]["actual_gradient"]==0 and wheel["training"]["expected_gradient"]==-36)
    for item in meta["files"]:
        path = ROOT/("upstream-"+item["path"].replace("/","_"))
        require("sha256_"+item["path"], hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"])
    require("source_fetch_success", not meta["errors"])
    source = (ROOT/"baseline-primitives.cpp").read_text()
    upstream = (ROOT/"upstream-mlx_primitives.cpp").read_text()
    patched = (ROOT/"patched-baseline-primitives.cpp").read_text()
    require("same_public_method", method(source)==method(upstream))
    require("one_replacement_only", source.replace(
        "          block_size_,\n          primals[2],\n          lhs_mask,\n          rhs_mask,\n          stream());",
        "          block_size_,\n          std::nullopt,\n          lhs_mask,\n          rhs_mask,\n          stream());",1)==patched)
    for label, failures, failed in (("before",77,76),("after",0,0)):
        text = (ROOT/("run-"+label+".log")).read_text()
        m = re.search(r"SUMMARY scenarios=(\d+) checks=(\d+) failures=(\d+) failed_scenarios=(\d+)",text)
        require(label+"_summary", m and tuple(map(int,m.groups()))==(377,604,failures,failed))
        lines = [x for x in text.splitlines() if x.startswith("FAIL ")]
        require(label+"_fail_count", len(lines)==failures)
        require(label+"_no_exceptions", "exception=" not in text)
        require(label+"_only_output_mask_failures", all("/arg=2" in x or x.startswith(
            ("FAIL simple/vjp","FAIL training/")) for x in lines))
        require(label+"_training_evidence",
                ("EVIDENCE training initial=18 gradient=0 next=18" if label=="before"
                 else "EVIDENCE training initial=18 gradient=-36 next=0") in text)
    require("build_process_count",len(build)==7)
    require("build_exit_codes", all(x["returncode"]==(1 if x["label"]=="run-before" else 0) for x in build))
    searches = json.loads((ROOT/"duplicate-search.json").read_text())
    require("public_search_success",len(searches)==3 and all("result" in q for q in searches))
    for link in re.findall(r"\]\(([^)]+)\)",(ROOT/"README.md").read_text()):
        if not link.startswith("http"):
            require("local_link_"+link,(ROOT/link).exists())
    result = dict(checks=len(checks), failures=sum(not x["passed"] for x in checks), details=checks)
    (ROOT/"artifact-validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="details"}))
    for item in checks:
        if not item["passed"]: print(item)
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
