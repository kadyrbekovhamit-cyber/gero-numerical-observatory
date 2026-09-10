"""Validate the saved audit artifacts without MLX or network access."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
checks = []
def check(name, ok):
    checks.append(dict(name=name, passed=bool(ok)))

meta = json.loads((ROOT / "source-metadata.json").read_text())
check("source_fetch_errors_empty", not meta["errors"])
check("pinned_public_revision", meta["upstream_revision"] == "81ba1c6a0e50a9268b931579c2d4f1158b9aab5a")
for entry in meta["files"]:
    p = ROOT / ("upstream-" + entry["path"].replace("/", "_"))
    check("source_hash/"+entry["path"], hashlib.sha256(p.read_bytes()).hexdigest() == entry["sha256"])
for p in sorted(ROOT.glob("*.py")):
    ast.parse(p.read_text(), filename=str(p))
    check("python_syntax/"+p.name, True)

original = (ROOT / "upstream-python_mlx_optimizers_optimizers.py").read_text()
patched = (ROOT / "patched-optimizers.py").read_text()
candidate = (ROOT / "candidate.py").read_text()
def function(source):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "clip_grad_norm")
check("tested_body_matches_patch", ast.dump(ast.Module(body=function(patched).body[1:], type_ignores=[])) ==
      ast.dump(ast.Module(body=function(candidate).body, type_ignores=[])))
before_tree, after_tree = ast.parse(original), ast.parse(patched)
before_tree.body = [n for n in before_tree.body if not isinstance(n, ast.FunctionDef) or n.name != "clip_grad_norm"]
after_tree.body = [n for n in after_tree.body if not isinstance(n, ast.FunctionDef) or n.name != "clip_grad_norm"]
check("all_other_top_level_code_unchanged", ast.dump(before_tree) == ast.dump(after_tree))
check("public_matches_installed_function", ast.dump(function(original)) ==
      ast.dump(function((ROOT / "installed-clip-source.py").read_text())))
check("frozen_function_exact", ast.dump(function(original)) ==
      ast.dump(function((ROOT / "frozen-clip-source.py").read_text())))

with tempfile.TemporaryDirectory(prefix="patch-check-", dir=ROOT) as temp:
    target = Path(temp) / "python/mlx/optimizers/optimizers.py"
    target.parent.mkdir(parents=True)
    target.write_text(original)
    for flags in (["--check"], []):
        p = subprocess.run(["git", "apply", *flags, str(ROOT / "clip-grad-norm-range.patch")],
                           cwd=temp, capture_output=True, text=True, timeout=5)
        check("git_apply/"+("check" if flags else "apply"), p.returncode == 0)
        if p.returncode: print(p.stderr)
    check("applied_patch_equals_saved_source", target.read_text() == patched)

main = json.loads((ROOT / "regression-results.json").read_text())
compat = json.loads((ROOT / "compatibility-results.json").read_text())
check("main_before_summary", main["summary"]["before"] == dict(checks=1701, failed=230))
check("main_after_summary", main["summary"]["after"] == dict(checks=1701, failed=0))
check("compat_before_summary", compat["summary"]["before"] == dict(checks=31, failed=17))
check("compat_after_summary", compat["summary"]["after"] == dict(checks=39, failed=0))
for label, result in (("main", main), ("compat", compat)):
    for mode in ("before", "after"):
        rows = [r for r in result["checks"] if r["mode"] == mode]
        summary = dict(checks=len(rows), failed=sum(not r["passed"] for r in rows))
        check(label+"/recomputed_summary/"+mode, summary == result["summary"][mode])
    check(label+"/real_cpu_wheel", result["version"] == "0.32.2" and result["device"] == "Device(cpu, 0)")
    check(label+"/bounded_cpu_time", result["cpu_seconds"] < 30)
check("219_input_scenarios", sum(r["mode"] == "after" and r["name"].endswith("/norm") for r in main["checks"]) == 219)
check("first_candidate_failure_preserved", json.loads((ROOT / "compatibility-v1-results.json").read_text())["summary"]["after"]["failed"] == 2)

search = json.loads((ROOT / "duplicate-search.json").read_text())
check("all_four_searches_complete", len(search) == 4 and all(
    "error" not in q and not q["result"]["incomplete_results"] and
    q["result"]["total_count"] == len(q["result"]["items"]) for q in search))
check("nine_unique_hits", len({i["number"] for q in search for i in q["result"]["items"]}) == 9)
comments = json.loads((ROOT / "duplicate-comments.json").read_text())
check("thirteen_issue_comments_fetched", len(comments) == 4 and all("error" not in q for q in comments)
      and sum(len(q["comments"]) for q in comments) == 13)
for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
    if not target.startswith("https://"):
        check("report_link/"+target, (ROOT / target).exists())

result = dict(checks=checks, total=len(checks), failed=sum(not c["passed"] for c in checks))
(ROOT / "artifact-validation.json").write_text(json.dumps(result, indent=2)+"\n")
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir())
            if p.is_file() and p.name != "SHA256SUMS.json"}
(ROOT / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(dict(total=result["total"], failed=result["failed"])))
for item in checks:
    if not item["passed"]: print(item)
raise SystemExit(bool(result["failed"]))
