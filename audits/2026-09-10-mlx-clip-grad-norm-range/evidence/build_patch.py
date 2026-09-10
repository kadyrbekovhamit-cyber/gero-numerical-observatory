"""Build an isolated patch against the frozen public source."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
original = (ROOT / "upstream-python_mlx_optimizers_optimizers.py").read_text()
candidate = (ROOT / "candidate.py").read_text()
node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == "clip_grad_norm")
lines = original.splitlines(keepends=True)
old_function = "".join(lines[node.lineno-1:node.end_lineno])
old_head = old_function[:old_function.index("    norm_squared = tree_reduce")]
new_body = candidate[candidate.index("    leaves = "):]
new_function = old_head + new_body
patched = "".join(lines[:node.lineno-1]) + new_function + "".join(lines[node.end_lineno:])
(ROOT / "patched-optimizers.py").write_text(patched)
path = "python/mlx/optimizers/optimizers.py"
patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), patched.splitlines(keepends=True),
                                     fromfile="a/"+path, tofile="b/"+path))
(ROOT / "clip-grad-norm-range.patch").write_text(patch)
new_node = next(n for n in ast.parse(patched).body if isinstance(n, ast.FunctionDef) and n.name == "clip_grad_norm")
candidate_node = next(n for n in ast.parse(candidate).body if isinstance(n, ast.FunctionDef) and n.name == "clip_grad_norm")
assert ast.dump(ast.Module(body=new_node.body[1:], type_ignores=[])) == ast.dump(ast.Module(body=candidate_node.body, type_ignores=[]))
meta = dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            patched_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            tested_candidate_body_matches_patch=True, changed_function="clip_grad_norm")
(ROOT / "patch-metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
