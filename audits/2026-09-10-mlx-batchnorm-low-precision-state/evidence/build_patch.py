"""Construct an isolated patch; do not edit the MLX checkout."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
source = (ROOT / "upstream-python_mlx_nn_layers_normalization.py").read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "BatchNorm")
lines = source.splitlines(keepends=True)
before = "".join(lines[node.lineno-1:node.end_lineno])
after = before
old = "        reduction_axes = tuple(range(0, x.ndim - 1))\n"
new = """        if x.dtype in (mx.float16, mx.bfloat16):
            x = x.astype(mx.float32)
        reduction_axes = tuple(range(0, x.ndim - 1))
"""
assert after.count(old) == 1
after = after.replace(old, new)
old = "        x = (x - mean) * mx.rsqrt(var + self.eps)\n"
new = """        input_dtype = x.dtype
        x = (x - mean) * mx.rsqrt(var + self.eps)
        if input_dtype in (mx.float16, mx.bfloat16) and (
            self.training or not self.track_running_stats
        ):
            x = x.astype(input_dtype)
"""
assert after.count(old) == 1
after = after.replace(old, new)
patched = "".join(lines[:node.lineno-1]) + after + "".join(lines[node.end_lineno:])
(ROOT / "patched-normalization.py").write_text(patched)
path = "python/mlx/nn/layers/normalization.py"
patch = "".join(difflib.unified_diff(source.splitlines(keepends=True), patched.splitlines(keepends=True),
                                     fromfile="a/"+path, tofile="b/"+path))
(ROOT / "batchnorm-fp32-statistics.patch").write_text(patch)
meta = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            patched_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            changed_class="BatchNorm", added_lines=sum(l.startswith('+') and not l.startswith('+++') for l in patch.splitlines()))
(ROOT / "patch-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
print(patch)
