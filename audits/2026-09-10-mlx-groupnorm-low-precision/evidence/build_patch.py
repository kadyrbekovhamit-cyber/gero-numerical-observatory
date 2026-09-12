"""Build an isolated low-precision GroupNorm patch against the public snapshot."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT/'upstream-python_mlx_nn_layers_normalization.py').read_text()
klass=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='GroupNorm')
method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_group_norm')
lines=source.splitlines(keepends=True)
before=''.join(lines[method.lineno-1:method.end_lineno])
old='''        means = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
'''
new='''        if x.size and x.dtype in (mx.float16, mx.bfloat16):
            x = mx.fast.layer_norm(
                x.transpose(0, 2, 1), eps=self.eps, weight=None, bias=None
            ).transpose(0, 2, 1)
        else:
            means = mx.mean(x, axis=1, keepdims=True)
            var = mx.var(x, axis=1, keepdims=True)
            x = (x - means) * mx.rsqrt(var + self.eps)
'''
assert before.count(old)==1
after=before.replace(old,new)
patched=''.join(lines[:method.lineno-1])+after+''.join(lines[method.end_lineno:])
(ROOT/'patched-normalization.py').write_text(patched)
path='python/mlx/nn/layers/normalization.py'
patch=''.join(difflib.unified_diff(source.splitlines(keepends=True),patched.splitlines(keepends=True),fromfile='a/'+path,tofile='b/'+path))
(ROOT/'groupnorm-low-precision.patch').write_text(patch)
metadata=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              patched_sha256=hashlib.sha256(patched.encode()).hexdigest(),
              changed_method='GroupNorm._group_norm')
(ROOT/'patch-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(patch)
