"""Reproduce numerical observations on native MLX with synthetic CPU arrays."""
import ast
import base64
from decimal import Decimal, localcontext
import hashlib
import importlib.metadata as md
import inspect
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import types

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import mlx.core as mx
import mlx.nn.losses as installed_losses

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('MLX_REPO', str(ROOT / 'mlx-source')))
PIN = 'ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == PIN
source = subprocess.check_output(['git', 'show', PIN + ':python/mlx/nn/losses.py'], cwd=REPO)
(ROOT / 'pinned-losses.py').write_bytes(source)
module = types.ModuleType('pinned_losses')
exec(compile(source, str(ROOT / 'pinned-losses.py'), 'exec'), module.__dict__)


def encode(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf')
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def reference(x, y, v, eps):
    with localcontext() as ctx:
        ctx.prec = 80
        xd, yd, vd, ed = map(Decimal.from_float, (x, y, v, eps))
        z = max(vd, ed)
        r = xd - yd
        loss = (z.ln() + r * r / z) / 2
        dv = (z-r*r)/(2*z*z) if vd > ed else Decimal(0)
        return {'loss': float(loss), 'dx': float(r/z), 'dy': float(-r/z),
                'dv': float(dv), 'loss_decimal': str(loss), 'dv_decimal': str(dv)}


def normalized(x, y, v):
    v = mx.maximum(v, 1e-6)
    return (0.5 * (mx.log(v) + mx.square((x-y)/mx.sqrt(v)))).mean()


def ratio_first(x, y, v):
    v = mx.maximum(v, 1e-6)
    r = x-y
    return (0.5 * (mx.log(v) + r*(r/v))).mean()


def promoted(x, y, v):
    return module.gaussian_nll_loss(x.astype(mx.float32), y.astype(mx.float32),
                                    v.astype(mx.float32))


def promoted_normalized(x, y, v):
    return normalized(x.astype(mx.float32), y.astype(mx.float32), v.astype(mx.float32))


def measure(fn, x, y, v):
    out = fn(x, y, v)
    grads = mx.grad(fn, argnums=(0, 1, 2))(x, y, v)
    mx.eval(out, *grads)
    return dict(loss=out.item(), dx=grads[0].item(), dy=grads[1].item(),
                dv=grads[2].item(), loss_dtype=str(out.dtype))


observations = []
case_sets = {
    'float16': [
        ('control', 2., 0., 4.), ('reported_overflow', 300., 0., 300.),
        ('finite_loss_wrong_sign', 20., 0., 300.),
        ('finite_loss_wrong_magnitude', 10., 0., 300.),
        ('small_variance', .02, 0., .0001),
        ('zero_residual_small_variance', 0., 0., .0001),
        ('below_eps', 0., 0., 1e-7),
        ('boundary_255', 255., 0., 300.), ('boundary_256', 256., 0., 300.),
        ('unrepresentable_loss_control', 400., 0., 1.),
    ],
    'float32': [
        ('control', 2., 0., 4.), ('reported_overflow', 1e20, 0., 1e20),
        ('finite_loss_wrong_sign', 2e10, 0., 1e20),
        ('extreme_normalization_limit', 3e38, 0., 3e38),
    ],
}
functions = {'pinned': module.gaussian_nll_loss,
             'installed': installed_losses.gaussian_nll_loss,
             'normalized_diagnostic': normalized, 'ratio_first_diagnostic': ratio_first,
             'promote_fp32_only': promoted, 'promote_and_normalize': promoted_normalized}
for label, cases in case_sets.items():
    dtype = getattr(mx, label)
    for name, a, b, c in cases:
        x, y, v = (mx.array([z], dtype=dtype) for z in (a, b, c))
        vals = [z.item() for z in (x, y, v)]
        effective_eps = mx.array(1e-6, dtype=dtype).item()
        row = dict(case=name, dtype=label, rounded_inputs=vals,
                   effective_eps=effective_eps, reference=reference(*vals, effective_eps),
                   intermediates={'residual': (x-y).item(), 'square_residual': mx.square(x-y).item(),
                                  'square_variance': mx.square(v).item()},
                   variants={key: measure(fn, x, y, v) for key, fn in functions.items()})
        observations.append(row)
        print(label, name, encode(row['variants']['pinned']), 'reference', row['reference']['dv'], flush=True)

division = []
for label, a, b in [('float16', 300., 300.), ('float16', .0001, .0001),
                    ('float16', 2., 4.), ('float32', 1e20, 1e20)]:
    dtype = getattr(mx, label); x = mx.array([a], dtype=dtype); y = mx.array([b], dtype=dtype)
    fn = lambda n, d: (n/d).sum()
    out=fn(x,y); grads=mx.grad(fn,argnums=(0,1))(x,y); mx.eval(out,*grads)
    division.append(dict(dtype=label, inputs=[x.item(),y.item()], value=out.item(),
                         denominator_gradient=grads[1].item(),
                         reference_denominator_gradient=-x.item()/y.item()**2,
                         denominator_square=mx.square(y).item()))

# Compare the actual forward curve around the finite FP16 sign reversal.
fd=[]
for h in [2., 8., 16., 32.]:
    x=mx.array([20.],dtype=mx.float16); y=mx.array([0.],dtype=mx.float16)
    vp=mx.array([300.+h],dtype=mx.float16); vm=mx.array([300.-h],dtype=mx.float16)
    plus=module.gaussian_nll_loss(x,y,vp).item(); minus=module.gaussian_nll_loss(x,y,vm).item()
    fd.append(dict(step=h, plus=plus, minus=minus, native_forward_slope=(plus-minus)/(2*h)))

steps=[]
for candidate in [300., 298., 302.]:
    x=mx.array([20.],dtype=mx.float16); y=mx.array([0.],dtype=mx.float16); v=mx.array([candidate],dtype=mx.float16)
    steps.append(dict(variance=candidate, native_loss=module.gaussian_nll_loss(x,y,v).item(),
                      reference_loss=reference(20.,0.,candidate,1e-6)['loss']))

def function_ast(text):
    n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='gaussian_nll_loss')
    return ast.dump(n, include_attributes=False)

binary=Path(mx.__file__); records=[]
for package in ['mlx', 'mlx-metal']:
    try: dist=md.distribution(package)
    except md.PackageNotFoundError: continue
    for file in dist.files or []:
        path=Path(dist.locate_file(file))
        if path.resolve()==binary.resolve() and file.hash:
            computed=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
            records.append({'distribution':package,'version':dist.version,'record_hash_matches':computed==file.hash.value})
provenance=dict(pinned_revision=PIN, python=sys.version, executable=sys.executable, platform=platform.platform(),
                mlx_version=md.version('mlx'), default_device=str(mx.default_device()),
                source_sha256=hashlib.sha256(source).hexdigest(), native_core_path=str(binary),
                native_core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), native_record_checks=records,
                gaussian_ast_equal_to_installed=function_ast(source.decode())==function_ast(Path(installed_losses.__file__).read_text()),
                scope='Native installed core on CPU; pinned Python source and separately installed public API. No rebuild, Metal kernels or model training.')
result=encode(dict(provenance=provenance, gaussian=observations, primitive_divide=division,
                   native_forward_finite_differences=fd, local_variance_steps=steps))
(ROOT/'mlx-review-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('DIVIDE',encode(division),flush=True)
print('FINITE DIFFERENCES',fd,flush=True)
print('PROVENANCE',provenance,flush=True)
