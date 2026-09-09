"""Small sequential CPU checks of complex unary differentiation."""

import cmath
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

import mlx.core as mx

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent


def pair(value):
    return [value.real, value.imag]


z = 0.25 + 0.5j
c = 0.75 - 0.375j
t = 0.5 + 0.25j
za, ca, ta = (mx.array(value, dtype=mx.complex64) for value in (z, c, t))
cases = [
    ("cos", cmath.cos, lambda z: -cmath.sin(z)),
    ("arcsin", cmath.asin, lambda z: 1/cmath.sqrt(1-z*z)),
    ("arccos", cmath.acos, lambda z: -1/cmath.sqrt(1-z*z)),
    ("arctan", cmath.atan, lambda z: 1/(1+z*z)),
    ("arcsinh", cmath.asinh, lambda z: 1/cmath.sqrt(1+z*z)),
    ("arccosh", cmath.acosh, lambda z: 1/(cmath.sqrt(z-1)*cmath.sqrt(z+1))),
    ("arctanh", cmath.atanh, lambda z: 1/(1-z*z)),
    ("exp", cmath.exp, cmath.exp),
    ("sin", cmath.sin, cmath.cos),
    ("log", cmath.log, lambda z: 1/z),
]
rows = []
started = time.process_time()
for name, forward, derivative in cases:
    try:
        fn = getattr(mx, name)
        actual_forward = fn(za).item()
        _, (vjp,) = mx.vjp(fn, [za], [ca])
        _, (jvp,) = mx.jvp(fn, [za], [ta])
        v, j = vjp.item(), jvp.item()
        expected_v = c*derivative(z).conjugate()
        expected_j = t*derivative(z)
        objective = lambda value: (mx.conj(ca)*fn(mx.array(value, dtype=mx.complex64))).real.item()
        h = 2**-10
        fd = complex((objective(z+h)-objective(z-h))/(2*h), (objective(z+1j*h)-objective(z-1j*h))/(2*h))
        rows.append(dict(name=name, forward=pair(actual_forward), forward_error=abs(actual_forward-forward(z)),
                         vjp=pair(v), expected_vjp=pair(expected_v), vjp_error=abs(v-expected_v),
                         jvp_error=abs(j-expected_j), finite_difference=pair(fd), fd_error=abs(fd-expected_v),
                         adjoint_left=(c.conjugate()*j).real, adjoint_right=(v.conjugate()*t).real))
    except Exception as error:
        rows.append(dict(name=name, error=repr(error)))
    print(json.dumps(rows[-1]), flush=True)
result = dict(mlx=importlib.metadata.version("mlx"), device=str(mx.default_device()),
              z=pair(z), cotangent=pair(c), tangent=pair(t), cases=rows,
              cpu_seconds=time.process_time()-started)
loss = lambda x: mx.real(mx.cos(1j*x))
x = mx.array(1.0)
gradient = mx.grad(loss)(x)
next_x = x - 0.01*gradient
result['real_loss'] = dict(x=x.item(), loss=loss(x).item(), gradient=gradient.item(),
                          expected_gradient=cmath.sinh(1).real,
                          next_x=next_x.item(), next_loss=loss(next_x).item(),
                          reference_next_loss=cmath.cosh(1-0.01*cmath.sinh(1).real).real)
print(json.dumps({'real_loss':result['real_loss']}), flush=True)
(ROOT/'probe-results.json').write_text(json.dumps(result, indent=2)+'\n')
