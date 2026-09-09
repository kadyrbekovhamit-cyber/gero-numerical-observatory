"""Small sequential CPU checks of extrema gradient conservation."""
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

import mlx.core as mx

mx.set_default_device(mx.cpu)
started = time.process_time()
records = []


def record(name, fn, x, expected_gradient, expected_second=None):
    eps = 1 / 256
    row = dict(name=name, input=x.item(), forward=fn(x).item(),
               gradient=mx.grad(fn)(x).item(), expected_gradient=expected_gradient,
               directional_finite_difference=((fn(x+eps)-fn(x-eps))/(2*eps)).item())
    if expected_second is not None:
        row.update(second=mx.grad(mx.grad(fn))(x).item(), expected_second=expected_second)
    records.append(row)
    print(json.dumps(row), flush=True)


for mode in ("maximum", "minimum"):
    direct = getattr(mx, mode)
    record(mode+"_ordinary_identity", lambda t: direct(t,t), mx.array(2.), 1., 0.)
    for n in (1, 2, 4):
        idx = mx.array([0]*n)
        def shared(t, mode=mode, n=n, idx=idx):
            src = t.reshape(1)
            return getattr(src.at[idx], mode)(mx.broadcast_to(t,(n,))).sum()
        record(mode+"_scatter_shared_"+str(n), shared, mx.array(2.), 1., 0.)
        record(mode+"_scatter_shared_square_"+str(n), lambda t: shared(t)**2/2,
               mx.array(2.), 2., 1.)
    def sliced(t, mode=mode):
        return getattr(t.reshape(1).at[0:1],mode)(t.reshape(1)).sum()
    record(mode+"_slice_identity",sliced,mx.array(2.),1.,0.)
    def partial_slice(t, mode=mode):
        return getattr(mx.broadcast_to(t,(3,)).at[1:2],mode)(t.reshape(1)).sum()
    record(mode+"_partial_slice_translation",partial_slice,mx.array(2.),3.,0.)
    src=mx.array([1., 5., 0.])
    upd=mx.array([1.,1.,4.,0.])
    idx=mx.array([0,0,1,2])
    cot=mx.array([2.,-3.,4.])
    def operation(a,b,mode=mode):
        return getattr(a.at[idx],mode)(b)
    _, gradients=mx.vjp(operation,[src,upd],[cot])
    row=dict(name=mode+"_vjp_sum",source=src.tolist(),updates=upd.tolist(),
             indices=idx.tolist(),cotangent=cot.tolist(),
             source_gradient=gradients[0].tolist(),update_gradient=gradients[1].tolist(),
             total_gradient=(gradients[0].sum()+gradients[1].sum()).item(),
             expected_total=cot.sum().item())
    records.append(row)
    print(json.dumps(row),flush=True)
    record(mode+"_common_shift",lambda t:(operation(src+t,upd+t)*cot).sum(),
           mx.array(0.),cot.sum().item(),0.)

result=dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
            cases=records,cpu_seconds=time.process_time()-started)
Path(__file__).with_name("probe-results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
