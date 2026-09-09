"""Tiny, CPU-only check of masked assignment batch independence."""
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent


def assign(dst, src, mask):
    out = dst + 0
    out[mask] = src
    return out


def main():
    started = time.process_time()
    dst = mx.zeros((2, 4))
    src = mx.array([[10., 20., 99.], [30., 40., 88.]])
    mask = mx.array([[True, False, True, False], [False, True, False, True]])
    cot = mx.array([[1., 2., 3., 4.], [10., 20., 30., 40.]])
    mapped = lambda a, b: mx.vmap(assign)(a, b, mask)
    loop = lambda a, b: mx.stack([assign(a[i], b[i], mask[i]) for i in range(2)])
    results = []
    for name, op in [('vmap', mapped), ('explicit_loop', loop)]:
        try:
            out, grads = mx.vjp(op, [dst, src], [cot])
            loss = lambda s: (op(dst, s) * cot).sum()
            fd = []
            for i in range(src.size):
                h = [0.] * src.size
                h[i] = 1 / 256
                delta = mx.array(h).reshape(src.shape)
                fd.append(((loss(src + delta) - loss(src - delta)) / (2 / 256)).item())
            tangent = mx.array([[0., 0., 1.], [0., 0., 0.]])
            _, jvp = mx.jvp(lambda s: op(dst, s), [src], [tangent])
            row = dict(name=name, forward=out[0].tolist(), destination_gradient=grads[0].tolist(),
                       source_gradient=grads[1].tolist(), source_coordinate_fd=fd,
                       unused_coordinate_jvp=jvp[0].tolist(),
                       adjoint_lhs=(jvp[0]*cot).sum().item(), adjoint_rhs=(grads[1]*tangent).sum().item())
        except Exception as exc:
            row = dict(name=name,error=repr(exc))
        results.append(row)
        print(json.dumps(row),flush=True)
    descent_cot = mx.array([[1., 2., 3., 4.], [10., -20., 30., 40.]])
    descent_loss = lambda s: (mapped(dst, s) * descent_cot).sum()
    reference_loss = lambda s: (loop(dst, s) * descent_cot).sum()
    bad_gradient = mx.grad(descent_loss)(src)
    good_gradient = mx.grad(reference_loss)(src)
    eta = 1 / 8
    initial = descent_loss(src).item()
    descent = dict(cotangent=descent_cot.tolist(),step_size=eta,initial_loss=initial,
                   mapped_gradient=bad_gradient.tolist(),reference_gradient=good_gradient.tolist(),
                   loss_after_mapped_step=descent_loss(src-eta*bad_gradient).item(),
                   loss_after_reference_step=descent_loss(src-eta*good_gradient).item())
    print(json.dumps({'descent':descent}),flush=True)
    report = dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                  destination=dst.tolist(),source=src.tolist(),mask=mask.tolist(),cotangent=cot.tolist(),
                  expected_source_gradient=[[1.,3.,0.],[20.,40.,0.]],cases=results,descent=descent,
                  cpu_seconds=time.process_time()-started)
    (ROOT/'probe-results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
