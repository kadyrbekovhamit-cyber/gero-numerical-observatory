"""Check sorted left gather gradients using tiny CPU tensors."""
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent


def main():
    started = time.process_time()
    a = mx.array([2., 7.]).reshape(2, 1, 1)
    b = mx.array([3., 5.]).reshape(2, 1, 1)
    indices = mx.array([0, 0], dtype=mx.uint32)
    rows = []
    for side in ('left', 'right'):
        for sorted_indices in (False, True):
            kw = {'lhs_indices' if side == 'left' else 'rhs_indices': indices}
            f = lambda a,b: mx.gather_mm(a,b,sorted_indices=sorted_indices,**kw)
            out, grads = mx.vjp(f, [a,b], [mx.ones((2,1,1))])
            rows.append(dict(side=side, sorted_indices=sorted_indices,
                             forward=out[0].tolist(), gradients=[x.tolist() for x in grads]))
    reference = lambda a,b: mx.take(a,indices,axis=0) @ b
    outputs, gradients = mx.vjp(reference,[a,b],[mx.ones((2,1,1))])
    report = dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                  cases=rows, left_reference_forward=outputs[0].tolist(),
                  left_reference_gradients=[x.tolist() for x in gradients],
                  cpu_seconds=time.process_time()-started)
    (ROOT/'probe-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
