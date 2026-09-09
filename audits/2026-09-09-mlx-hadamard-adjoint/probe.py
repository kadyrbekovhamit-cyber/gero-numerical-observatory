"""CPU-only Hadamard adjoint and energy checks on the official wheel."""
import importlib.metadata
import json
import math
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
    start = time.process_time()
    rows = []
    for n in (4, 12, 20, 28, 40, 56):
        x = mx.array([1.] + [0.] * (n - 1))
        cot = mx.array([((k * 3) % 7 - 3) / 4 for k in range(n)])
        f = mx.hadamard_transform
        h_transpose = f(mx.eye(n))
        expected = cot @ h_transpose.T
        actual = mx.vjp(f, [x], [cot])[1][0]
        loss = lambda x: 0.5 * mx.sum(mx.square(f(x)))
        gradient = mx.grad(loss)(x)
        eps = 1 / 512
        e0 = x
        rows.append(dict(n=n, adjoint_max_error=mx.max(mx.abs(actual-expected)).item(),
                         energy=loss(x).item(), expected_energy=0.5,
                         energy_gradient=gradient.tolist(), expected_energy_gradient=x.tolist(),
                         first_coordinate_fd=((loss(x+eps*e0)-loss(x-eps*e0))/(2*eps)).item(),
                         learning_rate=0.125,
                         energy_after_step=loss(x-0.125*gradient).item(),
                         expected_energy_after_step=0.5*(1-0.125)**2))
    report=dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                cases=rows,cpu_seconds=time.process_time()-start)
    (ROOT/'probe-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if 'gradient' not in k} |
                      {'first_gradient':row['energy_gradient'][0]} for row in rows],indent=2))


if __name__ == '__main__':
    main()
