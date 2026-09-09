"""Tiny CPU checks; no quantization approximation in the reference weights."""
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
    ids = mx.array([0, 0], dtype=mx.uint32)
    x = mx.broadcast_to(mx.array([2., 7.]).reshape(2, 1, 1), (2, 1, 32))
    # Every packed 4-bit integer is 1. Each dequantized matrix is exactly
    # constant 3 or 5: q*scale+bias, with unit scales.
    w = mx.full((2, 32, 4), 0x11111111, dtype=mx.uint32)
    s = mx.ones((2, 32, 1))
    bias = mx.broadcast_to(mx.array([2., 4.]).reshape(2, 1, 1), (2, 32, 1))
    cot = mx.ones((2, 1, 32))
    cases = []
    for side in ('left', 'right'):
        for sorted_indices in (False, True):
            kw = {'lhs_indices' if side == 'left' else 'rhs_indices': ids}
            f = lambda x,s,b: mx.gather_qmm(x,w,s,b,group_size=32,bits=4,
                                           sorted_indices=sorted_indices,**kw)
            y,g = mx.vjp(f,[x,s,bias],[cot])
            expected_x = [256.,0.] if side == 'left' else [96.,96.]
            expected_sb = [64.,64.] if side == 'left' else [288.,0.]
            expected = [mx.broadcast_to(mx.array(expected_x).reshape(2,1,1),x.shape),
                        mx.broadcast_to(mx.array(expected_sb).reshape(2,1,1),s.shape)]
            cases.append(dict(side=side, sorted_indices=sorted_indices,
                              output_first=[v[0][0] for v in y[0].tolist()],
                              gradient_first=[z.reshape(2,-1)[:,0].tolist() for z in g],
                              expected_gradient_first=[expected_x,expected_sb,expected_sb],
                              max_errors=[mx.max(mx.abs(z-e)).item()
                                          for z,e in zip(g,[expected[0],expected[1],expected[1]])]))
    a = mx.array([2.,7.]).reshape(2,1,1)
    b = mx.array([3.,5.]).reshape(2,1,1)
    def loss(a, sorted_indices):
        y = mx.gather_mm(a,b,lhs_indices=ids,sorted_indices=sorted_indices).reshape(2)
        return .5*(y[0]-y[1])**2
    loss_cases=[]
    for sorted_indices in (False,True):
        fn=lambda a:loss(a,sorted_indices)
        g=mx.grad(fn)(a)
        eps=1/1024
        finite_differences=[]
        for i in range(2):
            direction=mx.array([float(i==j) for j in range(2)]).reshape(a.shape)
            finite_differences.append(((fn(a+eps*direction)-fn(a-eps*direction))/(2*eps)).item())
        updated=a-g/32
        loss_cases.append(dict(sorted_indices=sorted_indices,loss=fn(a).item(),
                               gradient=g.reshape(2).tolist(),finite_differences=finite_differences,
                               updated_a=updated.reshape(2).tolist(),new_loss=fn(updated).item()))
    report=dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                quantized=cases,loss_step=loss_cases,cpu_seconds=time.process_time()-started)
    (ROOT/'quantized-and-loss-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
