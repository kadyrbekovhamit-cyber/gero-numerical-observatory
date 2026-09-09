"""Probe transpose=False affine parameter VJPs on tiny synthetic CPU tensors."""
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent

def main():
    started=time.process_time()
    rows=[]
    for K,N in ((32,32),(32,64)):
        x=mx.array([1.]+[0.]*(K-1)).reshape(1,1,K)
        for trans in (False,True):
            shape=(1,N,K//8) if trans else (1,K,N//8)
            sb_shape=(1,N,K//32) if trans else (1,K,N//32)
            w=mx.full(shape,0x11111111,dtype=mx.uint32)
            scales=mx.ones(sb_shape)
            biases=mx.zeros(sb_shape)
            cot=mx.ones((1,1,N))
            ids=mx.array([0],dtype=mx.uint32)
            for op in ('gather','ordinary'):
                for sorted_indices in ((False,True) if op=='gather' else (False,)):
                    row=dict(K=K,N=N,transpose=trans,op=op,sorted_indices=sorted_indices)
                    if op=='gather':
                        f=lambda x,s,b:mx.gather_qmm(x,w,s,b,rhs_indices=ids,
                                transpose=trans,group_size=32,bits=4,sorted_indices=sorted_indices)
                    else:
                        f=lambda x,s,b:mx.quantized_matmul(x,w,s,b,transpose=trans,group_size=32,bits=4)
                    expected_sb=([1.]*(N*(K//32)) if trans else
                                 [32.]*(N//32)+[0.]*((K-1)*(N//32)))
                    try:
                        y,g=mx.vjp(f,[x,scales,biases],[cot])
                        mx.eval(*y,*g)
                        row.update(forward_first=y[0].reshape(-1)[:3].tolist(),
                                   dx_first=g[0].reshape(-1)[:3].tolist(),
                                   ds_shape=list(g[1].shape),db_shape=list(g[2].shape),
                                   ds=g[1].reshape(-1).tolist(),db=g[2].reshape(-1).tolist(),
                                   expected_shape=list(sb_shape),expected_sb=expected_sb)
                        # Direct forward finite differences for two affine parameters.
                        fd=[]
                        for coord in (0,N//32):
                            direction=mx.array([float(i==coord) for i in range(scales.size)]).reshape(sb_shape)
                            eps=1/1024
                            diff=(f(x,scales+eps*direction,biases).sum()-
                                  f(x,scales-eps*direction,biases).sum())/(2*eps)
                            fd.append(dict(index=coord,ds=diff.item()))
                        row['finite_differences']=fd
                    except Exception as e:row['error']=str(e)
                    rows.append(row)
    result=dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                cases=rows,cpu_seconds=time.process_time()-started)
    (ROOT/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print({k:v for k,v in r.items() if k not in ('ds','db','expected_sb')},
              'first ds',r.get('ds',[])[:4], 'expected',r.get('expected_sb',[])[:4])
    print('cpu_seconds',result['cpu_seconds'])

if __name__=='__main__':main()
