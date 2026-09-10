"""Independent scalar references against frozen before/after MLX classes."""
from decimal import Decimal, localcontext
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatchNorm
BEFORE = load("audit_before", "upstream-python_mlx_nn_layers_normalization.py")
AFTER = load("audit_after", "patched-normalization.py")
TOL = {mx.float16: .004, mx.bfloat16: .025, mx.float32: 6e-6, mx.float64: 1e-11}
EPS = 1e-5
MU = .125
rows = []
scenarios = []
started = time.process_time()

def clean(x):
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x>0 else "-Infinity"
    return x

def check(name,mode,actual,expected,rtol=0,atol=0):
    if not isinstance(actual,list): actual,expected=[actual],[expected]
    ok=len(actual)==len(expected) and all(math.isfinite(a) and abs(a-b)<=atol+rtol*abs(b) for a,b in zip(actual,expected))
    rows.append(dict(name=name,mode=mode,passed=ok,actual=actual,expected=expected,rtol=rtol,atol=atol))

def flat(x): return x.reshape(-1).tolist()

def stats(matrix):
    # Reference calculations only use scalar Decimal arithmetic.
    n,c=len(matrix),len(matrix[0])
    with localcontext() as ctx:
        ctx.prec=60
        data=[[Decimal.from_float(float(v)) for v in row] for row in matrix]
        means=[sum(row[j] for row in data)/n for j in range(c)]
        variances=[sum((row[j]-means[j])**2 for row in data)/n for j in range(c)]
        return [float(m) for m in means],[float(v) for v in variances]

def data(shape,dtype,amplitude,offset=0.):
    n,c=math.prod(shape[:-1]),shape[-1]
    pattern=[-1.,0.,2.,-2.,1.,3.]
    values=[[amplitude*pattern[i%6]*(j+1)+offset*(j+1) for j in range(c)] for i in range(n)]
    return mx.array(values,dtype=dtype).reshape(shape)

def normalized(matrix,mean,var,dtype,weight,bias):
    values=[(row[j]-mean[j])/math.sqrt(var[j]+EPS) for row in matrix for j in range(len(mean))]
    # This cast represents the public low-precision normalization output.
    values=mx.array(values,dtype=dtype).tolist()
    return [v*weight[i%len(mean)]+bias[i%len(mean)] for i,v in enumerate(values)]

for dtype in [mx.float16,mx.bfloat16,mx.float32,mx.float64]:
    for shape in [(3,2),(2,3,2),(2,2,3,2)]:
        for amplitude in [1/128,1.,256.,8192.]:
            for affine in [False,True]:
                for track in [False,True]:
                    name=f"{dtype}/{shape}/a{amplitude}/affine{affine}/track{track}"
                    scenarios.append(name)
                    layers={mode:cls(shape[-1],eps=EPS,momentum=MU,affine=affine,track_running_stats=track)
                            for mode,cls in [("before",BEFORE),("after",AFTER)]}
                    weights=[1.5,-.75] if affine else [1.,1.]
                    biases=[.25,.125] if affine else [0.,0.]
                    for layer in layers.values():
                        if affine:
                            layer.weight=mx.array(weights)
                            layer.bias=mx.array(biases)
                    running_mean=[0.,0.]
                    running_var=[1.,1.]
                    for stage,(amp,offset) in enumerate([(amplitude,0.),(.25,2.)]):
                        x=data(shape,dtype,amp,offset)
                        values=x.reshape(-1,shape[-1]).tolist()
                        mean,var=stats(values)
                        n=len(values)
                        running_mean=[(1-MU)*old+MU*m for old,m in zip(running_mean,mean)]
                        running_var=[(1-MU)*old+MU*v*n/(n-1) for old,v in zip(running_var,var)]
                        expected=normalized(values,mean,var,dtype,weights,biases)
                        for mode,layer in layers.items():
                            out=layer(x)
                            tag=name+f"/train{stage}"
                            tol=TOL[dtype]
                            check(tag+"/output",mode,flat(out),expected,tol,tol*1e-3)
                            wanted_dtype=mx.result_type(dtype,mx.float32) if affine else dtype
                            check(tag+"/dtype",mode,int(out.dtype==wanted_dtype),1)
                            check(tag+"/shape",mode,int(out.shape==x.shape),1)
                            check(tag+"/input_unchanged",mode,int(values==x.reshape(-1,shape[-1]).tolist()),1)
                            if track:
                                check(tag+"/running_mean",mode,flat(layer.running_mean),running_mean,tol,tol*1e-3)
                                check(tag+"/running_var",mode,flat(layer.running_var),running_var,tol,tol*1e-3)
                                check(tag+"/state_dtype",mode,int(layer.running_var.dtype==(mx.float64 if dtype==mx.float64 else mx.float32)),1)
                    x=data(shape,dtype,1.,-.5)
                    values=x.reshape(-1,shape[-1]).tolist()
                    mean,var=(running_mean,running_var) if track else stats(values)
                    calc_dtype=(mx.float64 if dtype==mx.float64 else mx.float32) if track else dtype
                    expected=normalized(values,mean,var,calc_dtype,weights,biases)
                    for mode,layer in layers.items():
                        layer.eval()
                        old_state=[flat(layer.running_mean),flat(layer.running_var)] if track else None
                        out=layer(x)
                        check(name+"/eval/output",mode,flat(out),expected,TOL[dtype],TOL[dtype]*1e-3)
                        check(name+"/eval/dtype",mode,int(out.dtype==(mx.result_type(calc_dtype,mx.float32) if affine else calc_dtype)),1)
                        state=[flat(layer.running_mean),flat(layer.running_var)] if track else None
                        check(name+"/eval/state_unchanged",mode,int(state==old_state),1)

# Input, gamma, beta gradients in a genuine nn.value_and_grad call.
gradient_cases=0
for dtype in [mx.float16,mx.bfloat16,mx.float32,mx.float64]:
    for amplitude in [1.,256.,4096.]:
        x=data((3,2),dtype,amplitude)
        matrix=x.tolist()
        means,variances=stats(matrix)
        upstream=[[-.5,2.],[2.,-.25],[-1.,1.]]
        weights=[1.5,-.75]
        gamma_grad=[sum(upstream[i][j]*(matrix[i][j]-means[j])/math.sqrt(variances[j]+EPS) for i in range(3)) for j in range(2)]
        beta_grad=[sum(upstream[i][j] for i in range(3)) for j in range(2)]
        dx=[]
        for i in range(3):
            for j in range(2):
                inv=1/math.sqrt(variances[j]+EPS)
                v=[upstream[k][j]*weights[j] for k in range(3)]
                z=[(matrix[k][j]-means[j])*inv for k in range(3)]
                dx.append(inv*(v[i]-sum(v)/3-z[i]*sum(v[k]*z[k] for k in range(3))/3))
        for mode,cls in [("before",BEFORE),("after",AFTER)]:
            layer=cls(2,eps=EPS,momentum=MU)
            layer.weight=mx.array(weights)
            layer.bias=mx.array([.25,.125])
            w=mx.array(upstream,dtype=dtype)
            loss=lambda model,input:mx.sum(model(input)*w)
            value,grads=nn.value_and_grad(layer,loss)(layer,x)
            input_layer=cls(2,eps=EPS,track_running_stats=False)
            input_layer.weight=mx.array(weights)
            input_layer.bias=mx.array([.25,.125])
            found_dx=mx.grad(lambda z:mx.sum(input_layer(z)*w))(x)
            tag=f"gradient/{dtype}/a{amplitude}"
            tol=TOL[dtype]*3
            check(tag+"/gamma",mode,flat(grads["weight"]),mx.array(gamma_grad,dtype=grads["weight"].dtype).tolist(),tol,tol*1e-3)
            check(tag+"/beta",mode,flat(grads["bias"]),mx.array(beta_grad,dtype=grads["bias"].dtype).tolist(),tol,tol*1e-3)
            check(tag+"/input",mode,flat(found_dx),mx.array(dx,dtype=dtype).tolist(),tol,
                  2**-23 if dtype==mx.float16 else 1e-11 if dtype==mx.bfloat16 else 1e-12)
            check(tag+"/frozen_state",mode,int("running_var" not in grads and "running_mean" not in grads),1)
            mean_ref=[MU*m for m in means]
            var_ref=[1-MU+MU*v*1.5 for v in variances]
            check(tag+"/state_mean",mode,flat(layer.running_mean),mean_ref,TOL[dtype],1e-6)
            check(tag+"/state_var",mode,flat(layer.running_var),var_ref,TOL[dtype],1e-6)
        gradient_cases+=1

summary={mode:dict(checks=sum(r['mode']==mode for r in rows),failed=sum(r['mode']==mode and not r['passed'] for r in rows))
         for mode in ['before','after']}
result=clean(dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
                  cpu_seconds=time.process_time()-started,forward_scenarios=len(scenarios),gradient_scenarios=gradient_cases,
                  summary=summary,checks=rows))
(ROOT/"regression-results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
for row in result['checks']:
    if row['mode']=='after' and not row['passed']:print(json.dumps(row))
raise SystemExit(bool(summary['after']['failed']))
