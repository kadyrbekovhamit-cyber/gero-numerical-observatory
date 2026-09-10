"""Exact public tests plus small state, dtype and endpoint checks."""
import ast
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time
from types import SimpleNamespace
import unittest
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent
def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatchNorm
classes={"before":load("before_compat","upstream-python_mlx_nn_layers_normalization.py"),
         "after":load("after_compat","patched-normalization.py")}
started=time.process_time()
rows=[]
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return "NaN" if math.isnan(x) else "Infinity" if x>0 else "-Infinity"
    return x
def check(name,mode,actual,expected,atol=0,rtol=0):
    if not isinstance(actual,list):actual,expected=[actual],[expected]
    ok=len(actual)==len(expected) and all(math.isfinite(a) and abs(a-b)<=atol+rtol*abs(b) for a,b in zip(actual,expected))
    rows.append(dict(name=name,mode=mode,passed=ok,actual=actual,expected=expected,atol=atol,rtol=rtol))
def flat(x):return x.reshape(-1).tolist()

test_ast=ast.parse((ROOT/"upstream-python_tests_test_nn.py").read_text())
methods=[n for klass in test_ast.body if isinstance(klass,ast.ClassDef) for n in klass.body
         if isinstance(n,ast.FunctionDef) and n.name in ("test_batch_norm","test_batch_norm_stats")]
assert len(methods)==2
for mode,cls in classes.items():
    for method in methods:
        env=dict(mx=mx,nn=SimpleNamespace(BatchNorm=cls))
        exec(compile(ast.Module(body=[method],type_ignores=[]),"public-batchnorm-tests","exec"),env)
        case=type("PublicBatchNormTest",(unittest.TestCase,),{"test_method":env[method.name]})("test_method")
        result=unittest.TestResult()
        case.run(result)
        check("upstream/"+method.name,mode,int(result.wasSuccessful()),1)
        if not result.wasSuccessful():
            rows[-1]["errors"]=[str(e) for _,e in result.errors+result.failures]

    for mu in [0.,.1,.125,1.]:
        layer=cls(1,momentum=mu,affine=False)
        x=mx.array([[-256.],[256.]],dtype=mx.float16)
        output=layer(x)
        state=1-mu+mu*131072
        tag=f"momentum/{mu}"
        check(tag+"/train",mode,flat(output),[-1.,1.],atol=.001)
        check(tag+"/state",mode,flat(layer.running_var),[state],rtol=2e-6)
        layer(mx.array([[-1.],[1.]],dtype=mx.float16))
        state=(1-mu)*state+mu*2
        check(tag+"/next_state",mode,flat(layer.running_var),[state],rtol=2e-6)
        layer.eval()
        output=layer(mx.array([[-1.],[1.]],dtype=mx.float16))
        check(tag+"/eval",mode,flat(output),[-1/math.sqrt(state+layer.eps),1/math.sqrt(state+layer.eps)],rtol=3e-6)

    for eps in [1e-3,1e-5,1e-8,1e-12]:
        layer=cls(2,eps=eps,affine=False,track_running_stats=False)
        output=layer(mx.full((3,2),3.,dtype=mx.float16))
        check(f"zero_variance/eps{eps}",mode,flat(output),[0.]*6)

    for shape in [(2,),(2,1,1,1,1),(1,2)]:
        layer=cls(2)
        try:
            layer(mx.zeros(shape))
            rejected=False
        except ValueError:rejected=True
        check(f"invalid_training_shape/{shape}",mode,int(rejected),1)

    for dtype in [mx.float16,mx.bfloat16]:
        layer=cls(1,momentum=.125,affine=False)
        # Explicitly pre-cast saved buffers, then check the prototype's policy.
        layer.running_mean=layer.running_mean.astype(dtype)
        layer.running_var=layer.running_var.astype(dtype)
        layer(mx.array([[-256.],[256.]],dtype=dtype))
        if mode=="after":
            check(f"precast_buffers/{dtype}/promoted",mode,int(layer.running_var.dtype==mx.float32),1)
            check(f"precast_buffers/{dtype}/value",mode,flat(layer.running_var),[16384.875],rtol=2e-6)
        layer.eval()
        query=mx.array([[-1.],[1.]],dtype=dtype)
        output=layer(query)
        with tempfile.TemporaryDirectory(prefix="checkpoint-",dir=ROOT) as tmp:
            path=str(Path(tmp)/"state.npz")
            layer.save_weights(path)
            restored=cls(1,momentum=.125,affine=False)
            restored.load_weights(path)
            restored.eval()
            check(f"checkpoint/{dtype}/roundtrip",mode,
                  int(clean(flat(restored(query)))==clean(flat(output))),1)
            check(f"checkpoint/{dtype}/state_shape",mode,int(restored.running_var.shape==(1,)),1)

# Probe the explicit remaining limitation: widening alone cannot protect
# float32/bfloat16 statistics whose true variance exceeds float32 range.
limits=[]
for dtype in [mx.bfloat16,mx.float32]:
    layer=classes['after'](1,affine=False)
    output=layer(mx.array([[-2.**80],[2.**80]],dtype=dtype))
    limits.append(dict(dtype=str(dtype),output=flat(output),running_var=flat(layer.running_var),
                       note="Outside repair scope: the variance exceeds float32 range."))

summary={mode:dict(checks=sum(r['mode']==mode for r in rows),failed=sum(r['mode']==mode and not r['passed'] for r in rows))
         for mode in classes}
result=clean(dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
                  cpu_seconds=time.process_time()-started,summary=summary,checks=rows,remaining_limits=limits))
(ROOT/'compatibility-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
for row in result['checks']:
    if row['mode']=='after' and not row['passed']:print(json.dumps(row))
raise SystemExit(bool(summary['after']['failed']))
