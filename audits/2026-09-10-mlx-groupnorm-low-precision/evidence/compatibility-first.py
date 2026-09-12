"""Public test, channel-group membership, dtype controls, and explicit limits."""
import ast
import importlib.metadata
import json
import math
import time
from types import SimpleNamespace
import unittest
from audit_common import *
import numpy as np
started=time.process_time()
rows=[]
test_tree=ast.parse((ROOT/'upstream-python_tests_test_nn.py').read_text())
method=next(n for c in test_tree.body if isinstance(c,ast.ClassDef) for n in c.body
            if isinstance(n,ast.FunctionDef) and n.name=='test_group_norm')
for mode,cls in CLASSES.items():
    env=dict(mx=mx,np=np,nn=SimpleNamespace(GroupNorm=cls))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'upstream-groupnorm-test','exec'),env)
    test=type('PublicGroupNormTest',(unittest.TestCase,),{'test_method':env[method.name]})('test_method')
    result=unittest.TestResult()
    test.run(result)
    add_check(rows,'public/test_group_norm',mode,int(result.wasSuccessful()),1)
    if not result.wasSuccessful():rows[-1]['errors']=[str(e) for _,e in result.errors+result.failures]

    for dtype in [mx.float16,mx.bfloat16,mx.float32,mx.float64]:
        x=mx.array([[[0.,1.,20.,30.,3.,5.]]],dtype=dtype)
        outputs=[]
        for compatible in [False,True]:
            layer=cls(3,6,affine=False,pytorch_compatible=compatible)
            output=layer(x)
            expected,_=reference(x,3,compatible,layer.eps)
            outputs.append(flat(output))
            tol=max(TOL[dtype],5e-6) if compatible else TOL[dtype]
            add_check(rows,f'group_membership/{dtype}/compatible{compatible}',mode,flat(output),expected,tol,tol*1e-3)
        add_check(rows,f'group_membership/{dtype}/modes_are_distinct',mode,int(max(abs(a-b) for a,b in zip(*outputs))>1.),1)
        # G=1 and G=C have identical memberships in both modes.
        for groups in [1,6]:
            a=cls(groups,6,affine=False)(x)
            b=cls(groups,6,affine=False,pytorch_compatible=True)(x)
            add_check(rows,f'equal_membership/{dtype}/G{groups}',mode,flat(a),flat(b),max(TOL[dtype],5e-6),1e-6)

    # A blanket switch to the current fast path would narrow float64 inputs.
    x=mx.array([[[2.**40-2,2.**40-1,2.**40+1,2.**40+2]]],dtype=mx.float64)
    layer=cls(1,4,affine=False)
    expected,_=reference(x,1,False,layer.eps)
    add_check(rows,'float64_default_large_offset',mode,flat(layer(x)),expected,1e-11,1e-13)

    for eps in [0.,-1.,-1e-30]:
        try:cls(2,6,eps=eps);rejected=False
        except ValueError:rejected=True
        add_check(rows,f'negative_or_zero_eps/{eps}',mode,int(rejected),1)

# Existing accepted edge behavior is compared directly, including NaN equality.
def outcome(cls,x,compatible):
    try:
        y=cls(2,x.shape[-1],affine=False,pytorch_compatible=compatible)(x)
        return clean(dict(values=flat(y),shape=list(y.shape),dtype=str(y.dtype)))
    except Exception as e:return dict(error=type(e).__name__,message=str(e))
for dtype in [mx.float16,mx.bfloat16,mx.float32,mx.float64,mx.int32,mx.complex64]:
    for compatible in [False,True]:
        x=mx.array([[[1,2,3,4],[5,6,7,8]]],dtype=dtype)
        # Unmodified dtype/mode combinations must preserve their exact behavior.
        if compatible or dtype not in [mx.float16,mx.bfloat16]:
            add_check(rows,f'unmodified_path/{dtype}/compatible{compatible}','after',
                      int(outcome(CLASSES['before'],x,compatible)==outcome(CLASSES['after'],x,compatible)),1)
for dtype in [mx.float16,mx.bfloat16]:
    for label,x in [('empty',mx.zeros((2,0,6),dtype=dtype)),
                    ('nan',mx.array([[[float('nan'),1.,2.,3.]]],dtype=dtype)),
                    ('inf',mx.array([[[float('inf'),1.,2.,3.]]],dtype=dtype))]:
        add_check(rows,f'edge_behavior/{dtype}/{label}','after',
                  int(outcome(CLASSES['before'],x,False)==outcome(CLASSES['after'],x,False)),1)

# Record failures outside this patch's claimed range, rather than count them as fixed.
limits=[]
for dtype in [mx.bfloat16,mx.float32]:
    x=mx.array([[[-2.**80,2.**80,-2.**80,2.**80]]],dtype=dtype)
    for compatible in [False,True]:
        limits.append(dict(kind='variance_exceeds_float32',dtype=str(dtype),compatible=compatible,
                           output=flat(CLASSES['after'](1,4,affine=False,pytorch_compatible=compatible)(x))))
x=mx.array([[[2.**40-2,2.**40-1,2.**40+1,2.**40+2]]],dtype=mx.float64)
limits.append(dict(kind='unchanged_fast_path_float64_narrowing',
                   output=flat(CLASSES['after'](1,4,affine=False,pytorch_compatible=True)(x)),
                   expected=reference(x,1,True,1e-5)[0]))
result=clean(dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                  cpu_seconds=time.process_time()-started,summary=summary(rows),checks=rows,remaining_limits=limits))
(ROOT/'compatibility-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
for row in result['checks']:
    if row['mode']=='after' and not row['passed']:print(json.dumps(row))
raise SystemExit(bool(result['summary']['after']['failed']))
