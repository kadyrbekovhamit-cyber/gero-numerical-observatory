"""Actual before/after GroupNorm against scalar forward and gradient references."""
import importlib.metadata
import json
import math
import time
from audit_common import *
started=time.process_time()
rows=[]
forward_scenarios=0
gradient_scenarios=0

def forward_case(name,x,groups,compatible,affine,eps=1e-5):
    global forward_scenarios
    forward_scenarios+=1
    channels=x.shape[-1]
    reference_z,_=reference(x,groups,compatible,eps)
    rounded=mx.array(reference_z,dtype=x.dtype).tolist()
    weights=[1.,-.5,2.,.25,-1.5,.75][:channels] if affine else [1.]*channels
    bias=[.125,-.25,0.,.5,-.125,.25][:channels] if affine else [0.]*channels
    expected=[v*weights[i%channels]+bias[i%channels] for i,v in enumerate(rounded)]
    original=flat(x)
    tol=max(TOL[x.dtype],5e-6) if compatible else TOL[x.dtype]
    for mode,cls in CLASSES.items():
        layer=cls(groups,channels,eps=eps,affine=affine,pytorch_compatible=compatible)
        if affine:
            layer.weight=mx.array(weights)
            layer.bias=mx.array(bias)
        output=layer(x)
        tag=name+f'/G{groups}/compatible{compatible}/affine{affine}'
        add_check(rows,tag+'/output',mode,flat(output),expected,tol,tol*1e-3)
        expected_dtype=mx.result_type(x.dtype,mx.float32) if affine else x.dtype
        add_check(rows,tag+'/dtype',mode,int(output.dtype==expected_dtype),1)
        add_check(rows,tag+'/shape',mode,int(output.shape==x.shape),1)
        add_check(rows,tag+'/input_unchanged',mode,int(flat(x)==original),1)

for dtype in [mx.float16,mx.bfloat16,mx.float32,mx.float64]:
    for shape in [(2,6),(2,3,6),(2,2,3,6)]:
        for groups in [1,2,3]:
            for amplitude in [1/32,128.]:
                x=input_data(shape,dtype,amplitude)
                for compatible in [False,True]:
                    for affine in [False,True]:
                        forward_case(f'{dtype}/{shape}/a{amplitude}',x,groups,compatible,affine)

# Strides, broadcasted batches, small positive epsilon, and singleton groups.
for dtype in [mx.float16,mx.bfloat16]:
    base=input_data((2,2,3,6),dtype,128.)
    variants={'spatial_transpose':base.transpose(0,2,1,3),'channel_reverse':base[...,::-1],
              'broadcast_batch':mx.broadcast_to(base[:1],base.shape)}
    for label,x in variants.items():
        for compatible in [False,True]:forward_case(f'{dtype}/{label}',x,3,compatible,True)
    for eps in [1e-3,1e-5,1e-8,1e-12]:
        for compatible in [False,True]:
            forward_case(f'{dtype}/constant/eps{eps}',mx.full((2,2,6),3.,dtype=dtype),3,compatible,False,eps)
            forward_case(f'{dtype}/singleton/eps{eps}',input_data((2,6),dtype,128.),6,compatible,True,eps)

# Analytic input/gamma/beta gradients, and forward-mode directional derivatives.
for dtype in [mx.float16,mx.bfloat16,mx.float32,mx.float64]:
    for groups in [1,2,3]:
        for compatible in [False,True]:
            gradient_scenarios+=1
            amplitude=128.
            x=input_data((2,3,6),dtype,amplitude)
            z,blocks=reference(x,groups,compatible,1e-5)
            upstream=[((i*5)%9-4)*.25 for i in range(x.size)]
            direction=[((i*3)%7-3)*.5 for i in range(x.size)]
            weights=[1.,-.5,2.,.25,-1.5,.75]
            rounded_z=mx.array(z,dtype=dtype).tolist()
            gamma=[sum(upstream[i]*rounded_z[i] for i in range(x.size) if i%6==c) for c in range(6)]
            beta=[sum(upstream[i] for i in range(x.size) if i%6==c) for c in range(6)]
            dx=[0.]*x.size
            jvp=[0.]*x.size
            for block in blocks:
                indices,zn,inv=block['indices'],block['z'],block['inverse_std']
                n=len(indices)
                v=[upstream[i]*weights[i%6] for i in indices]
                d=[direction[i] for i in indices]
                mv,md=sum(v)/n,sum(d)/n
                mvz=sum(a*b for a,b in zip(v,zn))/n
                mdz=sum(a*b for a,b in zip(d,zn))/n
                for k,i in enumerate(indices):
                    dx[i]=inv*(v[k]-mv-zn[k]*mvz)
                    jvp[i]=weights[i%6]*inv*(d[k]-md-zn[k]*mdz)
            for mode,cls in CLASSES.items():
                layer=cls(groups,6,pytorch_compatible=compatible)
                layer.weight=mx.array(weights)
                layer.bias=mx.array([.125,-.25,0.,.5,-.125,.25])
                w=mx.array(upstream,dtype=dtype).reshape(x.shape)
                loss=lambda model,x:mx.sum(model(x)*w)
                _,grads=nn.value_and_grad(layer,loss)(layer,x)
                found_dx=mx.grad(lambda x:mx.sum(layer(x)*w))(x)
                _,found_jvp=mx.jvp(layer,[x],[mx.array(direction,dtype=dtype).reshape(x.shape)])
                tol=max(TOL[dtype]*3,2e-5 if compatible else 0)
                tag=f'gradient/{dtype}/G{groups}/compatible{compatible}'
                add_check(rows,tag+'/gamma',mode,flat(grads['weight']),mx.array(gamma,dtype=grads['weight'].dtype).tolist(),tol,tol*1e-3)
                add_check(rows,tag+'/beta',mode,flat(grads['bias']),mx.array(beta,dtype=grads['bias'].dtype).tolist(),tol,tol*1e-3)
                floor=2**-23 if dtype==mx.float16 else 1e-10 if dtype==mx.bfloat16 else 1e-11 if compatible else 1e-14
                vjp_floor=jvp_floor=floor
                if dtype==mx.float32 or (compatible and dtype==mx.float64):
                    vjp_floor=max(floor,8*float(mx.finfo(mx.float32).eps)*max(abs(v) for v in dx))
                    jvp_floor=max(floor,8*float(mx.finfo(mx.float32).eps)*max(abs(v) for v in jvp))
                add_check(rows,tag+'/input_vjp',mode,flat(found_dx),mx.array(dx,dtype=dtype).tolist(),tol,vjp_floor)
                add_check(rows,tag+'/input_jvp',mode,flat(found_jvp[0]),jvp,tol,jvp_floor)

result=clean(dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                  cpu_seconds=time.process_time()-started,forward_scenarios=forward_scenarios,
                  gradient_scenarios=gradient_scenarios,summary=summary(rows),checks=rows))
(ROOT/'regression-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
for row in result['checks']:
    if row['mode']=='after' and not row['passed']:print(json.dumps(row))
raise SystemExit(bool(result['summary']['after']['failed']))
