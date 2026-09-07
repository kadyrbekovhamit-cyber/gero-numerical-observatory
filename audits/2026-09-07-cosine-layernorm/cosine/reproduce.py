"""Minimal reproduction using official MLX, with an optional local workaround."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import mlx.core as mx
import mlx.nn as nn
from stable_cosine import cosine_similarity_loss as repaired

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='reproduction.json')
args=parser.parse_args()
rows=[]
for device in ['cpu','gpu']:
    mx.set_default_device(getattr(mx,device))
    for name,dtype,a_values,b_values in [
        ('float16_self',mx.float16,[1000.,-1000.,1000.,-1000.],[1000.,-1000.,1000.,-1000.]),
        ('float32_self',mx.float32,[1e20,-1e20,1e20,-1e20],[1e20,-1e20,1e20,-1e20]),
        ('zero_vector',mx.float32,[0.,0.,0.,0.],[1e-5,2e-5,-1e-5,1e-5]),
    ]:
        a=mx.array([a_values],dtype);b=mx.array([b_values],dtype)
        row={'case':name,'device':device,'dtype':str(dtype),'input_a':a.tolist(),'input_b':b.tolist()}
        for label,function in [('original',nn.losses.cosine_similarity_loss),('repaired',repaired)]:
            value=function(a,b)
            gradient=mx.grad(lambda x: mx.sum(function(x,b)))(a)
            mx.eval(value,gradient)
            row[label]={'value':value.astype(mx.float32).tolist(),'gradient_a':gradient.astype(mx.float32).tolist()}
        row['expected_value']=0. if name=='zero_vector' else 1.
        row['expected_gradient_a']=[[float(x)/1e-8 for x in b.tolist()[0]]] if name=='zero_vector' else [[0.,0.,0.,0.]]
        rows.append(row)
data={'environment':{'mlx':importlib.metadata.version('mlx'),'mlx_metal':importlib.metadata.version('mlx-metal'),'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine()},'rows':rows}
Path(args.output).write_text(json.dumps(data,indent=2)+'\n')
for row in rows:print(json.dumps(row))
