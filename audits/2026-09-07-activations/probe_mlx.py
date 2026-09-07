import importlib.util
import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from load_mlx import SOURCE_ROOT
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='mlx-probe-current.json')
args = parser.parse_args()
root = SOURCE_ROOT / 'mlx'
spec = importlib.util.spec_from_file_location('audited_activations', root / 'python/mlx/nn/layers/activations.py')
activations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activations)

def clean(x):
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return str(x)
    return x

record = []
for device in [mx.cpu, mx.gpu]:
    mx.set_default_device(device)
    for dtype in [mx.float16, mx.float32]:
        for name in ['elu', 'selu', 'gelu', 'gelu_approx', 'softplus', 'mish']:
            values = [-100., -2., -.01, 0., .01, 2., 100., 40000.]
            x = mx.array(values, dtype=dtype)
            fn = getattr(activations, name)
            try:
                output = fn(x).tolist()
                grad = mx.grad(lambda t: fn(t).sum())(x).tolist()
                row={'device':str(device),'dtype':str(dtype),'function':name,'input':x.tolist(),'output':output,'gradient':grad}
            except Exception as error:
                row={'device':str(device),'dtype':str(dtype),'function':name,'error':str(error)}
            record.append(clean(row))
            print(json.dumps(clean(row)), flush=True)
Path(__file__).with_name(args.output).write_text(json.dumps(record,indent=2)+'\n')
