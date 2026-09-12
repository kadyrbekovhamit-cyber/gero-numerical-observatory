"""Minimal paired reproduction using the frozen public and patched classes."""
import json
import time
from audit_common import *
started=time.process_time()
x=mx.array([[[-128.,128.,-128.,128.]]],dtype=mx.float16)
rows=[]
for mode,cls in CLASSES.items():
    y=cls(1,4,affine=False)(x)
    rows.append(dict(mode=mode,output=flat(y),dtype=str(y.dtype)))
result=dict(device=str(mx.default_device()),cpu_seconds=time.process_time()-started,
            input=flat(x),expected=[-1.,1.,-1.,1.],cases=rows)
(ROOT/'paired-reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert rows[1]['output']==result['expected']
