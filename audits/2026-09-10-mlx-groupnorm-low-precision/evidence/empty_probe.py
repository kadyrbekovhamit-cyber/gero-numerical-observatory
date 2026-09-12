"""Isolated empty-input diagnostic, bounded to two CPU seconds."""
import sys
from audit_common import *
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
mode=sys.argv[1]
print('mode',mode,'construct',flush=True)
x=mx.zeros((2,0,6),dtype=mx.float16)
layer=CLASSES[mode](2,6,affine=False)
print('mode',mode,'call',flush=True)
y=layer(x)
print('mode',mode,'eval',y.shape,flush=True)
mx.eval(y)
print('mode',mode,'done',y.tolist(),flush=True)
