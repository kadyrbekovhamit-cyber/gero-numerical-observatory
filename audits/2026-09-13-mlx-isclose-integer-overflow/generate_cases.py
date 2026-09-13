"""Deterministic integer oracle using exact rational tolerances, not MLX."""
from pathlib import Path
from fractions import Fraction
import json,math
R=Path(__file__).resolve().parent
cases=[]
bounds={'bool':(0,1),'int8':(-128,127),'uint8':(0,255),'int16':(-32768,32767),'uint16':(0,65535),'int32':(-(2**31),2**31-1),'uint32':(0,2**32-1)}
tols=[(0.,0.),(0.,1.),(0.,128.),(1/128,0.),(.5,0.),(1.,0.)]
def oracle(a,b,rt,at):
 if isinstance(a,int) and isinstance(b,int):
  q=Fraction.from_float(rt);t=Fraction.from_float(at)
  return abs(a-b)*q.denominator*t.denominator <= t.numerator*q.denominator+q.numerator*abs(b)*t.denominator
 if math.isnan(a) or math.isnan(b): return False
 if math.isinf(a) or math.isinf(b): return a==b
 return abs(a-b)<=at+rt*abs(b)
def add(name,da,db,av,bv,rt,at,group='integer',shape=None,transpose=False,equal_nan=False):
 expected=[(equal_nan and isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b)) or oracle(a,b,rt,at) for a in av for b in bv]
 def encode(x):return str(x) if isinstance(x,float) and not math.isfinite(x) else x
 cases.append(dict(name=name,group=group,da=da,db=db,a=list(map(encode,av)),b=list(map(encode,bv)),rtol=rt,atol=at,expected=expected,transpose=transpose,equal_nan=equal_nan))
for name,dt,a,b,rt,at in [
 ('minimal-unsigned-small-gap','uint8',0,1,0.,1.),
 ('minimal-unsigned-wrap','uint8',0,255,0.,1.),
 ('minimal-signed-min8','int8',-128,0,0.,0.),
 ('minimal-signed-min32','int32',-(2**31),0,0.,0.),
 ('minimal-negative-reference','int8',-127,-128,.01,0.),
 ('float32-cast-regression-guard','int32',2**24+1,2**24,0.,0.)]:
 add(name,dt,dt,[a],[b],rt,at)
for dtype in ['int8','uint8']:
 lo,hi=bounds[dtype];v=list(range(lo,hi+1))
 for rt,at in tols:add('exhaustive-'+dtype+'-'+str(rt)+'-'+str(at),dtype,dtype,v,v,rt,at)
values={}
for dtype,(lo,hi) in bounds.items():
 v=sorted(set(x for x in [lo,lo+1,-32768,-129,-128,-127,-2,-1,0,1,2,127,128,255,256,32767,2**24-1,2**24,2**24+1,hi-1,hi] if lo<=x<=hi)); values[dtype]=v
for da in bounds:
 for db in bounds:
  for rt,at in tols:add(f'boundary-{da}-{db}-{rt}-{at}',da,db,values[da],values[db],rt,at)
for dtype in ['int8','uint8','int16','uint16','int32','uint32']:
 for rt,at in [(0.,1.),(1/128,0.)]:
  add(f'noncontiguous-{dtype}-{rt}',dtype,dtype,values[dtype],values[dtype],rt,at,transpose=True)
for dtype in ['float16','bfloat16','float32','float64']:
 for rt,at in [(0.,0.),(0.,1.),(.5,0.)]:
  for eq in [False,True]:add(f'float-control-{dtype}-{rt}-{at}-{eq}',dtype,dtype,[-math.inf,-2.,-1.,0.,1.,2.,math.inf,math.nan],[-math.inf,-2.,-1.,0.,1.,2.,math.inf,math.nan],rt,at,'float_control',equal_nan=eq)
for dtype in ['int32','uint32','float32']:
 add('empty-'+dtype,dtype,dtype,[],[1],0.,0.,'empty_control')
(R/'cases.json').write_text(json.dumps(cases,separators=(',',':'),allow_nan=False)+'\n')
summary={'scenarios':len(cases),'element_comparisons':sum(len(c['expected']) for c in cases),'oracle':'Exact Python integers and Fraction.from_float for integer cases; documented special-value semantics for float controls','full_pair_domains':['int8 x int8','uint8 x uint8'],'tolerances':tols,'wide_integer_scope':'64-bit types excluded from correction suite and checked separately as limitations'}
(R/'oracle-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary)
