"""High-precision independent first-derivative screen for native MLX CPU."""
import csv,hashlib,json,math,struct,subprocess,sys,datetime
from pathlib import Path
import mpmath as mp
A=Path(__file__).resolve().parent

def f32(x):
 try:return struct.unpack('<f',struct.pack('<f',float(x)))[0]
 except OverflowError:return math.copysign(math.inf,float(x))
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def value(x):return struct.unpack('<f',struct.pack('<I',int(x)))[0]
def exact(x):
 n,d=x.as_integer_ratio();return mp.mpf(n)/d

def rounded(v):
 if v==0:return 0.0
 a=abs(v);sgn=-1 if v<0 else 1
 exp=int(mp.floor(mp.log(a,2)));q=mp.mpf(2)**max(exp-23,-149)
 r=mp.nint(a/q)*q
 if r>mp.mpf(2)**128-mp.mpf(2)**104:return math.copysign(math.inf,sgn)
 return f32(sgn*r)

def generate(dps):
 mp.mp.dps=dps
 seeds=[0.,-0.,1.,-1.,.125,-.125,1e30,-1e30,2.5e38,-2.5e38,2.9e38,-2.9e38,3e38,-3e38,3.05e38,-3.05e38,3.1e38,-3.1e38,3.2e38,-3.2e38,value(0x7f7fffff),-value(0x7f7fffff),2**-126,-2**-126]
 # Neighbors of the prescale overflow threshold, including rounding boundaries.
 threshold=bits(f32((2**128-2**104)/(2/math.sqrt(math.pi))))
 for u in range(threshold-3,threshold+4):seeds.extend([value(u),-value(u)])
 seeds=list(dict.fromkeys(bits(f32(s)) for s in seeds))
 xs=[0.,-0.,.125,-.125,.25,-.25,.5,-.5,.75,-.75,1.,-1.,1.5,-1.5,2.,-2.,3.,-3.,4.,-4.,5.,-5.,6.,-6.,8.,-8.,9.,-9.,10.,-10.,11.,-11.,12.,-12.]
 rows=[]
 for x0 in xs:
  x=f32(x0);xx=exact(x);coef=2/mp.sqrt(mp.pi)*mp.exp(-xx*xx)
  for vb in seeds:
   v=value(vb);vv=exact(v);dy=coef*vv;dy2=-2*xx*dy
   expected=rounded(dy);forward=rounded(mp.erf(xx));second=rounded(dy2)
   family='primary' if abs(x)<=8 else 'tail_limit'
   if not math.isfinite(expected):family='legitimate_overflow'
   rows.append({'id':len(rows),'x_bits':bits(x),'v_bits':vb,'family':family,'expected_bits':bits(expected),'forward_bits':bits(forward),'second_bits':bits(second)})
 return rows

def prepare():
 rows=generate(100);assert rows==generate(150),'Reference changes with precision'
 (A/'inputs.txt').write_text(''.join(f"{r['id']} {r['x_bits']} {r['v_bits']}\n" for r in rows))
 (A/'oracle.json').write_text(json.dumps(rows,indent=2)+'\n');print('PREPARED',len(rows),'cases,100/150 decimal digits agree')

def agrees(actual,expected):
 if math.isinf(expected):return actual==expected
 return math.isfinite(actual) and abs(actual-expected)<=3e-6*abs(expected)+4*2**-149

def run(variant):
 rows=json.loads((A/'oracle.json').read_text());outdir=A/'results';outdir.mkdir(exist_ok=True)
 summary={};base=None
 for layout in ['flat','row','column']:
  cmd=[str(A/('erf_probe-'+variant)),str(A/'inputs.txt'),layout]
  p=subprocess.run(cmd,capture_output=True,text=True,check=True)
  (outdir/f'{variant}-{layout}.csv').write_text(p.stdout);(outdir/f'{variant}-{layout}.stderr').write_text(p.stderr)
  got=list(csv.DictReader(p.stdout.splitlines()));assert len(rows)==len(got)
  stat={family:{'cases':0,'first_failures':[],'forward_failures':[],'second_failures':[]} for family in ['primary','tail_limit','legitimate_overflow']}
  for r,g in zip(rows,got):
   assert int(g['id'])==r['id'] and int(g['x_bits'])==r['x_bits'] and int(g['v_bits'])==r['v_bits']
   s=stat[r['family']];s['cases']+=1
   wanted=value(r['expected_bits'])
   if any(not agrees(value(g[k]),wanted) for k in ['jvp_bits','vjp_bits','grad_bits']):s['first_failures'].append(r['id'])
   if not agrees(value(g['forward_bits']),value(r['forward_bits'])):s['forward_failures'].append(r['id'])
   if not agrees(value(g['second_bits']),value(r['second_bits'])):s['second_failures'].append(r['id'])
  summary[layout]=stat
  if base is None:base=p.stdout
  else:assert base==p.stdout,'Shape-layout changes output bits'
 (outdir/f'{variant}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(variant,json.dumps({f:{k:len(v) if isinstance(v,list) else v for k,v in d.items()} for f,d in summary['flat'].items()}))
 if variant=='restored':
  for l in ['flat','row','column']:assert (outdir/f'original-{l}.csv').read_bytes()==(outdir/f'restored-{l}.csv').read_bytes(),'Restoration not identical'
  print('Original/restored CSV outputs byte-identical')

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 else:run(sys.argv[1])
