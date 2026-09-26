"""Pre-output construction of 160 confirmation scenarios. No solver calls."""
from fractions import Fraction as F
from itertools import product
import json,hashlib,datetime,platform
from pathlib import Path
import mpmath as mp
import flint
from lab.iv_interval import Contract,QuoteInterval,rounding_cell
from lab.iv_reference import value,number

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['lab/iv_interval.py','lab/iv_reference.py','lab/iv_corpus.py','lab/run_iv.py',
         'iv-tests/test_iv_interval.py','iv_range.py','EXPERIMENT_IV_V1.md','requirements-iv.txt']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    rows=[]
    def add(family,c,q,expected='interval',seed=None,kwargs=None,extra=None):
        row={'id':f'iv-{len(rows)+1:03d}','family':family,'contract':c.as_dict(),
             'quote':q.as_dict(),'expected_status':expected,'expected_resolved':True,'kwargs':kwargs or {}}
        if seed is not None:row['compatible_sigma']=str(seed)
        if extra:row.update(extra)
        rows.append(row)
    for option,ratio,t,sigma in product(('call','put'),(F('0.65'),F('.99'),F('1.01'),F('1.7')),
                                       (F('0.125'),F('.5'),F('1.75'),F('3.25')),(F('.17'),F('.83'))):
        idx=len(rows);c=Contract(103*ratio,103,t,F(3 if idx%3 else -1,100),F(1,100),option)
        tick=F(103,10**6 if sigma<F('.5') else 10**4)
        with mp.workdps(260):
            price=value(c.as_dict(),sigma);floor=int(mp.floor(price/number(tick)))
        add('decimal_quote',c,QuoteInterval(floor*tick,(floor+1)*tick),seed=sigma)
    for option,scale,sigma in product(('call','put'),(F('1e-200'),F(1),F('1e100'),F('1e-308')),
                                     (F('.09'),F('.31'),F('1.2'))):
        c=Contract(119*scale,101*scale,F(7,4),F(1,100),F(2,100),option)
        with mp.workdps(260):price=float(value(c.as_dict(),sigma))
        add('rounded_quote',c,rounding_cell(price),seed=sigma)
    for option in ('call','put'):
        c=Contract(127,103,1,option=option);floor=24 if option=='call' else 0;cap=127 if option=='call' else 103
        patterns=[(QuoteInterval(floor-2,floor-1),'empty'),(QuoteInterval(floor,floor),'singleton_zero'),
                  (QuoteInterval(floor-1,floor,True,False),'empty'),
                  (QuoteInterval(floor,cap),'interval'),(QuoteInterval(floor,cap,False,False),'interval'),
                  (QuoteInterval(cap,cap+1),'empty'),(QuoteInterval(floor+2,floor+2,False,True),'empty'),
                  (QuoteInterval(-4,-3),'empty')]
        for quote,status in patterns:add('boundaries',c,quote,status)
    for option,s,inside in product(('call','put'),(81,107),(True,False)):
        c=Contract(s,101,0,option=option);floor=max(s-101 if option=='call' else 101-s,0)
        add('expiry',c,QuoteInterval(floor,floor) if inside else QuoteInterval(floor+1,floor+2),
            'all_volatilities' if inside else 'empty')
    for option,(s,k),which in product(('call','put'),((184,93),(221,117),(103,219),(88,201)),('intrinsic','ceiling')):
        c=Contract(s,k,1,option=option)
        price=max(s-k if option=='call' else k-s,0) if which=='intrinsic' else (s if option=='call' else k)
        add('rounded_boundary',c,rounding_cell(float(price)))
    for option,s in product(('call','put'),(79,109,139,189)):
        c=Contract(s,101,F(5,4),option=option);price=max(s-101 if option=='call' else 101-s,0)+F(13,4)
        add('point_quote',c,QuoteInterval(price,price))
    for j in range(2):
        c=Contract(137+j,113,1);p=c.s-c.k+2
        add('budget_sigma',c,QuoteInterval(p,p+1),kwargs={'max_sigma':str(F(1,1000*(j+1)))},
            extra={'expected_resolved':False,'expected_endpoint_state':'sigma_budget_exhausted'})
    for j in range(2):
        c=Contract(138+j,115,1);p=c.s-c.k+3
        add('budget_iterations',c,QuoteInterval(p,p+1),kwargs={'max_steps':j+1},
            extra={'expected_resolved':False,'expected_endpoint_state':'iteration_budget_exhausted'})
    for j,sigma in enumerate((F(1),F(1),F(1,2))):
        c=Contract(106+j,106+j,1)
        with mp.workdps(260):price=F(mp.nstr(value(c.as_dict(),sigma),100))
        add('budget_precision',c,QuoteInterval(price,price),kwargs={'precisions':[64]},
            extra={'expected_resolved':False,'expected_endpoint_state':
                   'unresolved_bracketing' if sigma==1 else 'precision_budget_exhausted'})
    add('budget_boundary',Contract(91,91,1,r='1e-32'),QuoteInterval('1e-30',10),expected='unresolved',
        kwargs={'precisions':[64]},extra={'expected_resolved':False})
    for j in range(4):
        s,k=143+17*j,101;c=Contract(s,k,1);q=QuoteInterval(s-k+8+j,s-k+9+j)
        group=f'symmetry-{j}'
        add('symmetry',c,q,extra={'symmetry_group':group,'sigma_multiplier':'1'})
        add('symmetry',Contract(s,k,1,option='put'),QuoteInterval(8+j,9+j),
            extra={'symmetry_group':group,'sigma_multiplier':'1'})
        scale=F(10**120) if j%2==0 else F(1,10**120)
        add('symmetry',Contract(s*scale,k*scale,1),QuoteInterval(q.lower*scale,q.upper*scale),
            extra={'symmetry_group':group,'sigma_multiplier':'1'})
        add('symmetry',Contract(s,k,4),q,extra={'symmetry_group':group,'sigma_multiplier':'2'})
    assert len(rows)==160
    keys=[json.dumps({k:r[k] for k in ('contract','quote','kwargs')},sort_keys=True) for r in rows]
    assert len(set(keys))==len(keys)
    return rows


def dependency_manifest(module):
    root=Path(module.__file__).parent
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def freeze():
    target=ROOT/'iv-evidence/corpus-v1.json';receipt=ROOT/'iv-evidence/freeze-v1.json'
    assert not target.exists() and not receipt.exists(),'Never overwrite a frozen corpus'
    rows=build();target.write_text(json.dumps(rows,indent=2)+'\n')
    data={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'purpose':'freeze before first candidate outputs on these 160 scenarios',
          'sources':{n:sha(ROOT/n) for n in SOURCES},'corpus_sha256':sha(target),
          'python':platform.python_version(),'platform':platform.platform(),
          'python_flint_version':flint.__version__,'mpmath_version':mp.__version__,
          'flint_files':dependency_manifest(flint),'mpmath_files':dependency_manifest(mp)}
    worker=Path('/Users/khamit/Documents/something/channel-growth-2026-09-18/pipeline/one_worker.py')
    data['original_worker_sha256']=sha(worker)
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k in ('time_utc','corpus_sha256','python','python_flint_version')}))


def verify(portable=False):
    receipt=json.loads((ROOT/'iv-evidence/freeze-v1.json').read_text())
    for n,h in receipt['sources'].items():assert sha(ROOT/n)==h, n
    assert sha(ROOT/'iv-evidence/corpus-v1.json')==receipt['corpus_sha256']
    assert flint.__version__==receipt['python_flint_version'] and mp.__version__==receipt['mpmath_version']
    assert dependency_manifest(mp)==receipt['mpmath_files']
    if not portable:assert dependency_manifest(flint)==receipt['flint_files']
    return receipt


if __name__=='__main__':freeze()
