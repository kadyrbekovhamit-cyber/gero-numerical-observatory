"""Second pre-output corpus, after disclosed v1 harness failures; same candidate."""
from fractions import Fraction as F
from itertools import product
import json,datetime,platform
from pathlib import Path
import mpmath as mp
import flint
from lab.iv_interval import Contract,QuoteInterval,rounding_cell
from lab.iv_reference_v2 import value
from lab.iv_corpus import ROOT,sha,dependency_manifest

SOURCES=['lab/iv_interval.py','lab/iv_reference_v2.py','lab/iv_corpus_v2.py','lab/run_iv_v2.py',
         'iv-tests/test_iv_interval.py','iv_range.py','EXPERIMENT_IV_V2.md','requirements-iv.txt',
         'lab/iv_corpus.py','iv-evidence/freeze-v1.json','iv-evidence/corpus-v1.json',
         'iv-evidence/confirmation-v1.json']


def rounded_quote(c,sigma):
    # Python's rational-to-float conversion, avoiding float(mpf) underflow rounding.
    quotes=[]
    for digits in (180,260):
        with mp.workdps(digits):quotes.append(float(F(mp.nstr(value(c.as_dict(),sigma),digits))))
    assert quotes[0]==quotes[1],'Unstable input construction'
    return rounding_cell(quotes[1])


def build():
    rows=[]
    def add(family,c,q,status='interval',resolved=True,seed=None,kwargs=None,state=None):
        r={'id':f'iv2-{len(rows)+1:03d}','family':family,'contract':c.as_dict(),'quote':q.as_dict(),
           'expected_status':status,'expected_resolved':resolved,'kwargs':kwargs or {}}
        if seed is not None:r['compatible_sigma']=str(seed)
        if state:r['expected_endpoint_state']=state
        rows.append(r)
    for option,scale,t,sigma in product(('call','put'),(F('1e-309'),F('1e-210'),F(1),F('1e110')),
                                       (F(3,8),F(9,4)),(F('.11'),F('.27'),F('.61'),F('1.4'))):
        c=Contract(131*scale,107*scale,t,F(3,200),F(1,40),option)
        add('rounded_recovery',c,rounded_quote(c,sigma),seed=sigma)
    for option,(s,k),which in product(('call','put'),((177,97),(237,119),(101,227),(83,207)),('intrinsic','ceiling')):
        c=Contract(s,k,1,option=option)
        price=max(s-k if option=='call' else k-s,0) if which=='intrinsic' else (s if option=='call' else k)
        add('rounded_boundary',c,rounding_cell(float(price)))
    for j in range(2):
        c=Contract(149+j,117,1);p=c.s-c.k+2
        add('budget_sigma',c,QuoteInterval(p,p+1),resolved=False,kwargs={'max_sigma':str(F(1,3000*(j+1)))},state='sigma_budget_exhausted')
    for j in range(2):
        c=Contract(152+j,119,1);p=c.s-c.k+3
        add('budget_iterations',c,QuoteInterval(p,p+1),resolved=False,kwargs={'max_steps':j+1},state='iteration_budget_exhausted')
    for j,sigma in enumerate((F(1),F(1),F(1,2))):
        c=Contract(116+j,116+j,1)
        with mp.workdps(260):p=F(mp.nstr(value(c.as_dict(),sigma),100))
        add('budget_precision',c,QuoteInterval(p,p),resolved=False,kwargs={'precisions':[64]},
            state='unresolved_bracketing' if sigma==1 else 'precision_budget_exhausted')
    # Upper bound of max(A-B,0) overlaps this exact lower quote at 64 bits.
    add('budget_boundary',Contract(113,113,1,r='1e-30'),QuoteInterval('1e-29',9),
        status='unresolved',resolved=False,kwargs={'precisions':[64]})
    for option,s,inside in product(('call','put'),(87,111),(True,False)):
        c=Contract(s,103,0,option=option);floor=max(s-103 if option=='call' else 103-s,0)
        add('expiry',c,QuoteInterval(floor,floor) if inside else QuoteInterval(floor+1,floor+2),
            status='all_volatilities' if inside else 'empty')
    assert len(rows)==96
    key=lambda r:json.dumps({k:r[k] for k in ('contract','quote','kwargs')},sort_keys=True)
    old={key(r) for r in json.loads((ROOT/'iv-evidence/corpus-v1.json').read_text())}
    assert len({key(r) for r in rows})==len(rows) and not old.intersection(key(r) for r in rows)
    return rows


def freeze():
    target=ROOT/'iv-evidence/corpus-v2.json';receipt=ROOT/'iv-evidence/freeze-v2.json'
    assert not target.exists() and not receipt.exists()
    rows=build();target.write_text(json.dumps(rows,indent=2)+'\n')
    data={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'purpose':'new inputs frozen after v1 diagnosis, before candidate v2-corpus outputs',
          'sources':{n:sha(ROOT/n) for n in SOURCES},'corpus_sha256':sha(target),
          'python':platform.python_version(),'platform':platform.platform(),
          'python_flint_version':flint.__version__,'mpmath_version':mp.__version__,
          'flint_files':dependency_manifest(flint),'mpmath_files':dependency_manifest(mp)}
    assert data['sources']['lab/iv_interval.py']==json.loads((ROOT/'iv-evidence/freeze-v1.json').read_text())['sources']['lab/iv_interval.py']
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ('time_utc','corpus_sha256')}))


def verify(portable=False):
    d=json.loads((ROOT/'iv-evidence/freeze-v2.json').read_text())
    for n,h in d['sources'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'iv-evidence/corpus-v2.json')==d['corpus_sha256']
    assert flint.__version__==d['python_flint_version'] and mp.__version__==d['mpmath_version']
    assert dependency_manifest(mp)==d['mpmath_files']
    if not portable:assert dependency_manifest(flint)==d['flint_files']
    return d


if __name__=='__main__':freeze()
