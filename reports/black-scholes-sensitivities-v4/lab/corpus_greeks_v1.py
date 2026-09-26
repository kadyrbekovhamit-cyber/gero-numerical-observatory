"""Freeze new sensitivity fixtures and local Python pipeline before outputs."""
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import mpmath as mp
from lab.corpus import FIELDS
from lab.corpus_v3 import dependency_hashes, WRAPPER, verify as verify_prices

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    rng=random.Random(2026092604)
    cases=[]
    seen=set()
    for version in (1,2,3):
        old=json.loads((ROOT/f'evidence/corpus-v{version}.json').read_text())['cases']
        seen.update(tuple(c['binary64_hex'][n] for n in FIELDS) for c in old)
    def add(family,s,k,t,r,q,sigma):
        values=list(map(float,(s,k,t,r,q,sigma)))
        assert all(math.isfinite(x) for x in values)
        assert min(s,k,t,sigma)>0 and sigma*math.sqrt(t)>0
        while tuple(x.hex() for x in values) in seen:
            values[-1]=math.nextafter(values[-1],math.inf)
        seen.add(tuple(x.hex() for x in values))
        cases.append({'id':f'g1-{len(cases):04d}','family':family,'partition':'confirmation_greeks_v1',
                      'binary64_hex':dict(zip(FIELDS,(x.hex() for x in values)))})
    for _ in range(96):
        k=10**rng.uniform(-5,7)
        t=10**rng.uniform(-3,1)
        r,q=rng.uniform(-.08,.2),rng.uniform(-.02,.1)
        m=rng.uniform(-2,2)
        v=10**rng.uniform(-2,math.log10(3))
        add('regular',k*math.exp(m-(r-q)*t),k,t,r,q,v/math.sqrt(t))
    for index in range(96):
        seam=index%4
        v=rng.uniform(.03,.12) if seam==0 else .125 if seam==1 else rng.uniform(.2,1.)
        z=-1. if seam==0 else rng.uniform(-.95,-.05) if seam==1 else -12. if seam==2 else 0.
        sign=-1 if index%2 else 1
        if seam==1:v=math.nextafter(v,0. if sign<0 else math.inf)
        else:z=math.nextafter(z,-math.inf if sign<0 else math.inf)
        m=sign*v*(v/2-z)
        scale=10**rng.uniform(-180,180)
        add('price_branch_neighborhood',scale*math.exp(m/2),scale*math.exp(-m/2),1.,0.,0.,v)
    for index in range(96):
        # Tail and monetary factors can cancel in a representable Greek even
        # when the density/CDF or a naive intermediate would already be zero.
        d1=rng.choice((-1.,1.))*rng.uniform(30.,45.)
        v=rng.uniform(.2,1.4)
        m=v*(d1-v/2)
        scale=10**rng.uniform(-250,250)
        rate=rng.choice((0.,-500.,500.))
        add('tail_and_monetary_range',scale*math.exp(m/2),scale*math.exp(-m/2),1.,rate,rate,v)
    for scale in (1e-280,1e-100,3.7,1e100,1e280,1e300):
        for v in (math.ulp(0.),1e-300,1e-150,1e-12,.017,.49,7.,28.):
            add('exact_atm_range',scale,scale,1.,0.,0.,v)
    for index in range(48):
        d1=rng.choice((-1.,1.))*12.
        if index%3:d1=math.nextafter(d1,0. if index%3==1 else math.copysign(math.inf,d1))
        v=10**rng.uniform(-2,0)
        m=v*(d1-v/2)
        scale=10**rng.uniform(-150,150)
        add('cdf_tail_switch',scale*math.exp(m/2),scale*math.exp(-m/2),1.,0.,0.,v)
    assert len(cases)==384
    return {'seed':2026092604,'status':'inputs generated before viewing confirmation outputs','cases':cases}


def verify():
    receipt=json.loads((ROOT/'evidence/freeze-greeks-v1.json').read_text())
    for name,expected in receipt['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
    if dependency_hashes()!=receipt['mpmath_source_sha256']:raise RuntimeError('mpmath changed')
    if digest(WRAPPER)!=receipt['worker_sha256']:raise RuntimeError('worker changed')
    actual_runtime={'python':digest(Path(sys.executable)),'math_extension':digest(Path(math.__file__))}
    if actual_runtime!=receipt['runtime_binary_sha256']:raise RuntimeError('Python/math runtime changed')
    verify_prices()
    return receipt


def main():
    corpus=ROOT/'evidence/corpus-greeks-v1.json'
    receipt=ROOT/'evidence/freeze-greeks-v1.json'
    if corpus.exists() or receipt.exists():raise FileExistsError('Never replace frozen confirmation')
    verify_prices()
    corpus.write_text(json.dumps(build(),sort_keys=True,indent=2)+'\n')
    names={'lab/__init__.py','lab/black_scholes.py','lab/black_scholes_v2.py','lab/black_scholes_v3.py',
           'lab/black_scholes_greeks.py','lab/greeks_reference.py','lab/corpus.py','lab/corpus_v3.py',
           'lab/corpus_greeks_v1.py','lab/run_greeks_v1.py','tests/test_black_scholes_greeks.py',
           'EXPERIMENT_GREEKS_V1.md','evidence/corpus-greeks-v1.json','evidence/freeze-v3.json'}
    data={'frozen_utc':datetime.now(timezone.utc).isoformat(),
          'source_sha256':{name:digest(ROOT/name) for name in sorted(names)},
          'worker_sha256':digest(WRAPPER),'mpmath_source_sha256':dependency_hashes(),
          'runtime_binary_sha256':{'python':digest(Path(sys.executable)),'math_extension':digest(Path(math.__file__))},
          'environment':{'python':sys.version,'mpmath':mp.__version__,'platform':platform.platform(),
                         'threads':{k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k=='CUDA_VISIBLE_DEVICES'}},
          'scope':'local Python source/runtime hashes; not a full hermetic OS/libm image or external preregistration'}
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'cases':384,'frozen_utc':data['frozen_utc'],'candidate':data['source_sha256']['lab/black_scholes_greeks.py']},indent=2))


if __name__=='__main__':main()
