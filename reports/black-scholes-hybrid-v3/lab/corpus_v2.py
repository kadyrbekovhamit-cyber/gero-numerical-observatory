"""Generate the prescribed new confirmation corpus without evaluating prices."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from lab.corpus import FIELDS

ROOT = Path(__file__).resolve().parents[1]


def build():
    rng = random.Random(2026092602)
    cases = []
    def add(family,s,k,t,r,q,sigma):
        cases.append({'id':f'v2-{len(cases):04d}','partition':'confirmation_v2','family':family,
                      'binary64_hex':{key:float(x).hex() for key,x in zip(FIELDS,(s,k,t,r,q,sigma))}})
    scales=(1e-250,1e-150,1e-50,1.,1e50,1e150,1e250)
    for index in range(96):
        k=scales[index%len(scales)]
        direction=rng.choice((0.,math.inf))
        s=k
        for _ in range(rng.randint(1,4)):
            s=math.nextafter(s,direction)
        add('near_atm',s,k,1.,0.,0.,10.**rng.uniform(-12.,-3.))
    for _ in range(96):
        v=10.**rng.uniform(-12.,math.log10(20.))
        m=rng.choice((-1.,1.))*v*10.**rng.uniform(-3.,2.4)
        m=max(-600.,min(600.,m))
        add('normalized',math.exp(m),1.,1.,0.,0.,v)
    for _ in range(72):
        k=10.**rng.uniform(-150.,150.)
        s=k*math.exp(rng.uniform(-2.,2.))
        t=10.**rng.uniform(-6.,1.)
        r,q=rng.uniform(-.12,.12),rng.uniform(-.06,.06)
        sigma=10.**rng.uniform(-4.,math.log10(2.))
        add('nonzero_carry',s,k,t,r,q,sigma)
    for index in range(24):
        k=(1e-200,1.,1e200)[index%3]
        m=rng.choice((-1.,1.))*rng.uniform(36.,43.)
        add('tail_and_range',k*math.exp(m),k,1.,0.,0.,rng.uniform(.7,1.3))
    assert len(cases)==288
    return {'schema':2,'seed':2026092602,'status':'frozen before first candidate evaluation',
            'input_contract':'exact binary64 BSM inputs','cases':cases}


def freeze():
    target=ROOT/'evidence/corpus-v2.json'
    receipt=ROOT/'evidence/freeze-v2.json'
    if target.exists() or receipt.exists():
        raise FileExistsError('Do not replace frozen confirmation fixtures')
    source_names=('lab/black_scholes_v2.py','lab/black_scholes.py','lab/reference.py',
                  'lab/metrics.py','lab/corpus_v2.py','EXPERIMENT_V2.md','tests/test_black_scholes_v2.py')
    target.write_text(json.dumps(build(),indent=2,sort_keys=True)+'\n')
    data={'frozen_utc':datetime.now(timezone.utc).isoformat(),
          'corpus_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
          'protocol':'No candidate modification after viewing v2 confirmation outputs',
          'development_corpus':'all 316 v1 cases are now development data'}
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    freeze()
