"""Construct inputs and freeze the full local experiment before outputs."""
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path('/Users/khamit/Documents/something/channel-growth-2026-09-18/pipeline/one_worker.py')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dependency_hashes():
    base = Path(mp.__file__).resolve().parent
    return {str(p.relative_to(base)):digest(p) for p in sorted(base.rglob('*.py'))}

def build():
    rng = random.Random(2026092603)
    cases,seen,adjustments = [],set(),0
    for version in (1,2):
        for case in json.loads((ROOT/f'evidence/corpus-v{version}.json').read_text())['cases']:
            seen.add(tuple(case['binary64_hex'][key] for key in FIELDS))
    def add(family,s,k,t,r,q,sigma):
        nonlocal adjustments
        values = list(map(float,(s,k,t,r,q,sigma)))
        assert all(math.isfinite(x) for x in values) and min(s,k,t,sigma)>0
        while tuple(x.hex() for x in values) in seen:
            values[-1] = math.nextafter(values[-1],math.inf)
            adjustments += 1
        seen.add(tuple(x.hex() for x in values))
        cases.append({'id':f'v3-{len(cases):04d}','partition':'confirmation_v3','family':family,
                      'binary64_hex':dict(zip(FIELDS,(x.hex() for x in values)))})
    def normalized(family,m,v,scale=1.):
        add(family,scale*math.exp(m/2),scale*math.exp(-m/2),1.,0.,0.,v)
    for _ in range(96):
        v=10**rng.uniform(-10.,math.log10(.125))
        z=rng.uniform(-1.,v/2)
        m=rng.choice((-1.,1.))*v*(v/2-z)
        normalized('small_v',m,v,10**rng.uniform(-180.,180.))
    for _ in range(80):
        v=10**rng.uniform(-10.,math.log10(5.))
        a=rng.uniform(12.,60.)
        m=rng.choice((-1.,1.))*v*(a+v/2)
        normalized('negative_tail',m,v,10**rng.uniform(-180.,180.))
    for _ in range(80):
        k=10**rng.uniform(-12.,12.)
        t=10**rng.uniform(-1.,1.)
        r,q=rng.uniform(-.08,.15),rng.uniform(-.03,.08)
        m=rng.uniform(-2.,2.)
        v=rng.uniform(.13,3.)
        add('regular_carry',k*math.exp(m-(r-q)*t),k,t,r,q,v/math.sqrt(t))
    for seam in ('small_z','tail_a','small_v'):
        for index in range(16):
            v=10**rng.uniform(-5.,math.log10(.12)) if seam=='small_z' else rng.uniform(.15,1.)
            z=-1. if seam=='small_z' else -12. if seam=='tail_a' else rng.uniform(-.95,-.05)
            scale=10**rng.uniform(-100.,100.)
            for direction in (-1.,1.):
                vv=math.nextafter(.125,0. if direction<0 else math.inf) if seam=='small_v' else v
                zz=z+direction*8*math.ulp(z) if seam!='small_v' else z
                m=(-1. if index%2 else 1.)*vv*(vv/2-zz)
                normalized('switching_neighborhood',m,vv,scale)
    for index in range(80):
        a=rng.uniform(36.,43.)
        v=rng.uniform(.5,1.5)
        target=(-747.,-745.,-744.,-742.,-738.,-710.,-500.,-300.)[index%8]
        log_s=target+a*a/2+2*math.log(a)-math.log(v)+.5*math.log(2*math.pi)
        s,k=math.exp(log_s),math.exp(log_s+v*(a+v/2))
        if index%2:
            s,k=k,s
        add('monetary_range',s,k,1.,0.,0.,v)
    for scale in (1e-250,1e-100,1.,1e100,1e250,1e300):
        for v in (1e-15,1e-7,1e-4,.02,.124,.5,4.,15.):
            add('exact_atm',scale,scale,1.,0.,0.,v)
    assert len(cases)==480
    return {'schema':3,'seed':2026092603,'status':'inputs fixed before first v3 confirmation outputs',
            'deduplication_sigma_nextafter_steps':adjustments,'cases':cases}

def verify():
    freeze=json.loads((ROOT/'evidence/freeze-v3.json').read_text())
    for name,value in freeze['source_sha256'].items():
        if digest(ROOT/name)!=value:
            raise RuntimeError('Frozen source changed: '+name)
    if dependency_hashes()!=freeze['mpmath_source_sha256']:
        raise RuntimeError('mpmath source tree changed')
    if digest(WRAPPER)!=freeze['one_worker_sha256']:
        raise RuntimeError('one_worker wrapper changed')
    return freeze

def freeze():
    target,receipt=ROOT/'evidence/corpus-v3.json',ROOT/'evidence/freeze-v3.json'
    if target.exists() or receipt.exists():
        raise FileExistsError('Never replace a frozen confirmation corpus')
    target.write_text(json.dumps(build(),indent=2,sort_keys=True)+'\n')
    native=json.loads((ROOT/'evidence/jackel-build-v1.json').read_text())
    if digest(ROOT/'build/jackel_cli') != native['binary_sha256']:
        raise RuntimeError('Historical native binary changed before freeze')
    for name,value in native['source_sha256'].items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Pinned native source changed: '+name)
    names={str(p.relative_to(ROOT)) for folder in ('lab','tests') for p in (ROOT/folder).glob('*.py')}
    names.update(native['source_sha256'])
    names.update(('EXPERIMENT_V3.md','evidence/corpus-v1.json','evidence/corpus-v2.json',
                  'evidence/corpus-v3.json','evidence/jackel-build-v1.json','build/jackel_cli'))
    data={'frozen_utc':datetime.now(timezone.utc).isoformat(),
          'source_sha256':{name:digest(ROOT/name) for name in sorted(names)},
          'mpmath_source_sha256':dependency_hashes(),'one_worker_sha256':digest(WRAPPER),
          'environment':{'python':sys.version,'mpmath':mp.__version__,'platform':platform.platform(),
                         'threads':{k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k=='CUDA_VISIBLE_DEVICES'}},
          'status':'local freeze, not external preregistration',
          'protocol':'No candidate/pipeline changes after viewing confirmation outputs; retain failures'}
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'cases':480,'source_files':len(names),'frozen_utc':data['frozen_utc'],
                      'candidate_sha256':data['source_sha256']['lab/black_scholes_v3.py']},indent=2))

if __name__=='__main__':
    freeze()
