from pathlib import Path
from decimal import Decimal, localcontext
import json, sys, zipfile, hashlib, subprocess
import numpy as np
import onnx
from onnx import numpy_helper

import argparse
parser = argparse.ArgumentParser(description="Replay the published snapshot and repeat the 80-digit diagnostic check. Exact observation identity requires the recorded environment.")
parser.add_argument('--snapshot', type=Path, default=Path(__file__).resolve().parents[1]/'benchmarks/2026-09-06')
parser.add_argument('--output', type=Path, default=Path('reports/reverification'))
args = parser.parse_args()
ROOT = args.snapshot.resolve()
RELEASE = args.output.resolve()
RELEASE.mkdir(parents=True, exist_ok=True)
report=json.loads((ROOT/'latest.json').read_text())
assert len(report['results'])==246
D=lambda v:Decimal.from_float(float(v))
results=[]
for row in report['results']:
    if row['status']=='pass':continue
    archive=ROOT/row['artifacts']['bundle']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['artifacts']['bundle_sha256']
    directory=RELEASE/'replays'/row['observation_id']
    directory.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:z.extractall(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,expected in manifest.items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected,name
    run=subprocess.run([sys.executable,'-m','gero_stability.cli','replay','.', '--output','rechecked'],cwd=directory,capture_output=True,text=True,timeout=45)
    assert run.returncode==0,run.stderr
    replay=json.loads((directory/'rechecked/latest.json').read_text())
    assert replay['results'][0]['observation_id']==row['observation_id']
    if row['optimization']!='disabled':continue
    with np.load(directory/'inputs.npz') as f: feeds={k:f[k] for k in f.files}
    with np.load(directory/'outputs.npz') as f: outputs={k:f[k] for k in f.files}
    model=onnx.load(directory/'model.onnx')
    constants={i.name:numpy_helper.to_array(i) for i in model.graph.initializer}
    op=row['operation']; answer=[]
    with localcontext() as ctx:
        ctx.prec=80
        for i,values in enumerate(feeds['x']):
            x=[D(v) for v in values]
            if op=='LogSoftmax':
                maximum=max(x);shift=[v-maximum for v in x]
                ln_sum=sum(v.exp() for v in shift).ln()
                answer.append([float(v-ln_sum) for v in shift])
            elif op=='LayerNormalization':
                assert np.all(constants['scale'] == 1) and np.all(constants['bias'] == 0)
                epsilon=next(a.f for a in model.graph.node[0].attribute if a.name=='epsilon')
                mean=sum(x)/Decimal(len(x));centered=[v-mean for v in x]
                var=sum(v*v for v in centered)/Decimal(len(x))
                denominator=(var+D(epsilon)).sqrt()
                answer.append([float(v/denominator) for v in centered])
            elif op=='LpNormalization':
                norm=sum(v*v for v in x).sqrt()
                answer.append([float(v/norm) if norm else 0.0 for v in x])
            elif op=='CosineSimilarity':
                t=[D(v) for v in feeds['target'][i]]
                denominator=max(sum(v*v for v in x).sqrt()*sum(v*v for v in t).sqrt(),D(constants['epsilon']))
                answer.append([float(sum(a*b for a,b in zip(x,t))/denominator)])
            else:raise AssertionError(op)
    exact=np.array(answer,dtype=np.float64)
    comparisons={}
    for backend,y in outputs.items():
        finite=np.isfinite(y)
        error=np.abs(y.astype(np.float64)-exact)
        bad=(error>row['tolerance']['atol']+row['tolerance']['rtol']*np.abs(exact))|~finite
        comparisons[backend]={'nonfinite':int((~finite).sum()),'total':int(y.size),'outside_tolerance':int(bad.sum()),'max_abs_finite':float(error[finite].max()) if finite.any() else None}
    results.append({'operation':op,'variant':row['variant'],'dtype':row['inputs']['x']['dtype'],'case_id':row['case_id'],'oracle':'Decimal, 80 digits; exact conversion of stored binary input and epsilon; final oracle exported as float64','oracle_all_finite':bool(np.isfinite(exact).all()),'comparisons':comparisons})
output={'run_id':report['run_id'],'test_count':57,'total_evaluations':246,'summary':report['summary'],'replayed_archives':14,'distinct_candidate_graphs':len(results),'status':'all original observations reproduced','high_precision_checks':results,'scope':'Algorithmically independent Decimal check, not third-party validation. No novelty or version-regression claim.'}
(RELEASE/'reverification.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
