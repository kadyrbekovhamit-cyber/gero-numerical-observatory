"""Check saved evidence and isolated patch application; no MLX execution."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
checks=[]
def check(name,value):
    checks.append(dict(name=name,passed=bool(value)))
def read(name):return json.loads((ROOT/name).read_text())

for p in ROOT.glob('*.py'):
    ast.parse(p.read_text());check('syntax/'+p.name,True)
probe=read('probe-results.json')
check('wheel_version',probe['version']=='0.32.2')
check('wheel_cpu',probe['device']=='Device(cpu, 0)')
def flat(x):
    if not isinstance(x,list):return [x]
    return [v for child in x for v in flat(child)]
for row in probe['cases']:
    expected=([3,5],[2,7]) if row['side']=='left' and row['sorted_indices'] else (
        ([8,0],[2,2]) if row['side']=='left' else ([3,3],[9,0]))
    check('wheel/'+row['side']+'/'+str(row['sorted_indices']),
          [flat(x) for x in row['gradients']]==list(expected))
q=read('quantized-and-loss-results.json')
for row in q['quantized']:
    expected=[160.,160.,160.] if row['side']=='left' and row['sorted_indices'] else [0.,0.,0.]
    check('qmm/'+row['side']+'/'+str(row['sorted_indices']),row['max_errors']==expected)
for row in q['loss_step']:
    check('finite_differences/'+str(row['sorted_indices']),row['finite_differences']==[8,0])
    check('step/'+str(row['sorted_indices']),row['new_loss']==(11.28125 if row['sorted_indices'] else 6.125))
tail=read('broadcast-tail-results.json')
for i,row in enumerate(tail['cases']):
    check('view/'+str(i),row['visible_input']==[[[2.]]] and row['loss']==30 and
          row['actual_db']==([2.]+row['tail'] if row['sorted_indices'] else [2.,2.,2.]))
for label,expected in [('before',(69,19,626,71)),('after',(69,0,626,0))]:
    log=(ROOT/f'run-{label}.log').read_text()
    m=re.search(r'SUMMARY scenarios=(\d+) failed_scenarios=(\d+) checks=(\d+) failures=(\d+)',log)
    check('native/'+label,bool(m) and tuple(map(int,m.groups()))==expected)
    check('native/'+label+'/no_exceptions','exception=' not in log)
meta=read('source-metadata.json')
for row in meta['files']:
    p=ROOT/('upstream-'+row['path'].replace('/','_'))
    check('hash/'+row['path'],hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
for row in read('duplicate-search.json'):
    check('public_search/'+row['query'],'result' in row and not row['result']['incomplete_results'])
with tempfile.TemporaryDirectory(prefix='mlx-gather-patch-') as temp:
    t=Path(temp);(t/'mlx').mkdir()
    target=t/'mlx/primitives.cpp'
    target.write_bytes((ROOT/'upstream-mlx_primitives.cpp').read_bytes())
    r=subprocess.run(['git','apply','--check',str(ROOT/'gather-vjp-preconditions.patch')],cwd=t,capture_output=True,text=True)
    check('patch/applies_to_pinned_main',r.returncode==0)
    if r.returncode==0:
        r=subprocess.run(['git','apply',str(ROOT/'gather-vjp-preconditions.patch')],cwd=t,capture_output=True,text=True)
        check('patch/apply_success',r.returncode==0)
        check('patch/matches_generated_source',target.read_bytes()==(ROOT/'patched-upstream-primitives.cpp').read_bytes())
report=dict(passed=sum(c['passed'] for c in checks),total=len(checks),checks=checks)
(ROOT/'validation-results.json').write_text(json.dumps(report,indent=2)+'\n')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
          and p.name!='SHA256SUMS.json' and p.suffix!='.o' and not p.name.startswith('native-')}
(ROOT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(passed=report['passed'],total=report['total'])))
if report['passed']!=report['total']:
    print(json.dumps([c for c in checks if not c['passed']],indent=2))
    raise SystemExit(1)
