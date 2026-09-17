"""Verify and replay the real pinned sources in separate processes, sequentially."""
from pathlib import Path,PurePosixPath
import argparse,base64,csv,difflib,hashlib,io,json,os,shutil,subprocess,sys,tarfile,zipfile
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();R=Path(a.out).resolve()
assert not R.exists(),'Choose a new output directory to preserve evidence'
source=json.loads((P/'SOURCE_REVIEW_RECEIPT.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
tar=P/'vendor/actuarialmath-source-7d18f11ad304.tar.gz';wheel=P/'vendor'/source['wheel_name']
assert sha(tar.read_bytes())==source['source_tar_sha256'];assert sha(wheel.read_bytes())==source['wheel_sha256']
R.mkdir(parents=True);base=R/'source/baseline';base.mkdir(parents=True)
with tarfile.open(tar) as z:
 for m in z.getmembers():
  if m.isdir():continue
  assert m.isfile(),m.name
  path=PurePosixPath(m.name);assert not path.is_absolute() and '..' not in path.parts
  rel=Path(*path.parts[1:]);assert str(rel)!='.'
  target=base/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.extractfile(m).read())
manifest=json.loads((P/'SOURCE_MANIFEST.json').read_text())
assert {x.relative_to(base).as_posix() for x in base.rglob('*') if x.is_file()}==set(manifest)
for n,h in manifest.items():assert sha((base/n).read_bytes())==h,n
release=R/'source/release/src';release.mkdir(parents=True)
with zipfile.ZipFile(wheel) as z:
 record=next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))
 for name,digest,size in csv.reader(io.StringIO(z.read(record).decode())):
  if digest:
   algo,value=digest.split('=',1);assert algo=='sha256'
   assert base64.urlsafe_b64encode(hashlib.sha256(z.read(name)).digest()).decode().rstrip('=')==value
   assert len(z.read(name))==int(size)
 for name in z.namelist():
  path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
  if name.startswith('actuarialmath/') and not name.endswith('/'):
   dest=release/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
name='src/actuarialmath/mortalitylaws.py';text=(base/name).read_text()
old='        def _f(x: int, s,t : float) -> float:\n            return alpha / (omega - (x+s))\n'
new='        def _f(x: int, s,t : float) -> float:\n            remaining = omega - (x+s)\n            return (alpha / remaining\n                    * ((remaining - t) / remaining)**(alpha - 1))\n'
assert text.count(old)==1
patch=''.join(difflib.unified_diff(text.splitlines(True),text.replace(old,new).splitlines(True),fromfile='a/'+name,tofile='b/'+name))
assert patch==(P/'candidate.patch').read_text()
for label in ['candidate','mutation']:
 target=R/'source'/label;shutil.copytree(base,target);(target/name).write_text(text.replace(old,new))
 if label=='mutation':(target/name).write_text((target/name).read_text().replace(new,old))
for n in ['check_beta.py','minimal_repro.py','verify.py','candidate.patch','SOURCE.json','SOURCE_MANIFEST.json']:shutil.copy2(P/n,R/n)
shutil.copy2(wheel,R/wheel.name);(R/'evidence').mkdir();shutil.copy2(P/'review/pypi.json',R/'evidence/pypi-current.json')
env=dict(os.environ)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
env.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(R/'mpl-cache'),XDG_CACHE_HOME=str(R/'cache'))
for label in ['baseline','candidate','mutation','release']:
 for script,output in [('check_beta.py',f'{label}.json'),('minimal_repro.py',f'minimal-{label}.json')]:
  cmd=[sys.executable,str(R/script),'--source',str(R/'source'/label)]
  if script=='check_beta.py':cmd+=['--output',str(R/'evidence'/output)]
  result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
  (R/'evidence'/f'{script}-{label}.log').write_text(result.stdout+result.stderr)
  assert result.returncode==0,(label,script,result.stderr)
  if script=='minimal_repro.py':(R/'evidence'/output).write_text(result.stdout)
  print(label,script,'complete',flush=True)
subprocess.run([sys.executable,str(R/'verify.py')],env=env,check=True)
comparisons={}
for label in ['baseline','candidate','mutation','release']:
 fresh=json.loads((R/'evidence'/f'{label}.json').read_text());old=json.loads((P/'recorded-evidence'/f'{label}.json').read_text())
 comparisons[label]={k:fresh[k]==old[k] for k in ['summary','density','integrals','insurance']}
 # Numeric values can vary by platform; hard assertions above enforce tolerances.
receipt={'verified_utc':datetime.now(timezone.utc).isoformat(),'source_pin':source['pin'],'full_source_hashes_verified':len(manifest),'wheel_record_verified':True,'recorded_numeric_rows_exact_equal':comparisons,'verification':'evidence/paired-verification.json','sequential_processes':True,'gpu_used':False}
(R/'PORTABLE_REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Portable replay complete')
