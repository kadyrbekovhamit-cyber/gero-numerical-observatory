"""Replay original, fixed upstream, restored multiplier and official wheel.

No network access or in-process replacement of FinancePy functions is used.
"""
from pathlib import Path, PurePosixPath
import argparse,base64,csv,hashlib,io,json,os,shutil,subprocess,sys,tarfile,zipfile
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();R=Path(a.out).resolve();assert not R.exists(),'Choose an unused output directory'
source=json.loads((P/'SOURCE.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
for k in ['original','current','wheel']:assert sha((P/source[k]['file']).read_bytes())==source[k]['sha256']
R.mkdir(parents=True);(R/'evidence').mkdir();counts={};identities={}
for kind,label,treefile in [('original','baseline','ORIGINAL_SOURCE_TREE.json'),('current','candidate','CURRENT_SOURCE_TREE.json')]:
 tree=json.loads((P/treefile).read_text());assert not tree.get('truncated');expected={x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'};found={};local={}
 with tarfile.open(P/source[kind]['file']) as tf:
  for m in tf.getmembers():
   if m.isdir():continue
   assert m.isfile(),m.name
   path=PurePosixPath(m.name);assert not path.is_absolute() and '..' not in path.parts
   rel=PurePosixPath(*path.parts[1:]).as_posix();assert rel!='.'
   data=tf.extractfile(m).read();oid=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert oid==expected[rel],rel;found[rel]=oid
   if rel.startswith('financepy/') or (kind=='current' and rel in source['upstream_support_paths']):
    dest=R/label/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);local[rel]=sha(data)
 assert found==expected
 counts[kind]=len(found);identities[label]=local
# Use the same current official tests/CSV fixtures for both package versions.
for name in source['upstream_support_paths']:
 f=R/'baseline'/name;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/'candidate'/name,f)
shutil.copytree(R/'candidate',R/'mutation');target=R/'mutation'/source['target'];text=target.read_text();assert text.count('        md = dd / fp\n')==1;target.write_text(text.replace('        md = dd / fp\n','        md = dd / fp * 10000\n'))
wheel=P/source['wheel']['file'];package_files=0
with zipfile.ZipFile(wheel) as z:
 record=next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))
 for name,digest,size in csv.reader(io.StringIO(z.read(record).decode())):
  if digest:
   algorithm,value=digest.split('=',1);data=z.read(name);assert base64.urlsafe_b64encode(hashlib.new(algorithm,data).digest()).decode().rstrip('=')==value;assert len(data)==int(size)
  path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
  if name.startswith('financepy/') and not name.endswith('/'):
   dest=R/'release'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name));package_files+=1
for name in ['audit.py','minimal.py','verify.py','run_tests.py','test_bond_zero_duration.py','inputs.json']:shutil.copyfile(P/name,R/name)
shutil.copyfile(P/'recorded-evidence/protocol.json',R/'evidence/protocol.json')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MPLBACKEND='Agg')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
steps=[]
def run(label,args,expected=0):
 out=subprocess.run([sys.executable,'-B',*args],cwd=R,env=env,capture_output=True,text=True,timeout=240);(R/'evidence'/(label+'.log')).write_text(out.stdout+out.stderr);assert out.returncode==expected,(label,out.returncode,out.stdout[-1200:],out.stderr[-1200:]);steps.append({'step':label,'exit_code':out.returncode,'expected':expected});print(label,'complete',flush=True);return out.stdout
# Regenerate the independent80/120-digit oracle in a separate empty location.
o=R/'oracle-check';o.mkdir();(o/'evidence').mkdir();shutil.copyfile(P/'audit.py',o/'audit.py');run('oracle-refreeze',[str(o/'audit.py'),'prepare']);assert json.loads((o/'inputs.json').read_text())==json.loads((P/'inputs.json').read_text())
for v in ['baseline','candidate','mutation','release']:
 run(v+'-grid',['audit.py',v]);run(v+'-minimal',['minimal.py',v])
for v in ['baseline','candidate']:assert '20 passed' in run(v+'-upstream',['run_tests.py',v,'upstream'])
assert '27 passed' in run('candidate-new',['run_tests.py','candidate','new'])
assert '24 failed, 3 passed' in run('mutation-new',['run_tests.py','mutation','new'],1)
run('verification',['verify.py'])
verification=json.loads((R/'evidence/verification.json').read_text());matches={}
for v in ['baseline','candidate','mutation','release']:
 assert verification[v]['duration_failures']==(0 if v=='candidate' else 1200)
 matches[v]=json.loads((R/'evidence'/(v+'-observations.json')).read_text())==json.loads((P/'recorded-evidence'/(v+'-observations.json')).read_text())
for label,manifest in identities.items():
 for n,h in manifest.items():assert sha((R/label/n).read_bytes())==h,n
receipt={'status':'portable_replay_verified','verified_at':datetime.now(timezone.utc).isoformat(),'labels':source['labels'],'source_archive_git_blob_counts':counts,'wheel_record_verified':True,'wheel_package_files':package_files,'oracle_regeneration_matches_frozen':True,'recorded_numeric_rows_exact_equal':matches,'steps':steps,'duration_failures_original_current_mutation_release':[verification[v]['duration_failures'] for v in ['baseline','candidate','mutation','release']],'source_identities_restored_verified':True,'sequential_processes':True,'gpu_used':False,'audio_playback':False}
(R/'PORTABLE_REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Portable replay complete',flush=True)
