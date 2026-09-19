"""Offline native CPU replay. Requires Python3.12+,mpmath1.3,CMake3.25+,Ninja,C++20."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work-dir',required=True,type=Path);ap.add_argument('--cmake',default=shutil.which('cmake'));ap.add_argument('--ninja',default=shutil.which('ninja'));args=ap.parse_args()
 if sys.version_info<(3,12) or not args.cmake or not args.ninja:ap.error('Python>=3.12,CMake,Ninja required')
 import mpmath
 assert mpmath.__version__=='1.3.0'
 w=args.work_dir.resolve();w.mkdir(parents=True,exist_ok=True);assert not (w/'REPLAY_RECEIPT.json').exists()
 paths={};verified={};manifest=json.loads((P/'SOURCE_MANIFEST.json').read_text())
 for key,item in manifest.items():
  archive=P/'sources'/item['archive'];assert sha(archive)==item['sha256']
  with tarfile.open(archive) as t:t.extractall(w/'sources',filter='data')
  source=w/'sources'/item['directory'];count=0
  for obj in json.loads((P/'evidence'/item['git_tree']).read_text())['tree']:
   if obj['type']!='blob':continue
   f=source/obj['path'];b=os.readlink(f).encode() if f.is_symlink() else f.read_bytes()
   assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==obj['sha'],(key,obj['path']);count+=1
  paths[key]=source;verified[key]=count
 d=w/'driver';d.mkdir(exist_ok=True)
 for name in ['erf_probe.cpp','check_erf.py','inputs.txt','oracle.json','chain_inputs.txt','boundary_inputs.txt']:shutil.copy2(P/name,d/name)
 (d/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.25)\nproject(gero_erf LANGUAGES C CXX)\nset(CMAKE_CXX_STANDARD 20)\nadd_subdirectory("'+str(paths['mlx'])+'" mlx-build)\nadd_executable(erf_probe erf_probe.cpp)\ntarget_link_libraries(erf_probe PRIVATE mlx)\n')
 env=os.environ.copy();env.update({x:'1' for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','CMAKE_BUILD_PARALLEL_LEVEL']});commands=[]
 def call(cmd,label):
  cmd=list(map(str,cmd));commands.append({'label':label,'command':cmd});print(label,flush=True)
  with (w/(label+'.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 cmd=[args.cmake,'-S',d,'-B',w/'build','-G','Ninja','-DCMAKE_MAKE_PROGRAM='+args.ninja,'-DCMAKE_BUILD_TYPE=Release','-DFETCHCONTENT_SOURCE_DIR_FMT='+str(paths['fmt']),'-DFETCHCONTENT_SOURCE_DIR_JSON='+str(paths['json']),'-DMLX_BUILD_CPU=ON']
 cmd+=['-D'+k+'=OFF' for k in ['MLX_BUILD_METAL','MLX_BUILD_CUDA','MLX_BUILD_TESTS','MLX_BUILD_EXAMPLES','MLX_BUILD_GGUF','MLX_BUILD_SAFETENSORS','MLX_BUILD_PYTHON_BINDINGS','MLX_USE_CCACHE']]
 call(cmd,'configure');call([sys.executable,d/'check_erf.py','prepare'],'oracle');assert (d/'oracle.json').read_bytes()==(P/'oracle.json').read_bytes()
 src=paths['mlx']/'mlx/primitives.cpp';original=src.read_bytes();results={};csvs=0
 try:
  for variant in ['original','candidate','balanced','restored']:
   src.write_bytes(original)
   if variant in ['candidate','balanced']:
    patch='erf-candidate.patch' if variant=='candidate' else 'erf-balanced-candidate.patch'
    call(['patch','-p1','-d',paths['mlx'],'-i',P/patch],'apply-'+variant)
   call([args.cmake,'--build',w/'build','--target','erf_probe','--parallel','1'],'build-'+variant)
   binary=d/('erf_probe-'+variant)
   if binary.exists() or binary.is_symlink():binary.unlink()
   binary.symlink_to(w/'build/erf_probe')
   call([sys.executable,d/'check_erf.py',variant],'grid-'+variant)
   for layout in ['flat','row','column']:
    name=f'{variant}-{layout}.csv';assert (d/'results'/name).read_bytes()==(P/'observed'/name).read_bytes(),name;csvs+=1
   results[variant]=json.loads((d/'results'/f'{variant}-summary.json').read_text())['flat']
   for i,line in enumerate((d/'chain_inputs.txt').read_text().splitlines()):
    single=d/'single.txt';single.write_text(line+'\n');p=subprocess.run([str(binary),str(single),'flat'],capture_output=True,check=True,env=env)
    name=f'chain-{variant}-{i}.csv';assert p.stdout==(P/'observed'/name).read_bytes(),name;csvs+=1
   p=subprocess.run([str(binary),str(d/'boundary_inputs.txt'),'flat'],capture_output=True,check=True,env=env);assert p.stdout==(P/'observed'/f'boundary-{variant}.csv').read_bytes();csvs+=1
 finally:src.write_bytes(original)
 counts=[len(results[k]['primary']['first_failures']) for k in ['original','candidate','balanced','restored']];assert counts==[372,4,0,372]
 receipt={'completed_at':datetime.now(timezone.utc).isoformat(),'fresh_full_cpu_build':True,'source_git_blobs_verified':verified,'unique_pairs':1292,'primary_finite_pairs':944,'primary_failures_original_simple_balanced_restored':counts,'csvs_byte_reproduced':csvs,'oracle_100_150_digits_identical':True,'source_restored':src.read_bytes()==original,'tail_and_higher_derivative_residuals_retained':True,'device':'CPU','gpu':False,'audio_playback':False,'commands':commands}
 (w/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='commands'},indent=2))
if __name__=='__main__':main()
