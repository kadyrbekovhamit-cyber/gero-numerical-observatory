"""Build pinned MLX source and reproduce the bounded CPU comparison audit."""
import argparse,hashlib,json,os,resource,subprocess,tarfile,time,urllib.request
from pathlib import Path
PIN='ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'
p=argparse.ArgumentParser();p.add_argument('--deps');p.add_argument('--work-dir');p.add_argument('--archive');a=p.parse_args()
here=Path(__file__).resolve().parent;root=Path(a.work_dir).resolve() if a.work_dir else here/'repro-build';root.mkdir(exist_ok=True,parents=True)
archive=Path(a.archive).resolve() if a.archive else root/'source.tar.gz'
if not archive.exists():
 with urllib.request.urlopen('https://api.github.com/repos/ml-explore/mlx/tarball/'+PIN,timeout=60) as f:archive.write_bytes(f.read())
src=root/'mlx';src.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  rel=Path(*Path(m.name).parts[1:])
  if '..' in rel.parts or rel.is_absolute():raise ValueError('Unsafe source path')
  if m.isdir():(src/rel).mkdir(parents=True,exist_ok=True)
  elif m.isfile():
   target=src/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
baseline=(here/'baseline-mlx_ops.cpp').read_bytes();assert (src/'mlx/ops.cpp').read_bytes()==baseline
(root/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.25)\nproject(isclose_audit LANGUAGES C CXX)\nset(CMAKE_CXX_STANDARD 20)\nadd_subdirectory(mlx mlx-build)\nadd_executable(isclose_regression "'+str(here/'native_regression.cpp')+'")\ntarget_link_libraries(isclose_regression PRIVATE mlx nlohmann_json::nlohmann_json)\nadd_executable(isclose_limits "'+str(here/'limitations.cpp')+'")\ntarget_link_libraries(isclose_limits PRIVATE mlx)\n')
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['cmake','-S',str(root),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG']
if a.deps:
 for n in ['FMT','JSON']:cmd+=['-DFETCHCONTENT_SOURCE_DIR_'+n+'='+str(Path(a.deps).resolve()/(n.lower()+'-src'))]
for n in ['BUILD_METAL','BUILD_CUDA','BUILD_TESTS','BUILD_EXAMPLES','BUILD_BENCHMARKS','BUILD_PYTHON_BINDINGS','BUILD_GGUF','BUILD_SAFETENSORS','USE_CCACHE']:cmd+=['-DMLX_'+n+'=OFF']
records=[]
with (here/'reproduction-build.log').open('w') as log:
 subprocess.run(cmd,env=env,stdout=log,stderr=log,check=True)
 for variant in ['before','after']:
  (src/'mlx/ops.cpp').write_bytes(baseline)
  if variant=='after':subprocess.run(['git','apply',str(here/'isclose-integer-cpu-prototype.patch')],cwd=src,stdout=log,stderr=log,check=True)
  subprocess.run(['cmake','--build',str(root/'build'),'--parallel','1'],env=env,stdout=log,stderr=log,check=True)
  start=time.monotonic();u=resource.getrusage(resource.RUSAGE_CHILDREN)
  x=subprocess.run([str(root/'build/isclose_regression'),str(here/'cases.json')],env=env,capture_output=True,text=True,timeout=45)
  v=resource.getrusage(resource.RUSAGE_CHILDREN)
  (here/('repro-'+variant+'.jsonl')).write_text(x.stdout);(here/('repro-'+variant+'.stderr')).write_text(x.stderr)
  limit=subprocess.run([str(root/'build/isclose_limits')],env=env,capture_output=True,text=True,timeout=20,check=True)
  (here/('limits-'+variant+'.log')).write_text(limit.stdout+limit.stderr)
  row=dict(variant=variant,exit=x.returncode,wall_seconds=time.monotonic()-start,cpu_seconds=v.ru_utime+v.ru_stime-u.ru_utime-u.ru_stime,summary=json.loads(x.stdout.splitlines()[-1]));records.append(row);print(json.dumps(row),flush=True)
  if x.returncode != (1 if variant=='before' else 0):raise RuntimeError('Unexpected regression result')
(here/'reproduction-results.json').write_text(json.dumps(records,indent=2)+'\n')
