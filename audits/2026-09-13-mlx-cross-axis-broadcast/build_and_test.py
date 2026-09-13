"""Build pinned MLX source and run original/candidate native CPU regressions."""
import argparse, hashlib, json, os, resource, subprocess, tarfile, time, urllib.request
from pathlib import Path
PIN='ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'
p=argparse.ArgumentParser()
p.add_argument('--deps',help='Optional existing CMake _deps containing fmt-src and json-src')
p.add_argument('--reuse',action='store_true',help='Reuse the pinned archive and build directory')
a=p.parse_args();here=Path(__file__).resolve().parent;root=here/'clean-build';root.mkdir(exist_ok=True)
archive=root/'source.tar.gz';src=root/'mlx'
if not archive.exists():
    with urllib.request.urlopen('https://api.github.com/repos/ml-explore/mlx/tarball/'+PIN,timeout=60) as r:archive.write_bytes(r.read())
if not a.reuse or not src.exists():
    src.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            rel=Path(*Path(m.name).parts[1:])
            if '..' in rel.parts or rel.is_absolute():raise RuntimeError('Unsafe archive path')
            if m.isdir():(src/rel).mkdir(parents=True,exist_ok=True)
            elif m.isfile():
                target=src/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
baseline=(here/'baseline-linalg.cpp').read_bytes()
assert (src/'mlx/linalg.cpp').read_bytes()==baseline or a.reuse
(root/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.25)
project(cross_axis_audit LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
add_subdirectory(mlx mlx-build)
add_executable(cross_regression ../native_regression.cpp)
target_link_libraries(cross_regression PRIVATE mlx nlohmann_json::nlohmann_json)
''')
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['cmake','-S',str(root),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG']
if a.deps:
    for n in ['FMT','JSON']:cmd+=['-DFETCHCONTENT_SOURCE_DIR_'+n+'='+str(Path(a.deps).resolve()/(n.lower()+'-src'))]
for n in ['BUILD_METAL','BUILD_CUDA','BUILD_TESTS','BUILD_EXAMPLES','BUILD_BENCHMARKS','BUILD_PYTHON_BINDINGS','BUILD_GGUF','BUILD_SAFETENSORS','USE_CCACHE']:cmd+=['-DMLX_'+n+'=OFF']
records=[]
with (here/'clean-build.log').open('w') as log:
    subprocess.run(cmd,env=env,stdout=log,stderr=log,check=True)
    for variant in ['before','after']:
        (src/'mlx/linalg.cpp').write_bytes(baseline)
        if variant=='after':subprocess.run(['git','apply',str(here/'cross-axis-broadcast.patch')],cwd=src,stdout=log,stderr=log,check=True)
        subprocess.run(['cmake','--build',str(root/'build'),'--parallel','1'],env=env,stdout=log,stderr=log,check=True)
        start=time.monotonic();u=resource.getrusage(resource.RUSAGE_CHILDREN)
        x=subprocess.run([str(root/'build/cross_regression'),str(here/'cases.json')],env=env,capture_output=True,text=True,timeout=45)
        v=resource.getrusage(resource.RUSAGE_CHILDREN)
        (here/('clean-'+variant+'.jsonl')).write_text(x.stdout);(here/('clean-'+variant+'.stderr')).write_text(x.stderr)
        row=dict(variant=variant,exit=x.returncode,wall_seconds=time.monotonic()-start,cpu_seconds=v.ru_utime+v.ru_stime-u.ru_utime-u.ru_stime,summary=json.loads(x.stdout.splitlines()[-1]))
        print(json.dumps(row),flush=True);records.append(row)
        if x.returncode != (1 if variant=='before' else 0):raise RuntimeError('Unexpected regression result; inspect logs')
(here/'clean-results.json').write_text(json.dumps(records,indent=2)+'\n')
(here/'source-archive.json').write_text(json.dumps(dict(commit=PIN,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source='https://api.github.com/repos/ml-explore/mlx/tarball/'+PIN),indent=2)+'\n')
