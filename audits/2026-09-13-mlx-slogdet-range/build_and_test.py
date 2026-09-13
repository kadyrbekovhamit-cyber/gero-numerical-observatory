#!/usr/bin/env python3
"""Build both variants from a pinned local MLX Git repository, one job."""
import argparse
import os
from pathlib import Path
import subprocess

PIN = 'ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--deps', help='Optional existing CMake _deps directory with fmt-src and json-src')
p.add_argument('--reuse', action='store_true', help='Reuse this script\'s existing clean build')
a = p.parse_args()
here = Path(__file__).resolve().parent
root = here / 'clean-build'
root.mkdir(exist_ok=True)
(root / 'mlx').mkdir(exist_ok=True)
if not a.reuse:
    archive = subprocess.check_output(['git', 'archive', PIN], cwd=a.repo)
    subprocess.run(['tar', '-x', '-C', str(root / 'mlx')], input=archive, check=True)
baseline = subprocess.check_output(['git', 'show', PIN+':mlx/linalg.cpp'], cwd=a.repo)
(root / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.25)
project(slogdet_audit LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
add_subdirectory(mlx mlx-build)
add_executable(slogdet_regression ../native_regression.cpp)
target_link_libraries(slogdet_regression PRIVATE mlx)
''')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
cmd = ['cmake', '-S', str(root), '-B', str(root/'build'), '-G', 'Ninja',
       '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG']
if a.deps:
    for name in ['FMT', 'JSON']:
        cmd.append('-DFETCHCONTENT_SOURCE_DIR_' + name + '=' + str(Path(a.deps).resolve()/(name.lower()+'-src')))
for name in ['BUILD_METAL', 'BUILD_CUDA', 'BUILD_TESTS', 'BUILD_EXAMPLES',
             'BUILD_BENCHMARKS', 'BUILD_PYTHON_BINDINGS', 'BUILD_GGUF',
             'BUILD_SAFETENSORS', 'USE_CCACHE']:
    cmd.append('-DMLX_' + name + '=OFF')
with (here/'variant-build.log').open('w') as log:
    subprocess.run(cmd, stdout=log, stderr=log, check=True, env=env)
    for variant, patch in [('before', None), ('row-scaling-rejected', 'slogdet-row-scaling.patch'), ('lu-forward-candidate', 'slogdet-lu-forward.patch')]:
        (root/'mlx/mlx/linalg.cpp').write_bytes(baseline)
        if patch:
            subprocess.run(['git', 'apply', str(here/patch)],
                           cwd=root/'mlx', stdout=log, stderr=log, check=True)
        subprocess.run(['cmake', '--build', str(root/'build'), '--parallel', '1'],
                       stdout=log, stderr=log, check=True, env=env)
        with (here/('clean-'+variant+'.jsonl')).open('w') as output:
            result = subprocess.run([str(root/'build/slogdet_regression')],
                                    stdout=output, stderr=subprocess.STDOUT, env=env)
        last = (here/('clean-'+variant+'.jsonl')).read_text().splitlines()[-1]
        print(variant, result.returncode, last, flush=True)
        if result.returncode != 1:
            raise SystemExit('Unexpected test exit status; inspect the logs.')
        if variant == 'before':
            with (here/'clean-widen-f32.jsonl').open('w') as output:
                result = subprocess.run([str(root/'build/slogdet_regression'), '--widen-f32'],
                                        stdout=output, stderr=subprocess.STDOUT, env=env)
            print('widen-f32', result.returncode, (here/'clean-widen-f32.jsonl').read_text().splitlines()[-1], flush=True)
