#!/usr/bin/env python3
"""Reproduce both variants from pinned MLX source, with one build job."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

PIN = 'ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True, help='Local MLX Git repository containing the pinned commit')
p.add_argument('--deps', help='Optional CMake _deps with fmt-src and json-src')
p.add_argument('--reuse', action='store_true')
a = p.parse_args()
here = Path(__file__).resolve().parent
root = here / 'clean-build'
source = root / 'mlx'
source.mkdir(parents=True, exist_ok=True)
if not a.reuse:
    data = subprocess.check_output(['git', 'archive', PIN], cwd=a.repo)
    subprocess.run(['tar', '-x', '-C', str(source)], input=data, check=True)
baseline = subprocess.check_output(['git', 'show', PIN + ':mlx/linalg.cpp'], cwd=a.repo)
(root / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.25)
project(integer_norm_audit LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
add_subdirectory(mlx mlx-build)
add_executable(norm_regression ../native_regression.cpp)
target_link_libraries(norm_regression PRIVATE mlx)
add_executable(norm_limits ../limitation_probe.cpp)
target_link_libraries(norm_limits PRIVATE mlx)
''')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
cmd = ['cmake', '-S', str(root), '-B', str(root / 'build'), '-G', 'Ninja',
       '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG']
if a.deps:
    for name in ['FMT', 'JSON']:
        cmd.append('-DFETCHCONTENT_SOURCE_DIR_' + name + '=' +
                   str(Path(a.deps).resolve() / (name.lower() + '-src')))
for name in ['BUILD_METAL', 'BUILD_CUDA', 'BUILD_TESTS', 'BUILD_EXAMPLES',
             'BUILD_BENCHMARKS', 'BUILD_PYTHON_BINDINGS', 'BUILD_GGUF',
             'BUILD_SAFETENSORS', 'USE_CCACHE']:
    cmd.append('-DMLX_' + name + '=OFF')
records = []
with (here / 'clean-build.log').open('w') as log:
    subprocess.run(cmd, stdout=log, stderr=log, check=True, env=env)
    for variant in ['before', 'after']:
        (source / 'mlx/linalg.cpp').write_bytes(baseline)
        if variant == 'after':
            subprocess.run(['git', 'apply', str(here / 'norm-integer-promotion.patch')],
                           cwd=source, stdout=log, stderr=log, check=True)
        subprocess.run(['cmake', '--build', str(root / 'build'), '--parallel', '1'],
                       stdout=log, stderr=log, check=True, env=env)
        start = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        out = subprocess.run([str(root / 'build/norm_regression')],
                             capture_output=True, text=True, env=env, timeout=30)
        end_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        (here / ('clean-' + variant + '.jsonl')).write_text(out.stdout)
        (here / ('clean-' + variant + '.stderr')).write_text(out.stderr)
        limits = subprocess.run([str(root / 'build/norm_limits')],
                                capture_output=True, text=True, env=env, timeout=15, check=True)
        (here / ('clean-' + variant + '-limits.jsonl')).write_text(limits.stdout)
        record = {'variant': variant, 'exit': out.returncode,
                  'summary': json.loads(out.stdout.splitlines()[-1]),
                  'wall_seconds': time.monotonic() - start,
                  'cpu_seconds': end_usage.ru_utime + end_usage.ru_stime - usage.ru_utime - usage.ru_stime}
        records.append(record)
        print(json.dumps(record), flush=True)
        if out.returncode != (1 if variant == 'before' else 0):
            raise SystemExit('Unexpected result; inspect the saved logs.')
(here / 'clean-results-summary.json').write_text(json.dumps(records, indent=2) + '\n')
