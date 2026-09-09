"""Compile two isolated primitives.cpp variants, sequentially, for CPU tests."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ['MLX_SOURCE_ROOT']).resolve()
BUILD=Path(os.environ['MLX_CPU_BUILD']).resolve()
ARCHIVE=BUILD/'mlx-build/libmlx.a'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(180,180))


def method(source, name):
    start=source.index('std::vector<array> Hadamard::'+name+'(')
    return source[start:source.index('\n}',start)+2]


def patch(source):
    old=method(source,'vjp')
    replacement='''  auto [n, m] = decompose_hadamard(cotangents[0].shape(-1));
  if (m == 1 || m == 12) {
    return jvp(primals, cotangents, argnums);
  }

  std::vector<float> entries;
  entries.reserve(m * m);
  auto matrix_text = hadamard_matrices().at(m);
  for (char c : matrix_text) {
    if (c == '+' || c == '-') {
      entries.push_back(c == '+' ? 1.0f : -1.0f);
    }
  }
  auto matrix = astype(
      array(entries.data(), {m, m}, float32), cotangents[0].dtype(), stream());
  auto blocks = unflatten(cotangents[0], -1, {m, n}, stream());
  auto transposed = matmul(swapaxes(blocks, -1, -2, stream()), matrix, stream());
  transposed = swapaxes(transposed, -1, -2, stream());
  return {reshape(
      hadamard_transform(transposed, scale_, stream()),
      cotangents[0].shape(),
      stream())};'''
    new=old.replace('  return jvp(primals, cotangents, argnums);',replacement)
    assert old!=new
    source=source.replace(old,new,1)
    return source.replace('#include "mlx/backend/common/utils.h"',
                          '#include "mlx/backend/common/hadamard.h"\n#include "mlx/backend/common/utils.h"',1)


records=[]
def run(command,label):
    started=time.monotonic()
    p=subprocess.run(command,cwd=BUILD,capture_output=True,text=True,timeout=60)
    (ROOT/(label+'.log')).write_text(p.stdout+p.stderr)
    records.append(dict(label=label,command=command,returncode=p.returncode,wall_seconds=time.monotonic()-started))
    (ROOT/'build-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,p.returncode,round(records[-1]['wall_seconds'],4),flush=True)
    if label.startswith('run-'):
        print('\n'.join(line for line in p.stdout.splitlines() if line.startswith('SUMMARY')),flush=True)
    elif p.returncode:
        print(p.stderr[-6000:],flush=True)
        raise SystemExit(p.returncode)
    return p


def main():
    current=(ROOT/'upstream-primitives.cpp').read_text()
    baseline=subprocess.run(['git','show','ce916dbbcaa88e433b6fd1e60a17f766d49c27fe:mlx/primitives.cpp'],cwd=REPO,capture_output=True,text=True,check=True).stdout
    matches={name:method(current,name)==method(baseline,name) for name in ('vjp','jvp')}
    matrix_match=(REPO/'mlx/backend/common/hadamard.h').read_bytes()==(ROOT/'upstream-mlx_backend_common_hadamard.h').read_bytes()
    assert all(matches.values()) and matrix_match
    for name,text in [('baseline-primitives.cpp',baseline),('patched-baseline-primitives.cpp',patch(baseline)),('patched-upstream-primitives.cpp',patch(current))]:
        (ROOT/name).write_text(text)
    diff=''.join(difflib.unified_diff(current.splitlines(True),patch(current).splitlines(True),fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp'))
    (ROOT/'hadamard-adjoint.patch').write_text(diff)
    metadata=json.loads((ROOT/'source-metadata.json').read_text())
    metadata.update(baseline_revision=subprocess.run(['git','rev-parse','ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip(),
                    Hadamard_methods_match_current_source=matches,constant_matrices_match_current=matrix_match,
                    archive=str(ARCHIVE),archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                    build='Pristine/patched primitives.cpp linked ahead of reused CPU-only libmlx.a; not a full clean current-main build.')
    (ROOT/'source-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    entry=next(c for c in json.loads((BUILD/'compile_commands.json').read_text()) if c['file'].endswith('/mlx/primitives.cpp'))
    args=shlex.split(entry['command']);base=['-O0' if arg=='-O3' else arg for arg in args[:args.index('-o')]]
    test=ROOT/'native_regression.o'
    run(base+['-o',str(test),'-c',str(ROOT/'native_regression.cpp')],'compile-test')
    for label,name in [('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp')]:
        obj=ROOT/(label+'.o');binary=ROOT/('native-'+label)
        run(base+['-o',str(obj),'-c',str(ROOT/name)],'compile-'+label)
        run(['/usr/bin/c++',str(test),str(obj),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-'+label)
        result=run([str(binary)],'run-'+label)
        assert result.returncode==(1 if label=='before' else 0)


if __name__=='__main__':main()
