"""Compile the audited C++ file sequentially against an existing CPU archive."""

import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT
REPO=Path(os.environ["MLX_SOURCE_ROOT"]).resolve()
BUILD=Path(os.environ["MLX_CPU_BUILD"]).resolve()
ARCHIVE=BUILD/'mlx-build/libmlx.a'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(180,180))


NEW_BODY='''  } else if (reduce_type_ == Scan::Prod) {
    auto s = stream();
    auto in = conjugate(primals[0], s);
    auto cotan = cotangents[0];
    int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
    int n = in.shape(axis);
    if (n < 2) {
      return {inclusive_ ? cotan : zeros_like(in, s)};
    }

    auto shift = [&](const array& a, int distance) {
      Shape start(a.ndim(), 0);
      Shape stop = a.shape();
      Shape padding_shape = a.shape();
      padding_shape[axis] = distance;
      if (reverse_) {
        stop[axis] -= distance;
      } else {
        start[axis] = distance;
      }
      auto part = slice(a, start, stop, s);
      auto padding = zeros(padding_shape, a.dtype(), s);
      return reverse_ ? concatenate({padding, part}, axis, s)
                      : concatenate({part, padding}, axis, s);
    };

    // Compose affine suffix recurrences without division or tests on input zeros.
    auto a = shift(in, 1);
    auto b = inclusive_ ? cotan : shift(cotan, 1);
    for (int64_t distance = 1; distance < n; distance *= 2) {
      b = add(b, multiply(a, shift(b, distance), s), s);
      if (2 * distance < n) {
        a = multiply(a, shift(a, distance), s);
      }
    }
    return {multiply(cumprod(in, axis, reverse_, false, s), b, s)};
'''


def method(source):
    start=source.index('std::vector<array> Scan::vjp(')
    return source[start:source.index('\n}',start)+2]


def patch(source):
    start=source.index('  } else if (reduce_type_ == Scan::Prod)',source.index('std::vector<array> Scan::vjp('))
    end=source.index('  } else {\n    // Cumulative max/min:',start)
    return source[:start]+NEW_BODY+source[end:]


records=[]
def run(command,label,timeout=120):
    begin=time.monotonic()
    p=subprocess.run(command,cwd=BUILD,capture_output=True,text=True,timeout=timeout)
    (ROOT/(label+'.log')).write_text(p.stdout+p.stderr)
    records.append(dict(label=label,command=command,returncode=p.returncode,wall_seconds=time.monotonic()-begin))
    (ROOT/'build-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,p.returncode,round(records[-1]['wall_seconds'],3),flush=True)
    if label.startswith('run-'):
        print('\n'.join(line for line in p.stdout.splitlines() if line.startswith('SUMMARY')),flush=True)
    elif p.returncode:
        print(p.stderr[-5000:],flush=True)
        raise SystemExit(p.returncode)
    return p


current=(PREVIOUS/'upstream-primitives.cpp').read_text()
baseline=subprocess.run(['git','show','HEAD:mlx/primitives.cpp'],cwd=REPO,capture_output=True,text=True,check=True).stdout
assert method(baseline)==method(current)
for name,source in [('upstream-primitives.cpp',current),('baseline-primitives.cpp',baseline),('patched-upstream-primitives.cpp',patch(current)),('patched-baseline-primitives.cpp',patch(baseline))]:
    (ROOT/name).write_text(source)
(ROOT/'cumprod-polynomial-vjp.patch').write_text(''.join(difflib.unified_diff(current.splitlines(True),patch(current).splitlines(True),fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp')))
meta={'pinned_revision':json.loads((PREVIOUS/'source-metadata.json').read_text())['pinned_revision'],
      'baseline_revision':subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip(),
      'Scan_vjp_matches_pinned_source':True,'archive':str(ARCHIVE),'archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256(current.encode()).hexdigest(),
      'source_retrieval':'Copied from the exact upstream snapshot fetched and hashed in the preceding audit on 2026-09-09.',
      'build':'New pristine/patched primitives.cpp linked ahead of reused CPU archive; not a full clean build of main.'}
(ROOT/'source-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
entry=next(c for c in json.loads((BUILD/'compile_commands.json').read_text()) if c['file'].endswith('/mlx/primitives.cpp'))
args=shlex.split(entry['command'])
base=['-O0' if arg=='-O3' else arg for arg in args[:args.index('-o')]]
test_obj=ROOT/'native_regression.o'
run(base+['-o',str(test_obj),'-c',str(ROOT/'native_regression.cpp')],'compile-test')
for label,name in [('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp')]:
    obj=ROOT/(label+'.o')
    binary=ROOT/('native-'+label)
    run(base+['-o',str(obj),'-c',str(ROOT/name)],'compile-'+label)
    run(['/usr/bin/c++',str(test_obj),str(obj),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-'+label)
    result=run([str(binary)],'run-'+label,60)
    assert result.returncode==(1 if label=='before' else 0),(label,result.returncode)
