"""Build only the audited translation unit; reuse an existing CPU archive."""

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

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ["MLX_SOURCE_ROOT"]).resolve()
BUILD = Path(os.environ["MLX_CPU_BUILD"]).resolve()
ARCHIVE = BUILD / "mlx-build/libmlx.a"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
try:
    os.nice(10)
except PermissionError:
    print('Sandbox prevents lowering process priority; builds remain sequential.', flush=True)


def patch(source):
    for name in ("Cos", "ArcCos", "ArcSin", "ArcTan", "ArcCosh", "ArcSinh", "ArcTanh"):
        pattern = r"(std::vector<array> " + name + r"::vjp\([\s\S]*?\) \{)[\s\S]*?\n\}"
        body = "\n  return {conjugate(\n      jvp(primals, {conjugate(cotangents[0], stream())}, argnums)[0],\n      stream())};\n}"
        source, count = re.subn(pattern, lambda m:m[1]+body, source, count=1)
        assert count==1, name
    begin = source.index("std::vector<array> ArcCosh::jvp(")
    end = source.index("\n}", begin)
    original = source[begin:end]
    before = "  array one = array(1., primals[0].dtype());"
    extra = """  if (primals[0].dtype() == complex64) {
    auto left = sqrt(subtract(primals[0], one, stream()), stream());
    auto right = sqrt(add(primals[0], one, stream()), stream());
    return {divide(tangents[0], multiply(left, right, stream()), stream())};
  }"""
    assert before in original
    source = source[:begin]+original.replace(before,before+"\n"+extra,1)+source[end:]
    return source


def method(source,name,kind):
    start=source.index("std::vector<array> "+name+"::"+kind+"(")
    return source[start:source.index("\n}",start)+2]


def run(command,label,timeout=150):
    started=time.monotonic()
    p=subprocess.run(command,cwd=BUILD,capture_output=True,text=True,timeout=timeout)
    (ROOT/(label+'.log')).write_text(p.stdout+p.stderr)
    row={'command':command,'returncode':p.returncode,'wall_seconds':time.monotonic()-started}
    records.append(row)
    (ROOT/'build-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,p.returncode,round(row['wall_seconds'],3),flush=True)
    if label.startswith('run-'):
        print('\n'.join(line for line in p.stdout.splitlines() if line.startswith(('SUMMARY','KIND','BRANCH','REAL_LOSS'))),flush=True)
    elif p.returncode:
        print(p.stderr[-6000:],flush=True)
        raise SystemExit(p.returncode)
    return p


records=[]
baseline=subprocess.run(['git','show','HEAD:mlx/primitives.cpp'],cwd=REPO,capture_output=True,text=True,check=True).stdout
base_revision=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
current=(ROOT/'upstream-primitives.cpp').read_text()
for name in ('Cos','ArcCos','ArcSin','ArcTan','ArcCosh','ArcSinh','ArcTanh','Exp','Sin','Log'):
    for kind in ('vjp','jvp'):
        assert method(baseline,name,kind)==method(current,name,kind),(name,kind)
(ROOT/'baseline-primitives.cpp').write_text(baseline)
(ROOT/'patched-baseline-primitives.cpp').write_text(patch(baseline))
(ROOT/'patched-upstream-primitives.cpp').write_text(patch(current))
(ROOT/'complex-derivatives.patch').write_text(''.join(difflib.unified_diff(current.splitlines(True),patch(current).splitlines(True),fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp')))
provenance={'baseline_revision':base_revision,'current_source_revision':json.loads((ROOT/'source-metadata.json').read_text())['revision'],
            'tested_vjp_jvp_bodies_equal_current':True,'archive':str(ARCHIVE),'archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
            'build_kind':'Pristine baseline or patched primitives.cpp linked before an existing CPU-only archive; no fresh full build of current main.'}
(ROOT/'native-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
entry=next(c for c in json.loads((BUILD/'compile_commands.json').read_text()) if c['file'].endswith('/mlx/primitives.cpp'))
args=shlex.split(entry['command'])
base=args[:args.index('-o')]
base=['-O0' if a=='-O3' else a for a in base]
test_obj=ROOT/'native_regression.o'
run(base+['-o',str(test_obj),'-c',str(ROOT/'native_regression.cpp')],'compile-test')
for label,filename in [('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp')]:
    obj=ROOT/(label+'.o')
    run(base+['-o',str(obj),'-c',str(ROOT/filename)],'compile-'+label)
    binary=ROOT/('native-'+label)
    run(['/usr/bin/c++',str(test_obj),str(obj),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-'+label)
    result=run([str(binary)],'run-'+label,30)
    assert result.returncode==(1 if label=='before' else 0),(label,result.returncode)
