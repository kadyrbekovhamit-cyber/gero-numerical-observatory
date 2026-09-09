"""Sequential partial CPU build, with no changes to the existing checkout."""
import ast
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
REPO=Path(os.environ['MLX_SOURCE_ROOT'])
BUILD=Path(os.environ['MLX_CPU_BUILD'])
ARCHIVE=BUILD/'mlx-build/libmlx.a'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(180,180))


def method(source,name):
    start=source.index('std::vector<array> Power::'+name+'(')
    return source[start:source.index('\n}',start)+2]


def patch(source):
    old=method(source,'vjp')
    new=old.replace('''    if (arg == 0) {
      vjps.push_back(multiply(''','''    if (arg == 0) {
      const auto dtype = primals[0].dtype();
      auto safe_base = primals[0];
      if (dtype != complex64) {
        safe_base = where(
            logical_and(
                equal(primals[0], array(0, dtype), stream()),
                equal(primals[1], array(0, dtype), stream()),
                stream()),
            array(1, dtype),
            primals[0],
            stream());
      }
      vjps.push_back(multiply(''',1)
    new=new.replace('''          power(
              primals[0],''','''          power(
              safe_base,''',1)
    assert old!=new
    return source.replace(old,new,1)


def patch_tests(source):
    assert source.count('    def test_power_grad(self):')==2
    source=source.replace('    def test_power_grad(self):','    def test_power_grad_base(self):',1)
    before='    def test_eval_in_grad(self):'
    new='''    def test_power_fixed_integer_higher_grad_at_zero(self):
        for n in range(4):
            def fn(x, n=n):
                return mx.power(x, float(n))

            for _ in range(n + 1):
                fn = mx.grad(fn)
            self.assertEqual(fn(mx.array(0.0)).item(), 0.0)

    def test_power_mixed_grad_at_zero_exponent(self):
        def first_base(a, b):
            return mx.grad(lambda x: mx.power(x, b))(a)

        mixed = mx.grad(lambda b: first_base(mx.array(2.0), b))
        self.assertEqual(mixed(mx.array(0.0)).item(), 0.5)

'''
    assert before in source
    source=source.replace(before,new+before,1)
    ast.parse(source)
    return source


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
    assert all(matches.values())
    tests=(ROOT/'upstream-test_autograd.py').read_text()
    for name,text in [('baseline-primitives.cpp',baseline),('patched-baseline-primitives.cpp',patch(baseline)),('patched-upstream-primitives.cpp',patch(current)),('patched-test_autograd.py',patch_tests(tests))]:
        (ROOT/name).write_text(text)
    diff=''.join(difflib.unified_diff(current.splitlines(True),patch(current).splitlines(True),fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp'))
    diff+=''.join(difflib.unified_diff(tests.splitlines(True),patch_tests(tests).splitlines(True),fromfile='a/python/tests/test_autograd.py',tofile='b/python/tests/test_autograd.py'))
    (ROOT/'power-zero-derivatives.patch').write_text(diff)
    metadata=json.loads((ROOT/'source-metadata.json').read_text())
    metadata.update(baseline_revision=subprocess.run(['git','rev-parse','ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip(),
                    Power_methods_match_current_source=matches,archive=str(ARCHIVE),
                    archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                    build='Pristine/patched primitives.cpp linked ahead of reused CPU-only libmlx.a; not a full clean current-main build. Python test additions are syntax checked only.')
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
