"""Rebuild isolated CPU translation units sequentially; never rebuild the archive."""
import hashlib, json, os, resource, shlex, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ['MLX_SOURCE_ROOT']).resolve()
BUILD=Path(os.environ['MLX_CPU_BUILD']).resolve()
ARCHIVE=Path(os.environ.get('MLX_CPU_ARCHIVE',str(BUILD/'mlx-build/libmlx.a')))
OUT=ROOT/'rerun'; OUT.mkdir(exist_ok=True)
for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
records=[]
db=json.loads((BUILD/'compile_commands.json').read_text())
def flags(suffix):
    entry=next(x for x in db if x['file'].endswith(suffix))
    args=entry.get('arguments') or shlex.split(entry['command'])
    return ['-O0' if a=='-O3' else a for a in args[:args.index('-o')]]
def sanitize(v):
    if isinstance(v,str):
        for path,label in ((ROOT,'<AUDIT>'),(SOURCE,'<MLX_SOURCE>'),(BUILD,'<CPU_BUILD>')):
            v=v.replace(str(path),label)
        return v
    if isinstance(v,list): return [sanitize(x) for x in v]
    if isinstance(v,dict): return {k:sanitize(x) for k,x in v.items()}
    return v
def run(command,label,expected=0):
    before=resource.getrusage(resource.RUSAGE_CHILDREN); start=time.monotonic()
    def limit(): resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    p=subprocess.run(command,cwd=BUILD,capture_output=True,text=True,timeout=90,preexec_fn=limit)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    (OUT/(label+'.log')).write_text(sanitize(p.stdout+p.stderr))
    rows=[json.loads(x[5:]) for x in p.stdout.splitlines() if x.startswith('CASE ')]
    if rows: (OUT/(label+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    records.append(dict(label=label,command=sanitize(command),returncode=p.returncode,
                        wall_seconds=time.monotonic()-start,
                        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (OUT/'build-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,p.returncode,*[x for x in p.stdout.splitlines() if x.startswith(('SUMMARY','EVIDENCE'))],flush=True)
    if p.returncode!=expected: raise RuntimeError((p.stdout+p.stderr)[-3000:])

assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()=='ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'
assert subprocess.check_output(['git','show','HEAD:mlx/primitives.cpp'],cwd=SOURCE)==(ROOT/'baseline-primitives.cpp').read_bytes(), 'Saved primitives must match the pinned commit'
assert (SOURCE/'mlx/backend/cpu/simd/math.h').read_bytes()==(ROOT/'exp-control/baseline-math.h').read_bytes()
overlay=OUT/'overlay/mlx/backend/cpu/simd/math.h';overlay.parent.mkdir(parents=True,exist_ok=True)
overlay.write_bytes((ROOT/'exp-control/baseline-math.h').read_bytes())
run(['/usr/bin/patch','-d',str(OUT/'overlay'),'-p1','-i',str(ROOT/'exp-control/float64-exp.patch')],'apply-exp-control')
prim=flags('/mlx/primitives.cpp'); unary=flags('/mlx/backend/cpu/unary.cpp')
for label,name in [('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp'),('test','native_regression.cpp')]:
    run(prim+['-o',str(OUT/(label+'.o')),'-c',str(ROOT/name)],'compile-'+label)
for label in ('before','after'):
    cmd=unary[:1]+(['-I'+str(OUT/'overlay')] if label=='after' else [])+unary[1:]
    run(cmd+['-o',str(OUT/('unary-'+label+'.o')),'-c',str(SOURCE/'mlx/backend/cpu/unary.cpp')],'compile-unary-'+label)
primary='scan-only' if (ROOT/'stable_scan_vjp.cpp.inc').exists() else 'vjp-only'
for label,p,u,expected in [('before','before','before',1),(primary,'after','before',1),('exp-only','before','after',1),('combined','after','after',0)]:
    binary=OUT/('native-'+label)
    run(['/usr/bin/c++',str(OUT/'test.o'),str(OUT/(p+'.o')),str(OUT/('unary-'+u+'.o')),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-'+label)
    run([str(binary)],'run-'+label,expected)
if (ROOT/'higher_order_regression.cpp').exists():
    run(prim+['-o',str(OUT/'higher.o'),'-c',str(ROOT/'higher_order_regression.cpp')],'compile-higher-order')
    binary=OUT/'native-higher-order'
    run(['/usr/bin/c++',str(OUT/'higher.o'),str(OUT/'after.o'),str(OUT/'unary-after.o'),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-higher-order')
    run([str(binary)],'run-higher-order')
report=dict(date='2026-09-10',archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
            baseline_revision='ce916dbbcaa88e433b6fd1e60a17f766d49c27fe',
            scope='Sequential isolated CPU translation-unit rebuild; no Metal; not a complete current-main build.',
            cpu_seconds=sum(x['cpu_seconds'] for x in records),records=records)
(OUT/'publication-rerun.json').write_text(json.dumps(report,indent=2)+'\n')
