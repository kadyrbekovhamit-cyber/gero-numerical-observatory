"""Compile isolated C++ variants sequentially; reuse the existing CPU archive."""
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

def method(source,name):
    start=source.index('std::vector<array> '+name+'::vjp(')
    return source[start:source.index('\n}',start)+2]

def patch(source):
    for name in ('GatherMM','GatherQMM'):
        old=method(source,name)
        new=old.replace('  bool sorted = left_sorted_ || right_sorted_;\n','')
        end=new.index('\n',new.index('  bool no_broadcast ='))
        new=new[:end]+'''\n  // The segmented VJP requires implicit left indices without broadcasting.
  bool sorted = right_sorted_ && no_broadcast;'''+new[end:]
        assert old!=new
        source=source.replace(old,new,1)
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
    current=(ROOT/'upstream-mlx_primitives.cpp').read_text()
    baseline=subprocess.run(['git','show','ce916dbbcaa88e433b6fd1e60a17f766d49c27fe:mlx/primitives.cpp'],cwd=REPO,capture_output=True,text=True,check=True).stdout
    matches={name:method(current,name)==method(baseline,name) for name in ('GatherMM','GatherQMM')}
    # The shared helper is also required for interpreting the native result.
    def helper(text):
        start=text.index('array gather_mm_grad(')
        return text[start:text.index('\n}',start)+2]
    matches['gather_mm_grad']=helper(current)==helper(baseline)
    assert matches['GatherMM'] and matches['gather_mm_grad']
    qdiff=''.join(difflib.unified_diff(method(baseline,'GatherQMM').splitlines(True),
                                    method(current,'GatherQMM').splitlines(True)))
    (ROOT/'GatherQMM-baseline-to-upstream.diff').write_text(qdiff)
    # Current main adds global_scale support. In affine mode that argument is
    # absent; the suspect guard and all three floating-point gradient paths remain.
    for name in ('GatherMM','GatherQMM'):
        assert 'bool sorted = left_sorted_ || right_sorted_;' in method(current,name)
    for name,text in [('baseline-primitives.cpp',baseline),('patched-baseline-primitives.cpp',patch(baseline)),
                      ('patched-upstream-primitives.cpp',patch(current))]:
        (ROOT/name).write_text(text)
    diff=''.join(difflib.unified_diff(current.splitlines(True),patch(current).splitlines(True),
                                     fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp'))
    (ROOT/'gather-vjp-preconditions.patch').write_text(diff)
    metadata=json.loads((ROOT/'source-metadata.json').read_text())
    metadata.update(baseline_revision=subprocess.run(['git','rev-parse','ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip(),
                    relevant_methods_match_current_source=matches,archive=str(ARCHIVE),
                    GatherQMM_current_difference='Current main adds global_scale support and index-argument checks; see saved diff. Tests use baseline affine mode only.',
                    archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                    build='Pristine/patched primitives.cpp linked ahead of reused CPU-only libmlx.a; not a full clean current-main build.')
    (ROOT/'source-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    entry=next(c for c in json.loads((BUILD/'compile_commands.json').read_text()) if c['file'].endswith('/mlx/primitives.cpp'))
    args=shlex.split(entry['command']);base=['-O0' if a=='-O3' else a for a in args[:args.index('-o')]]
    test=ROOT/'native_regression.o'
    run(base+['-o',str(test),'-c',str(ROOT/'native_regression.cpp')],'compile-test')
    for label,name in [('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp')]:
        obj=ROOT/(label+'.o');binary=ROOT/('native-'+label)
        run(base+['-o',str(obj),'-c',str(ROOT/name)],'compile-'+label)
        run(['/usr/bin/c++',str(test),str(obj),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-'+label)
        result=run([str(binary)],'run-'+label)
        assert result.returncode==(1 if label=='before' else 0)

if __name__=='__main__':main()
