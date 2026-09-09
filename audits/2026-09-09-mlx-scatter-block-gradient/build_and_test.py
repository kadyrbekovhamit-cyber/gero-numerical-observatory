"""Build only the audited translation unit, sequentially, CPU archive reused."""
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
    start=source.index('std::vector<array> '+name+'::vjp(')
    end=source.index('\n}',start)+2
    return source[start:end]


def patch(source):
    original=method(source,'Scatter')
    changed=original.replace('  std::vector<array> vjps;','''  std::optional<array> update_winners;
  std::optional<array> update_counts;
  if (reduce_type_ == Scatter::Max || reduce_type_ == Scatter::Min) {
    auto slice_sizes = Shape(
        updates.shape().end() - values.ndim(), updates.shape().end());
    update_winners = equal(
        updates, gather(result, indices, axes_, slice_sizes, stream()), stream());
    update_counts = scatter_add(
        zeros(result.shape(), uint32, stream()),
        indices,
        astype(*update_winners, uint32, stream()),
        axes_,
        stream());
  }

  std::vector<array> vjps;''',1)
    changed=changed.replace('''              equal(result, values, stream()),
              cotangents[0],''','''              logical_and(
                  equal(result, values, stream()),
                  equal(*update_counts, array(0, uint32), stream()),
                  stream()),
              cotangents[0],''',1)
    old='''          auto slice_sizes = cotangents[0].shape();
          for (auto ax : axes_) {
            slice_sizes[ax] = 1;
          }
          auto gathered_cotan =
              gather(cotangents[0], indices, axes_, slice_sizes, stream());
          auto gathered_result =
              gather(result, indices, axes_, slice_sizes, stream());
          vjps.push_back(
              multiply(gathered_cotan, gathered_result == updates, stream()));'''
    new='''          auto slice_sizes = Shape(
              updates.shape().end() - values.ndim(), updates.shape().end());
          auto gathered_cotan =
              gather(cotangents[0], indices, axes_, slice_sizes, stream());
          auto counts = gather(
              maximum(*update_counts, array(1, uint32), stream()),
              indices,
              axes_,
              slice_sizes,
              stream());
          vjps.push_back(multiply(
              divide(gathered_cotan, counts, stream()),
              *update_winners,
              stream()));'''
    assert old in changed
    changed=changed.replace(old,new,1)
    source=source.replace(original,changed,1)

    original=method(source,'SliceUpdate')
    old='''        case SliceUpdate::Max:
        case SliceUpdate::Min:
          vjps.push_back(where(
              equal(result, values, stream()),
              cotan,
              array(0, cotan.dtype()),
              stream()));
          break;'''
    new='''        case SliceUpdate::Max:
        case SliceUpdate::Min: {
          auto sliced_values =
              slice(values, start_indices_, end_indices_, strides_, stream());
          auto source_wins = reduce_type_ == SliceUpdate::Max
              ? greater(sliced_values, updates, stream())
              : less(sliced_values, updates, stream());
          auto sliced_cotan =
              slice(cotan, start_indices_, end_indices_, strides_, stream());
          vjps.push_back(slice_update(
              cotan,
              multiply(sliced_cotan, source_wins, stream()),
              start_indices_,
              end_indices_,
              strides_,
              stream()));
          break;
        }'''
    assert old in original
    return source.replace(original,original.replace(old,new,1),1)


records=[]
def block_patch(source):
    original=method(source,'Scatter')
    old='''          auto slice_sizes = cotangents[0].shape();
          for (auto ax : axes_) {
            slice_sizes[ax] = 1;
          }
          auto gathered_cotan ='''
    new='''          auto slice_sizes = Shape(
              updates.shape().end() - values.ndim(), updates.shape().end());
          auto gathered_cotan ='''
    assert original.count(old)==1
    return source.replace(original,original.replace(old,new,1),1)


def run(command,label,timeout=60):
    begin=time.monotonic()
    p=subprocess.run(command,cwd=BUILD,capture_output=True,text=True,timeout=timeout)
    (ROOT/(label+'.log')).write_text(p.stdout+p.stderr)
    records.append(dict(label=label,command=command,returncode=p.returncode,wall_seconds=time.monotonic()-begin))
    (ROOT/'build-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,p.returncode,round(records[-1]['wall_seconds'],4),flush=True)
    if label.startswith('run-'):
        print('\n'.join(x for x in p.stdout.splitlines() if x.startswith('SUMMARY')),flush=True)
    elif p.returncode:
        print(p.stderr[-6000:],flush=True)
        raise SystemExit(p.returncode)
    return p


current=(ROOT/'upstream-primitives.cpp').read_text()
baseline=(ROOT/'baseline-primitives.cpp').read_text()
matches={name:method(baseline,name)==method(current,name) for name in ['Scatter','SliceUpdate']}
assert all(matches.values()),matches
for name,text in [('baseline-primitives.cpp',baseline),('patched-baseline-primitives.cpp',patch(baseline)),('patched-upstream-primitives.cpp',patch(current))]:
    (ROOT/name).write_text(text)
(ROOT/'scatter-extrema-vjp.patch').write_text(''.join(difflib.unified_diff(current.splitlines(True),patch(current).splitlines(True),fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp')))
(ROOT/'scatter-block-extent-only.patch').write_text(''.join(difflib.unified_diff(current.splitlines(True),block_patch(current).splitlines(True),fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp')))
(ROOT/'block-only-baseline-primitives.cpp').write_text(block_patch(baseline))
metadata=json.loads((ROOT/'source-metadata.json').read_text())
metadata.update(baseline_revision='ce916dbbcaa88e433b6fd1e60a17f766d49c27fe',
                vjp_methods_match_current_source=matches,archive=str(ARCHIVE),archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                build='Pristine/patched primitives.cpp separately compiled and linked ahead of reused CPU-only libmlx.a; not a clean full build of main.')
(ROOT/'source-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
entry=next(c for c in json.loads((BUILD/'compile_commands.json').read_text()) if c['file'].endswith('/mlx/primitives.cpp'))
args=shlex.split(entry['command'])
base=['-O0' if a=='-O3' else a for a in args[:args.index('-o')]]
test_object=ROOT/'native_regression.o'
run(base+['-o',str(test_object),'-c',str(ROOT/'native_regression.cpp')],'compile-test')
for label,name in [('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp')]:
    obj=ROOT/(label+'.o'); binary=ROOT/('native-'+label)
    run(base+['-o',str(obj),'-c',str(ROOT/name)],'compile-'+label)
    run(['/usr/bin/c++',str(test_object),str(obj),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-'+label)
    result=run([str(binary)],'run-'+label)
    assert result.returncode==(1 if label=='before' else 0),(label,result.returncode)

obj=ROOT/'block-only.o'; binary=ROOT/'native-block-only'
run(base+['-o',str(obj),'-c',str(ROOT/'block-only-baseline-primitives.cpp')],'compile-block-only')
run(['/usr/bin/c++',str(test_object),str(obj),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-block-only')
for label,name in [('before-strict','native-before'),('block-only-strict','native-block-only')]:
    result=run([str(ROOT/name),'--strict-blocks'],'run-'+label)
    assert result.returncode==(1 if label=='before-strict' else 0)
