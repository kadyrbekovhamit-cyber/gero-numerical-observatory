"""Sequential CPU build of one audited translation unit against a saved archive."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ['MLX_SOURCE_ROOT']).resolve()
BUILD = Path(os.environ['MLX_CPU_BUILD']).resolve()
ARCHIVE = BUILD / 'mlx-build/libmlx.a'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (180,180))


def method(source):
    start = source.index('std::vector<array> MaskedScatter::vjp(')
    end = source.index('\n}', start) + 2
    return source[start:end]


def patch(source):
    original = method(source)
    start = original.index('      const array mask_flat =')
    end = original.index('\n    } else {', start)
    replacement = '''      if (src.size() == 0 || mask_b.size() == 0) {
        vjps.push_back(zeros_like(src, s));
        continue;
      }

      const auto batches = src.shape(0);
      const auto source_batch_size = src.size() / batches;
      const auto mask_rows = reshape(mask_b, {batches, -1}, s);
      auto idx_src = cumsum(astype(mask_rows, int64, s), 1, false, false, s);
      idx_src = where(mask_rows, idx_src, array(0, int64), s);
      const auto offsets = multiply(
          reshape(arange(batches, int64, s), {batches, 1}, s),
          array(static_cast<int64_t>(source_batch_size)),
          s);
      idx_src = flatten(add(idx_src, offsets, s), s);
      const auto cotan_src = flatten(
          where(mask_b, cotan, array(0, cotan.dtype()), s), s);
      auto gsrc_flat = scatter_add(
          zeros({static_cast<int>(src.size())}, cotan.dtype(), s),
          idx_src,
          reshape(cotan_src, {static_cast<int>(idx_src.size()), 1}, s),
          0,
          s);
      vjps.push_back(reshape(gsrc_flat, src.shape(), s));'''
    return source.replace(original, original[:start] + replacement + original[end:], 1)


records = []
def run(command, label):
    started = time.monotonic()
    p = subprocess.run(command, cwd=BUILD, capture_output=True, text=True, timeout=60)
    (ROOT / (label+'.log')).write_text(p.stdout+p.stderr)
    records.append(dict(label=label,command=command,returncode=p.returncode,wall_seconds=time.monotonic()-started))
    (ROOT/'build-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,p.returncode,round(records[-1]['wall_seconds'],4),flush=True)
    if label.startswith('run-'):
        print('\n'.join(line for line in p.stdout.splitlines() if line.startswith('SUMMARY')),flush=True)
    elif p.returncode:
        print(p.stderr[-7000:],flush=True)
        raise SystemExit(p.returncode)
    return p


def main():
    current = (ROOT/'upstream-primitives.cpp').read_text()
    baseline = subprocess.run(['git','show','ce916dbbcaa88e433b6fd1e60a17f766d49c27fe:mlx/primitives.cpp'],cwd=REPO,capture_output=True,text=True,check=True).stdout
    assert method(current) == method(baseline)
    for name,text in [('baseline-primitives.cpp',baseline),('patched-baseline-primitives.cpp',patch(baseline)),('patched-upstream-primitives.cpp',patch(current))]:
        (ROOT/name).write_text(text)
    (ROOT/'masked-scatter-batch-vjp.patch').write_text(''.join(difflib.unified_diff(current.splitlines(True),patch(current).splitlines(True),fromfile='a/mlx/primitives.cpp',tofile='b/mlx/primitives.cpp')))
    metadata = json.loads((ROOT/'source-metadata.json').read_text())
    metadata.update(baseline_revision=subprocess.run(['git','rev-parse','ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip(),
                    MaskedScatter_vjp_matches_current_source=True,archive=str(ARCHIVE),
                    archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                    build='Pristine/patched primitives.cpp separately compiled and linked ahead of reused CPU-only libmlx.a; not a full clean main build.')
    (ROOT/'source-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    entry = next(c for c in json.loads((BUILD/'compile_commands.json').read_text()) if c['file'].endswith('/mlx/primitives.cpp'))
    args = shlex.split(entry['command'])
    base = ['-O0' if arg=='-O3' else arg for arg in args[:args.index('-o')]]
    test = ROOT/'native_regression.o'
    run(base+['-o',str(test),'-c',str(ROOT/'native_regression.cpp')],'compile-test')
    for label,name in [('before','baseline-primitives.cpp'),('after','patched-baseline-primitives.cpp')]:
        obj = ROOT/(label+'.o'); binary = ROOT/('native-'+label)
        run(base+['-o',str(obj),'-c',str(ROOT/name)],'compile-'+label)
        run(['/usr/bin/c++',str(test),str(obj),str(ARCHIVE),'-framework','Accelerate','-o',str(binary)],'link-'+label)
        result = run([str(binary)],'run-'+label)
        assert result.returncode == (1 if label=='before' else 0)


if __name__ == '__main__':
    main()
