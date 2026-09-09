"""Sequential isolated C++ regression; reuse the existing CPU archive."""
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
REPO = Path(os.environ['MLX_SOURCE_ROOT'])
BUILD = Path(os.environ['MLX_CPU_BUILD'])
ARCHIVE = BUILD / 'mlx-build/libmlx.a'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (180, 180))


def method(source, name):
    start = source.index('std::vector<array> ' + name + '::vjp(')
    return source[start:source.index('\n}', start) + 2]


def patch(source):
    old = method(source, 'GatherQMM')
    before = '''        dsb = unflatten(
            gather_mm_grad(
                x,
                cotan,
                lhs_indices,
                rhs_indices,
                sorted,
                std::move(shape),
                stream()),
            -1,
            {-1, group_size_},
            stream());'''
    after = '''        auto dw = gather_mm_grad(
            x,
            cotan,
            lhs_indices,
            rhs_indices,
            sorted,
            std::move(shape),
            stream());
        if (!transpose_) {
          dw = swapaxes(dw, -1, -2, stream());
        }
        dsb = unflatten(dw, -1, {-1, group_size_}, stream());'''
    assert old.count(before) == 1
    return source.replace(old, old.replace(before, after), 1)


records = []


def run(command, label):
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run(command, cwd=BUILD, capture_output=True, text=True, timeout=60)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (ROOT / (label + '.log')).write_text(result.stdout + result.stderr)
    records.append(dict(label=label, command=command, returncode=result.returncode,
                        wall_seconds=time.monotonic()-started,
                        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    (ROOT / 'build-results.json').write_text(json.dumps(records, indent=2) + '\n')
    print(label, result.returncode, round(records[-1]['wall_seconds'], 4), flush=True)
    if label.startswith('run-'):
        print('\n'.join(line for line in result.stdout.splitlines()
                        if line.startswith(('SUMMARY', 'EVIDENCE'))), flush=True)
    elif result.returncode:
        print(result.stderr[-6000:], flush=True)
        raise SystemExit(result.returncode)
    return result


def main():
    current = (ROOT / 'upstream-mlx_primitives.cpp').read_text()
    baseline = subprocess.run(['git', 'show', 'ce916dbbcaa88e433b6fd1e60a17f766d49c27fe:mlx/primitives.cpp'], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout
    for name, text in (('baseline-primitives.cpp', baseline),
                       ('patched-baseline-primitives.cpp', patch(baseline)),
                       ('patched-upstream-primitives.cpp', patch(current))):
        (ROOT / name).write_text(text)
    diff = ''.join(difflib.unified_diff(current.splitlines(True), patch(current).splitlines(True),
                                        fromfile='a/mlx/primitives.cpp', tofile='b/mlx/primitives.cpp'))
    (ROOT / 'gather-qmm-transpose-vjp.patch').write_text(diff)
    method_diff = ''.join(difflib.unified_diff(
        method(baseline, 'GatherQMM').splitlines(True),
        method(current, 'GatherQMM').splitlines(True)))
    (ROOT / 'GatherQMM-baseline-to-upstream.diff').write_text(method_diff)
    metadata = json.loads((ROOT / 'source-metadata.json').read_text())
    metadata.update(
        baseline_revision=subprocess.run(['git', 'rev-parse', 'ce916dbbcaa88e433b6fd1e60a17f766d49c27fe'], cwd=REPO,
            capture_output=True, text=True, check=True).stdout.strip(),
        archive=str(ARCHIVE),
        archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        build='Pristine/patched baseline primitives.cpp linked before reused CPU libmlx.a; not a clean current-main build.',
        upstream_difference='Current main adds global_scale support and index argument handling; see saved method diff.',
        intervention='Only restore the physical weight orientation before grouped scale/bias reduction.',
    )
    (ROOT / 'source-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    entry = next(c for c in json.loads((BUILD / 'compile_commands.json').read_text())
                 if c['file'].endswith('/mlx/primitives.cpp'))
    args = shlex.split(entry['command'])
    base = ['-O0' if a == '-O3' else a for a in args[:args.index('-o')]]
    test = ROOT / 'native_regression.o'
    run(base + ['-o', str(test), '-c', str(ROOT / 'native_regression.cpp')], 'compile-test')
    for label, name in (('before', 'baseline-primitives.cpp'),
                        ('after', 'patched-baseline-primitives.cpp')):
        obj, binary = ROOT / (label + '.o'), ROOT / ('native-' + label)
        run(base + ['-o', str(obj), '-c', str(ROOT / name)], 'compile-' + label)
        run(['/usr/bin/c++', str(test), str(obj), str(ARCHIVE), '-framework', 'Accelerate',
             '-o', str(binary)], 'link-' + label)
        result = run([str(binary)], 'run-' + label)
        assert result.returncode == (1 if label == 'before' else 0)


if __name__ == '__main__':
    main()
