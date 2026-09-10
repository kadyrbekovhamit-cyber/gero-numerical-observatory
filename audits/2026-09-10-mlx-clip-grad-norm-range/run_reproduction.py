"""Reproduce in a new directory without changing the frozen evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--python', default=sys.executable)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source = root / 'evidence'
for name, expected in json.loads((source / 'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
if args.output.exists():
    parser.error('Output must be a new directory.')
shutil.copytree(source, args.output)
environment = os.environ.copy()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    environment[key] = '1'
for script in ('regression.py', 'compatibility.py', 'validate_artifacts.py'):
    subprocess.run([args.python, script], cwd=args.output, env=environment,
                   check=True, timeout=60)
comparison = {}
for name in ('regression-results.json', 'compatibility-results.json'):
    original = json.loads((source / name).read_text())
    fresh = json.loads((args.output / name).read_text())
    assert fresh['checks'] == original['checks'], name
    assert fresh['summary'] == original['summary'], name
    comparison[name] = {'all_rows_equal': True, 'summary': fresh['summary'],
                        'cpu_seconds': fresh['cpu_seconds']}
(args.output / 'fresh-rerun.json').write_text(json.dumps(comparison, indent=2) + '\n')
print('All fresh numerical rows match the archived evidence.')
