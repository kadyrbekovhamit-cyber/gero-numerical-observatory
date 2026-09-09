"""Run the actual C++ regression suite serially; no source mutations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--expect', choices=('original', 'patched'), required=True)
args = parser.parse_args()
binary = args.binary.resolve()
dest = args.output.resolve()
dest.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    env[name] = '1'
command = [str(binary),
           '--gtest_filter=PoolingPaddingAudit.*:PaddingCases/*:Pooling2DMax/*',
           '--gtest_color=no', '--gtest_output=xml:' + str(dest / 'results.xml')]
with (dest / 'runtime.log').open('w') as log:
    run = subprocess.run(command, cwd=binary.parents[3], env=env,
                         stdout=log, stderr=subprocess.STDOUT, timeout=45)
tree = ET.parse(dest / 'results.xml').getroot()
failed = [t.get('classname') + '.' + t.get('name')
          for t in tree.iter('testcase') if t.find('failure') is not None]
summary = {'expect': args.expect, 'returncode': run.returncode,
           'tests': int(tree.get('tests')), 'failures': int(tree.get('failures')),
           'errors': int(tree.get('errors', '0')), 'failed_tests': failed,
           'runtime_seconds': float(tree.get('time')), 'command': command}
(dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
expected = (58, 8, 1) if args.expect == 'original' else (58, 0, 0)
observed = (summary['tests'], summary['failures'], summary['returncode'])
assert summary['errors'] == 0 and observed == expected, (observed, expected)
print(json.dumps(summary, indent=2))
