"""Run focused native attention cases in three fresh processes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--expect', choices=['original', 'patched'], required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
names = ['RectangularFewerQueries', 'FutureValueHasZeroDerivative',
         'LargeFiniteFutureLogitIsExcluded', 'IncrementalChunksMatchFull']
filter_value = ':'.join('AttentionNumericalAudit.' + name for name in names)
records = []
for index in range(1, 4):
    xml = (args.output / f'{args.expect}-{index}.xml').resolve()
    result = subprocess.run([
        str(args.binary.resolve()), f'--gtest_filter={filter_value}',
        '--gtest_color=no', f'--gtest_output=xml:{xml}'], capture_output=True, text=True)
    log = result.stdout + result.stderr
    (args.output / f'{args.expect}-{index}.log').write_text(log)
    tree = ET.parse(xml).getroot()
    failures = int(tree.attrib['failures'])
    errors = int(tree.attrib.get('errors', 0))
    tests = int(tree.attrib['tests'])
    expected = 4 if args.expect == 'original' else 0
    assert tests == 4 and failures == expected and errors == 0
    assert result.returncode == (1 if expected else 0)
    values = [line for line in log.splitlines() if line.startswith('ATTENTION_OUTPUT')]
    records.append(dict(process=index, tests=tests, failures=failures, values=values))
assert records[0]['values'] == records[1]['values'] == records[2]['values']
summary = dict(checked_at=datetime.now(timezone.utc).isoformat(),
               expectation=args.expect, records=records, repeat_outputs_identical=True)
(args.output / f'{args.expect}-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
