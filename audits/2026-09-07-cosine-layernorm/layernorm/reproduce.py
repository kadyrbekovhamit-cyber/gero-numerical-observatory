"""Run the real native layer in three fresh processes, checking expected status."""
import argparse
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
binary = args.binary.resolve()
names = ['OrdinaryNonuniformGamma', 'InputGradientMatchesActualForwardFiniteDifference',
         'ZeroGammaMakesInputGradientZero', 'ConstantGammaScalesInputGradient']
test_filter = ':'.join('LayerNormNumericalAudit.' + name for name in names)
records = []
for run in range(1, 4):
    xml_file = (args.output / f'{args.expect}-{run}.xml').resolve()
    result = subprocess.run([str(binary), '--gtest_filter=' + test_filter,
                             '--gtest_color=no', '--gtest_output=xml:' + str(xml_file)],
                            cwd=binary.parent, text=True, capture_output=True)
    log = result.stdout + result.stderr
    (args.output / f'{args.expect}-{run}.log').write_text(log)
    root = ET.parse(xml_file).getroot()
    failures = int(root.attrib['failures'])
    expected_failures = 4 if args.expect == 'original' else 0
    assert failures == expected_failures, (args.expect, run, failures, log)
    assert int(root.attrib['tests']) == 4
    assert result.returncode == (1 if failures else 0)
    dx = next(line for line in log.splitlines() if line.startswith('AUDIT_DX '))
    finite_difference = [float(line.split()[-1]) for line in log.splitlines()
                         if line.startswith('AUDIT_FINITE_DIFFERENCE ')]
    records.append(dict(run=run, tests=4, failures=failures, returncode=result.returncode,
                        input_gradient=[float(x) for x in dx.split()[1:]],
                        forward_finite_difference=finite_difference))
assert all(record['input_gradient'] == records[0]['input_gradient'] for record in records)
(args.output / f'{args.expect}-reproduction.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2))
