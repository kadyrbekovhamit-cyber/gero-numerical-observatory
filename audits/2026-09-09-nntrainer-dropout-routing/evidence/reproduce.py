#!/usr/bin/env python3
"""Run the real layer regression; require the exact expected failure set."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expect', required=True, choices=['original', 'patched'])
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    xml_path = output / 'results.xml'
    # Never accept an XML file left by a previous invocation.
    xml_path.unlink(missing_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    command = [str(binary), '--gtest_filter=DropoutRoutingAudit.*',
               '--gtest_output=xml:' + str(xml_path)]
    result = subprocess.run(command, cwd=binary.parents[3], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=45)
    (output / 'runtime.log').write_text(result.stdout)
    root = ET.parse(xml_path).getroot()
    failed = {case.attrib['name'] for suite in root for case in suite
              if case.find('failure') is not None}
    expected = ({'ZeroRateRoutesEachIncomingGradientToItsInput',
                 'ZeroRateMatchesRuntimeFiniteDifferences',
                 'ReplayUsesEachInputsOwnGradientAndMask'}
                if args.expect == 'original' else set())
    tests = int(root.attrib['tests'])
    ok = (tests == 4 and failed == expected
          and int(root.attrib['failures']) == len(expected)
          and int(root.attrib['errors']) == 0
          and int(root.attrib['disabled']) == 0
          and result.returncode == (1 if expected else 0))
    summary = {'expect': args.expect, 'tests': tests,
               'failed_tests': sorted(failed), 'returncode': result.returncode,
               'matches_expected': ok, 'command': command}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
