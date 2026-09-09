"""Validate saved evidence and patch applicability without rerunning computation."""
import hashlib
import os
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ['MLX_SOURCE_ROOT'])


def main():
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))

    wheel = json.loads((ROOT / 'wheel-reproduction.json').read_text())
    check('wheel_version', wheel['version'] == '0.32.2')
    check('CPU_device', wheel['device'] == 'Device(cpu, 0)')
    check('silent_scale_error', wheel['square']['ds'] == [1.] * 32)
    check('silent_bias_error', wheel['square']['db'] == [1.] * 32)
    check('expected_gradient', wheel['square']['expected_ds_db'] == [32.] + [0.] * 31)
    for arg in ('scales', 'biases'):
        check('forward_finite_difference_' + arg,
              wheel['square']['finite_differences'][arg] == [32., 0.])
    check('loss_initial', wheel['loss']['initial'] == 16.)
    check('wrong_step_ascends', wheel['loss']['actual_next'] == 16.953125)
    check('correct_step_descends', wheel['loss']['oracle_next'] == 1.9375)
    check('rectangular_wrong_bias_shape',
          wheel['rectangular']['actual_bias_gradient_shape'] == [1,64,1] and
          wheel['rectangular']['expected_bias_gradient_shape'] == [1,32,2])
    check('rectangular_scale_exception', 'broadcast' in wheel['rectangular']['scale_error'])

    before = (ROOT / 'run-before.log').read_text()
    after = (ROOT / 'run-after.log').read_text()
    check('before_summary', 'scenarios=205 failed_scenarios=57 checks=315 failures=70' in before)
    check('after_summary', 'scenarios=205 failed_scenarios=0 checks=379 failures=0' in after)
    check('after_no_failures', not any(line.startswith('FAIL ') for line in after.splitlines()))
    failures = [line for line in before.splitlines() if line.startswith('FAIL ')]
    check('before_failure_count', len(failures) == 70)
    check('transpose_true_controls', not any('transpose=1' in line for line in failures))
    check('forward_controls', not any('/y ' in line for line in failures))
    check('x_gradient_controls', not any('/dx ' in line or '/only_0 ' in line for line in failures))
    check('no_indices_controls', not any('no_indices_fallback' in line for line in failures))
    for entry in json.loads((ROOT / 'build-results.json').read_text()):
        check('command_' + entry['label'],
              entry['returncode'] == (1 if entry['label'] == 'run-before' else 0))
    metadata = json.loads((ROOT / 'source-metadata.json').read_text())
    check('source_downloads_succeeded', not metadata['errors'])
    for entry in metadata['files']:
        saved = ROOT / ('upstream-' + entry['path'].replace('/', '_'))
        check('source_hash_' + entry['path'],
              hashlib.sha256(saved.read_bytes()).hexdigest() == entry['sha256'])
    check('archive_hash', hashlib.sha256((Path(os.environ['MLX_CPU_BUILD'])/'mlx-build/libmlx.a').read_bytes()).hexdigest()
          == metadata['archive_sha256'])
    patch = ROOT / 'gather-qmm-transpose-vjp.patch'
    applicable = subprocess.run(['git','apply','--check',str(patch)], cwd=REPO,
                                capture_output=True, text=True)
    check('patch_applies_without_mutation', applicable.returncode == 0)
    check('one_source_patch', patch.read_text().count('--- a/') == 1)
    check('orientation_guard_in_patch', '+        if (!transpose_) {' in patch.read_text())
    check('no_unrelated_sorted_change', 'left_sorted_' not in patch.read_text())
    for target in re.findall(r'\]\(([^)]+)\)', (ROOT / 'README.md').read_text()):
        if '://' not in target:
            check('local_link_' + target, (ROOT / target).exists())
    report = dict(passed=sum(x['passed'] for x in checks),
                  failed=sum(not x['passed'] for x in checks), checks=checks,
                  scope='Saved evidence, hashes, controls and dry-run applicability; no rerun or theorem certification.')
    (ROOT / 'artifact-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2))
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
