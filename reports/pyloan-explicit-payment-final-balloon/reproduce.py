#!/usr/bin/env python3
"""Sequential portable replay. Recorded expected/ evidence is never overwritten."""
from pathlib import Path
import os, sys, json, shutil, zipfile, hashlib, subprocess, re
B=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
(B/'evidence').mkdir(exist_ok=True)
manifest=json.loads((B/'expected/source-manifest.json').read_text())
for rel,entry in manifest['files'].items():
    assert hashlib.sha256((B/'source'/rel).read_bytes()).hexdigest()==entry['sha256'],rel
for version in ['0.7.0','0.7.2']:
    meta=json.loads((B/'expected'/f'pypi-{version}.json').read_text())
    artifact=next(a for a in meta['urls'] if a['filename'].endswith('.whl'))
    p=B/'artifacts'/artifact['filename']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==artifact['digests']['sha256']
    with zipfile.ZipFile(p) as z:
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(B/f'release-{version}')
for mode in ['candidate','restored']:
    shutil.copytree(B/'source/src/pyloan',B/mode/'pyloan',dirs_exist_ok=True)
p=B/'candidate/pyloan/pyloan.py'
s=p.read_text()
old='if is_last_payment and self.loan_type != LoanType.INTEREST_ONLY:'
new='if (is_last_payment and self.loan_type != LoanType.INTEREST_ONLY\n                        and not (self.loan_type == LoanType.ANNUITY\n                                 and self.payment_amount is not None)):'
assert s.count(old)==1
p.write_text(s.replace(old,new))
subprocess.run([sys.executable,'-B',str(B/'run_validation.py')],check=True,cwd=B)
counts=json.loads((B/'evidence/GRID_RECEIPT.json').read_text())
for mode,want in [('current',408),('release',408),('prior_release',0),('candidate',0),('restored',408)]:
    assert counts[mode]['cases']==864
    assert counts[mode]['payment_cap_failures']==want
    name=f'grid-{mode}.json'
    assert (B/'evidence'/name).read_bytes()==(B/'expected'/name).read_bytes(),name
assert counts['candidate_controls_identical']
for mode in ['current','candidate','restored']:
    name=f'controls-{mode}.json'
    assert (B/'evidence'/name).read_bytes()==(B/'expected'/name).read_bytes()
assert counts['candidate']['exact_oracle_mismatches']==24
tr=json.loads((B/'evidence/TEST_RECEIPT.json').read_text())
assert [tr[f'regression-{m}']['exit_code'] for m in ['current','candidate','restored']]==[1,0,1]
logs=[]
for mode in ['current','candidate','restored']:
    s=(B/'evidence'/f'upstream-{mode}.log').read_text()
    assert 'Ran 16 tests' in s and 'FAILED (failures=3)' in s
    s=re.sub(r'Ran 16 tests in [0-9.]+s','Ran 16 tests in <elapsed>s',s)
    logs.append(s.replace(str(B/'candidate-source'),str(B/'source')))
assert len(set(logs))==1
receipt={'status':'portable_replay_passed','grid_files_byte_identical_to_recorded':5,'scenario_count':864,'payment_cap_failures_current_release_candidate_restored':[408,408,0,408],'prior_release_failures':0,'candidate_exact_oracle_residuals':24,'unchanged_control_schedules':60,'focused_tests_candidate_pass':5,'upstream_failure_reports_in_all_three_states':3,'full_upstream_suite_green':False,'python':sys.version}
(B/'evidence/PORTABLE_REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
