"""Execute actual PyLoan and an explicit synthetic document/decision adapter. No network or payments."""
from pathlib import Path
from decimal import Decimal
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'replay-output'
OUT.mkdir(exist_ok=True)
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
source = ROOT/'source/pyloan'
manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name, sha in manifest['source_sha256'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == sha
wheel = ROOT/'artifacts/pyloan-0.7.2-py3-none-any.whl'
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == manifest['wheel_sha256']
with zipfile.ZipFile(wheel) as archive:
    assert all(not n.startswith('/') and '..' not in Path(n).parts for n in archive.namelist())
    archive.extractall(OUT/'release')
for state in ['candidate','restored']:
    shutil.copytree(source, OUT/state/'pyloan', dirs_exist_ok=True)
file = OUT/'candidate/pyloan/pyloan.py'
old = 'if is_last_payment and self.loan_type != LoanType.INTEREST_ONLY:'
new = 'if (is_last_payment and self.loan_type != LoanType.INTEREST_ONLY\n                        and not (self.loan_type == LoanType.ANNUITY\n                                 and self.payment_amount is not None)):'
text = file.read_text(); assert text.count(old) == 1; file.write_text(text.replace(old, new))
paths = {'release':OUT/'release', 'current':ROOT/'source', 'candidate':OUT/'candidate', 'restored':OUT/'restored'}
results = {}
for state in ['reference', *paths]:
    command = [sys.executable,'-B',str(ROOT/'capture_notice.py'),'--output',str(OUT/state/'documents')]
    command += ['--reference'] if state == 'reference' else ['--module-root',str(paths[state])]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    results[state] = json.loads(process.stdout)
assert results['release'] == results['current'] == results['restored']
assert results['candidate'] == results['reference']
assert results['reference']['notice']['amount_requested'] == '100.00'
assert results['current']['notice']['amount_requested'] == '919.10'
assert results['reference']['notice']['remaining_principal_in_schedule'] == '819.10'
assert results['current']['notice']['remaining_principal_in_schedule'] == '0.00'
assert results['reference']['decision']['result'] == 'WITHIN_BUDGET'
assert results['current']['decision']['result'] == 'EXCEEDS_BUDGET'
assert results['current']['decision']['cash_above_budget'] == '769.10'
summary = {'case_id':'pyloan-explicit-payment-final-balloon','evidence_type':'synthetic downstream demonstration',
           'new_independent_defect_count':0,'inputs':{'principal':'1000.00','annual_interest_percent':'12',
             'term_months':2,'configured_payment':'100.00','available_budget':'150.00','day_count':'30E/360 ISDA'},
           'chain':['PyLoan final-payment override','serialized synthetic payment notice','budget comparison on notice amount'],
           'extra_amount_requested':'819.10','requested_amount_multiple':'9.191','cash_above_example_budget':'769.10',
           'decision_flip':'WITHIN_BUDGET → EXCEEDS_BUDGET','candidate_restores_document_and_decision':True,
           'restoring_original_restores_difference':True,'results':results,
           'interpretation':'Additional requested cash is accelerated principal in this schedule, not a measured fee, extra interest or borrower loss.',
           'limitations':['GERO authored the notice and budget adapter; these are not built-in PyLoan features.',
                          'No bank integration, real borrower, executed collection, credit eligibility decision or production prevalence was tested.',
                          'This one downstream scenario supplements the original 864-case grid; the original rounding and upstream-test limitations remain.']}
(OUT/'CHAIN_RESULT.json').write_text(json.dumps(summary, indent=2)+'\n')
files = {p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(OUT.glob('*/documents/*'))}
files['CHAIN_RESULT.json'] = hashlib.sha256((OUT/'CHAIN_RESULT.json').read_bytes()).hexdigest()
(OUT/'REPLAY_RECEIPT.json').write_text(json.dumps({'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python':sys.version,'status':'passed','states_executed':list(results),'files_sha256':files,
    'audio_playback':False,'external_payment_or_message':False},indent=2)+'\n')
print(json.dumps({'status':'passed','extra_requested':'819.10','budget_difference':'WITHIN_BUDGET → EXCEEDS_BUDGET',
                  'candidate_restores_both':True,'restored_original_reproduces_both':True},indent=2))
