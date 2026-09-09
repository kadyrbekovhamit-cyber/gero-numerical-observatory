"""Check saved evidence and patch application without rerunning MLX."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
checks=[]


def check(name, condition):
    checks.append(dict(check=name, passed=bool(condition)))


def method(text, name):
    start=text.index('std::vector<array> '+name+'(')
    return text[start:text.index('\n}', start)+2]


def main():
    for p in ROOT.glob('*.py'):
        ast.parse(p.read_text())
        check('python_syntax/'+p.name,True)
    for label,expected in (('before',(64,34,30,624,186)),('after',(64,64,0,624,0))):
        log=(ROOT/('run-'+label+'.log')).read_text()
        match=re.search(r'SUMMARY cases=(\d+) passed=(\d+) failed_cases=(\d+) checks=(\d+) failures=(\d+)',log)
        check('native_summary/'+label,match and tuple(map(int,match.groups()))==expected)
        check('native_failure_rows/'+label,len(re.findall(r'^FAIL ',log,re.M))==expected[-1])
    build=json.loads((ROOT/'build-results.json').read_text())
    check('build_exit_codes',all(x['returncode']==(1 if x['label']=='run-before' else 0) for x in build))
    check('build_sequential_steps',len(build)==7)
    for label in ('compile-test','compile-before','compile-after'):
        check('compiler_diagnostics/'+label,(ROOT/(label+'.log')).read_text().strip()=='')
    before=(ROOT/'baseline-primitives.cpp').read_text()
    current=(ROOT/'upstream-primitives.cpp').read_text()
    patched=(ROOT/'patched-upstream-primitives.cpp').read_text()
    for name in ('vjp','jvp'):
        check('same_Hadamard_'+name,method(before,'Hadamard::'+name)==method(current,'Hadamard::'+name))
    check('patch_preserves_jvp',method(current,'Hadamard::jvp')==method(patched,'Hadamard::jvp'))
    check('patch_changes_vjp',method(current,'Hadamard::vjp')!=method(patched,'Hadamard::vjp'))
    with tempfile.TemporaryDirectory(prefix='mlx-hadamard-check-',dir='/private/tmp') as tmp:
        target=Path(tmp)/'mlx/primitives.cpp';target.parent.mkdir();target.write_text(current)
        p=subprocess.run(['git','apply','--check',str(ROOT/'hadamard-adjoint.patch')],cwd=tmp,capture_output=True,text=True)
        check('patch_applies_to_pinned_main',p.returncode==0)
        if p.returncode:print(p.stderr)
        p=subprocess.run(['git','apply',str(ROOT/'hadamard-adjoint.patch')],cwd=tmp,capture_output=True,text=True)
        check('patch_matches_saved_result',p.returncode==0 and target.read_text()==patched)
    cert=json.loads((ROOT/'exact-matrix-certificate.json').read_text())
    source=(ROOT/'upstream-mlx_backend_common_hadamard.h').read_text()
    for row in cert:
        n=row['size']
        symbols=re.search(r'h'+str(n)+r' = R"\(\n(.*?)\n\)";',source,re.S)[1].splitlines()
        h=[[1 if x=='+' else -1 for x in r] for r in symbols]
        check('integer_orthogonality/'+str(n),all(sum(h[k][i]*h[k][j] for k in range(n))==(n if i==j else 0) for i in range(n) for j in range(n)))
        check('matrix_asymmetry/'+str(n),sum(h[i][j]!=h[j][i] for i in range(n) for j in range(n))==row['asymmetric_entries'])
        check('exact_energy_gradient/'+str(n),[sum(h[i][k]*h[k][0] for k in range(n)) for i in range(n)]==row['energy_gradient_at_e0_numerator'])
    probe=json.loads((ROOT/'probe-results.json').read_text())
    check('official_wheel_version',probe['version']=='0.32.2')
    check('official_wheel_CPU',probe['device']=='Device(cpu, 0)')
    for row in probe['cases']:
        if row['n'] in (20,28,40,56):
            check('wrong_sign_reproduced/'+str(row['n']),abs(row['energy_gradient'][0]+.5)<1e-6 and abs(row['first_coordinate_fd']-1)<1e-4)
            check('loss_increases/'+str(row['n']),row['energy_after_step']>row['energy'] and row['expected_energy_after_step']<row['energy'])
        else:
            check('symmetric_control/'+str(row['n']),row['adjoint_max_error']<1e-6)
    searches=json.loads((ROOT/'duplicate-search.json').read_text())
    for row in searches:
        check('complete_search/'+row['query'],row.get('incomplete_results') is False and len(row['items'])==row['total_count'])
    summary=dict(passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),checks=checks)
    (ROOT/'validation-results.json').write_text(json.dumps(summary,indent=2)+'\n')
    for folder in (ROOT,):
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'}
        (folder/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'}))
    for c in checks:
        if not c['passed']:print(c)
    raise SystemExit(bool(summary['failed']))


if __name__=='__main__':main()
