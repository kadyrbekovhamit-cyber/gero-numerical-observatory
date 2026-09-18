from pathlib import Path
import shutil,subprocess,sys,os,json,difflib
B=Path(__file__).parent
copy=B/'candidate-source'
if not copy.exists(): shutil.copytree(B/'source',copy)
shutil.copy2(B/'candidate/pyloan/pyloan.py',copy/'src/pyloan/pyloan.py')
receipt={}
for mode,path in [('current',B/'source'),('candidate',copy),('restored',B/'source')]:
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(path/'src')+os.pathsep+str(path)
    for kind,args in [('upstream',['-m','unittest','discover','-s','tests','-v']),('regression',[str(B/'test_explicit_payment.py'),'-v'])]:
        p=subprocess.run([sys.executable,'-B',*args],cwd=path,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        out=B/'evidence'/f'{kind}-{mode}.log';out.write_text(p.stdout)
        receipt[kind+'-'+mode]={'exit_code':p.returncode,'log':str(out),'tail':p.stdout.splitlines()[-5:]}
(B/'candidate.patch').write_text(''.join(difflib.unified_diff((B/'source/src/pyloan/pyloan.py').read_text().splitlines(True),(B/'candidate/pyloan/pyloan.py').read_text().splitlines(True),fromfile='a/src/pyloan/pyloan.py',tofile='b/src/pyloan/pyloan.py')))
(B/'evidence/TEST_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
