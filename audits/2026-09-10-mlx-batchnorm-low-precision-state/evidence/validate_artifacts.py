"""Validate audit artifacts, without importing MLX or accessing the network."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parent
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
def read_json(name):return json.loads((ROOT/name).read_text())
meta=read_json('source-metadata.json')
check('source_fetch_succeeded',not meta['errors'])
check('pinned_revision',meta['upstream_revision']=='81ba1c6a0e50a9268b931579c2d4f1158b9aab5a')
check('initial_dns_failure_preserved',bool(read_json('source-metadata-first.json')['errors']))
for entry in meta['files']:
    data=(ROOT/('upstream-'+entry['path'].replace('/','_'))).read_bytes()
    check('source_hash/'+entry['path'],hashlib.sha256(data).hexdigest()==entry['sha256'])
for path in sorted(ROOT.glob('*.py')):
    ast.parse(path.read_text(),filename=str(path))
    check('python_syntax/'+path.name,True)

source=(ROOT/'upstream-python_mlx_nn_layers_normalization.py').read_text()
patched=(ROOT/'patched-normalization.py').read_text()
before_ast,after_ast=ast.parse(source),ast.parse(patched)
def get_class(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BatchNorm')
before_class,after_class=get_class(before_ast),get_class(after_ast)
check('installed_class_matches_public',ast.dump(before_class)==ast.dump(get_class(ast.parse((ROOT/'installed-batchnorm-source.py').read_text()))))
modified=[]
before_methods={n.name:n for n in before_class.body if isinstance(n,ast.FunctionDef)}
after_methods={n.name:n for n in after_class.body if isinstance(n,ast.FunctionDef)}
for name,node in before_methods.items():
    if ast.dump(node)!=ast.dump(after_methods[name]):modified.append(name)
check('only_stats_and_call_changed',set(modified)=={'_calc_stats','__call__'})
for tree in (before_ast,after_ast):tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!='BatchNorm']
check('other_top_level_code_unchanged',ast.dump(before_ast)==ast.dump(after_ast))
patch_meta=read_json('patch-metadata.json')
check('tested_source_matches_patch_metadata',hashlib.sha256(patched.encode()).hexdigest()==patch_meta['patched_sha256'])
check('patch_added_lines',patch_meta['added_lines']==7)
with tempfile.TemporaryDirectory(prefix='patch-check-',dir=ROOT) as temp:
    target=Path(temp)/'python/mlx/nn/layers/normalization.py'
    target.parent.mkdir(parents=True)
    target.write_text(source)
    for flag in (['--check'],[]):
        p=subprocess.run(['git','apply',*flag,str(ROOT/'batchnorm-fp32-statistics.patch')],cwd=temp,capture_output=True,text=True,timeout=5)
        check('git_apply/'+('check' if flag else 'apply'),p.returncode==0)
        if p.returncode:print(p.stderr)
    check('applied_patch_equals_tested_source',target.read_text()==patched)

reg=read_json('regression-results.json')
compat=read_json('compatibility-results.json')
expected={'regression':{'before':{'checks':2760,'failed':102},'after':{'checks':2760,'failed':0}},
          'compatibility':{'before':{'checks':29,'failed':18},'after':{'checks':33,'failed':0}}}
for label,result in [('regression',reg),('compatibility',compat)]:
    check(label+'/summary',result['summary']==expected[label])
    check(label+'/real_cpu_wheel',result['version']=='0.32.2' and result['device']=='Device(cpu, 0)')
    check(label+'/cpu_under_limit',result['cpu_seconds']<30)
    for mode in ('before','after'):
        rows=[r for r in result['checks'] if r['mode']==mode]
        actual=dict(checks=len(rows),failed=sum(not r['passed'] for r in rows))
        check(label+'/recount/'+mode,actual==result['summary'][mode])
check('192_forward_scenarios',reg['forward_scenarios']==192)
check('12_gradient_scenarios',reg['gradient_scenarios']==12)
numeric={'output','running_mean','running_var','gamma','beta','input','state_mean','state_var'}
check('1020_numerical_assertions',sum(r['mode']=='after' and r['name'].split('/')[-1] in numeric for r in reg['checks'])==1020)
check('first_harness_mismatches_retained',read_json('regression-first-results.json')['summary']['after']['failed']==3)
boundary=read_json('variance-boundary-results.json')
check('boundary_after_passes',all(boundary['cases'][1]['checks'].values()))
check('boundary_before_four_failures',sum(not x for x in boundary['cases'][0]['checks'].values())==4)
check('representable_variance_failure_recorded',boundary['direct_statistics']['expected_biased_variance']==36864.
      and boundary['direct_statistics']['biased_variance']=='Infinity')
check('extreme_fp32_limitation_preserved',len(compat['remaining_limits'])==2 and
      all(r['running_var']==['Infinity'] for r in compat['remaining_limits']))

search=read_json('duplicate-search.json')
check('four_searches_complete',len(search)==4 and all('result' in q and not q['result']['incomplete_results']
      and q['result']['total_count']==len(q['result']['items']) for q in search))
check('26_unique_search_hits',len({i['number'] for q in search for i in q['result']['items']})==26)
comments=read_json('duplicate-comments.json')
check('31_targeted_comments',len(comments)==4 and all('error' not in q for q in comments)
      and sum(len(q['comments']) for q in comments)==31)
for _,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
    if not target.startswith('https://'):
        check('report_link/'+target,(ROOT/target).exists() or target in ['artifact-validation.json','SHA256SUMS.json'])

result=dict(total=len(checks),failed=sum(not c['passed'] for c in checks),checks=checks)
(ROOT/'artifact-validation.json').write_text(json.dumps(result,indent=2)+'\n')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir())
          if p.is_file() and p.name!='SHA256SUMS.json'}
(ROOT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(total=result['total'],failed=result['failed'])))
for row in checks:
    if not row['passed']:print(row)
raise SystemExit(bool(result['failed']))
