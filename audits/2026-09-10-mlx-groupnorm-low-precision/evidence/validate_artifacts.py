"""Validate saved sources, patch, counts and limitations without MLX or network."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parent
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
def data(name):return json.loads((ROOT/name).read_text())
meta=data('source-metadata.json')
check('pinned_revision',meta['upstream_revision']=='81ba1c6a0e50a9268b931579c2d4f1158b9aab5a')
check('source_fetch_succeeded',not meta['errors'])
for entry in meta['files']:
    raw=(ROOT/('upstream-'+entry['path'].replace('/','_'))).read_bytes()
    check('source_hash/'+entry['path'],hashlib.sha256(raw).hexdigest()==entry['sha256'])
for path in sorted(ROOT.glob('*.py')):
    ast.parse(path.read_text(),filename=str(path))
    check('python_syntax/'+path.name,True)

source=(ROOT/'upstream-python_mlx_nn_layers_normalization.py').read_text()
patched=(ROOT/'patched-normalization.py').read_text()
old,new=ast.parse(source),ast.parse(patched)
def group(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GroupNorm')
old_group,new_group=group(old),group(new)
check('installed_class_matches_public',ast.dump(old_group)==ast.dump(group(ast.parse((ROOT/'installed-groupnorm-source.py').read_text()))))
old_methods={n.name:n for n in old_group.body if isinstance(n,ast.FunctionDef)}
new_methods={n.name:n for n in new_group.body if isinstance(n,ast.FunctionDef)}
changed={name for name,node in old_methods.items() if ast.dump(node)!=ast.dump(new_methods[name])}
check('only_default_group_norm_changed',changed=={'_group_norm'})
for tree in (old,new):tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!='GroupNorm']
check('other_top_level_code_unchanged',ast.dump(old)==ast.dump(new))
check('empty_input_guard_present','if x.size and x.dtype in (mx.float16, mx.bfloat16):' in patched)
check('metadata_matches_tested_source',hashlib.sha256(patched.encode()).hexdigest()==data('patch-metadata.json')['patched_sha256'])
with tempfile.TemporaryDirectory(prefix='patch-check-',dir=ROOT) as temp:
    target=Path(temp)/'python/mlx/nn/layers/normalization.py'
    target.parent.mkdir(parents=True)
    target.write_text(source)
    for flag in (['--check'],[]):
        p=subprocess.run(['git','apply',*flag,str(ROOT/'groupnorm-low-precision.patch')],cwd=temp,capture_output=True,text=True,timeout=5)
        check('git_apply/'+('check' if flag else 'apply'),p.returncode==0)
        if p.returncode:print(p.stderr)
    check('applied_patch_matches_tested_source',target.read_text()==patched)

reg=data('regression-results.json')
compat=data('compatibility-results.json')
expected={'regression':{'before':{'checks':1424,'failed':45},'after':{'checks':1424,'failed':0}},
          'compatibility':{'before':{'checks':25,'failed':0},'after':{'checks':41,'failed':0}}}
for label,result in [('regression',reg),('compatibility',compat)]:
    check(label+'/recorded_summary',result['summary']==expected[label])
    check(label+'/real_cpu_wheel',result['version']=='0.32.2' and result['device']=='Device(cpu, 0)')
    check(label+'/successful_cpu_bounded',result['cpu_seconds']<(5 if label=='compatibility' else 30))
    for mode in ['before','after']:
        rows=[r for r in result['checks'] if r['mode']==mode]
        found=dict(checks=len(rows),failed=sum(not r['passed'] for r in rows))
        check(label+'/recount/'+mode,found==result['summary'][mode])
check('332_forward_scenarios',reg['forward_scenarios']==332)
check('24_gradient_scenarios',reg['gradient_scenarios']==24)
numeric={'output','gamma','beta','input_vjp','input_jvp'}
check('428_numerical_assertions',sum(r['mode']=='after' and r['name'].split('/')[-1] in numeric for r in reg['checks'])==428)
check('first_tolerance_flag_retained',data('regression-first-results.json')['summary']['after']['failed']==1)
check('v1_nonempty_pass_retained',data('regression-v1-results.json')['summary']['after']['failed']==0)
check('v1_source_differs_from_final',(ROOT/'patched-normalization-v1.py').read_text()!=patched)
check('empty_original_completed','done [[], []]' in (ROOT/'empty-before.log').read_text())
check('empty_v1_did_not_complete','eval (2, 0, 6)' in (ROOT/'empty-after-v1.log').read_text()
      and 'done' not in (ROOT/'empty-after-v1.log').read_text())
check('empty_final_completed','done [[], []]' in (ROOT/'empty-after-final.log').read_text())
check('bounded_failures_recorded',[r['exit_code'] for r in data('bounded-failures.json')['runs']]==[152,152,0,152,0])
limits=compat['remaining_limits']
check('five_limit_observations_preserved',len(limits)==5 and all(r['output']==[0.,0.,0.,0.] for r in limits))
check('float64_limit_has_nonzero_reference',max(abs(v) for v in limits[-1]['expected'])>1.)
paired=data('paired-reproduction.json')
check('paired_original_zero',paired['cases'][0]['output']==[0.,0.,0.,0.])
check('paired_after_correct',paired['cases'][1]['output']==[-1.,1.,-1.,1.])

queries=data('duplicate-search.json')
check('four_searches_complete',len(queries)==4 and all('result' in q and not q['result']['incomplete_results']
      and q['result']['total_count']==len(q['result']['items']) for q in queries))
check('six_unique_search_hits',len({i['number'] for q in queries for i in q['result']['items']})==6)
comments=data('duplicate-comments.json')
check('twelve_comments_retrieved',sum(len(r.get('comments',[])) for r in comments)==12)
check('one_comment_request_unavailable',[r['number'] for r in comments if 'error' in r]==[3613])
check('unavailable_comment_retry_documented',data('comment-retry-result.json')['success'] is False)
report=(ROOT/'README.md').read_text()
check('search_limit_stated','не удалось получить' in report)
check('failed_prototype_stated','лимитом 30 CPU-секунд' in report)
for _,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',report):
    if not target.startswith('https://'):
        check('report_link/'+target,(ROOT/target).exists() or target in ['artifact-validation.json','SHA256SUMS.json'])
result=dict(total=len(checks),failed=sum(not r['passed'] for r in checks),checks=checks)
(ROOT/'artifact-validation.json').write_text(json.dumps(result,indent=2)+'\n')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'}
(ROOT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(total=result['total'],failed=result['failed'])))
for row in checks:
    if not row['passed']:print(row)
raise SystemExit(bool(result['failed']))
