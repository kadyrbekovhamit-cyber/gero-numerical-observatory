"""Read public GitHub metadata, code, and issues; no external writes."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'mlx-batchnorm-low-precision-state-2026-09-10'
def get(url):
    p=subprocess.run(['curl','--fail','--silent','--show-error','--max-time','12',url],capture_output=True,timeout=15)
    if p.returncode:raise RuntimeError(p.stderr.decode())
    return p.stdout
meta=dict(checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=[],errors=[])
try:
    sha=json.loads(get('https://api.github.com/repos/ml-explore/mlx/commits/main'))['sha']
    meta['upstream_revision']=sha
    old=json.loads((PREVIOUS/'source-metadata.json').read_text())
    for path in ['python/mlx/nn/layers/normalization.py','python/tests/test_nn.py','mlx/fast.cpp']:
        url=f'https://raw.githubusercontent.com/ml-explore/mlx/{sha}/{path}'
        saved=next((x for x in old['files'] if x['path']==path),None)
        name='upstream-'+path.replace('/','_')
        entry=dict(path=path,url=url)
        if saved and old['upstream_revision']==sha:
            data=(PREVIOUS/name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==saved['sha256']
            entry['reused_from']=str(PREVIOUS/name)
        else:data=get(url)
        (ROOT/name).write_bytes(data)
        entry['sha256']=hashlib.sha256(data).hexdigest()
        meta['files'].append(entry)
except Exception as e:meta['errors'].append(str(e))
(ROOT/'source-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta),flush=True)
rows=[]
for query in ['repo:ml-explore/mlx "GroupNorm"','repo:ml-explore/mlx "group_norm" "overflow"',
              'repo:ml-explore/mlx "GroupNorm" "NaN"','repo:ml-explore/mlx "GroupNorm" "float16"']:
    row=dict(query=query,attempts=[])
    for sort in [None,'updated']:
        args=dict(q=query,per_page=100)
        if sort:args['sort']=sort
        url='https://api.github.com/search/issues?'+urlencode(args)
        try:
            result=json.loads(get(url))
            row['attempts'].append(dict(url=url,result=result))
            row['result']=result
            if not result.get('incomplete_results'):break
        except Exception as e:row['attempts'].append(dict(url=url,error=str(e)))
    rows.append(row)
    (ROOT/'duplicate-search.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=row.get('result',{})
    print(query,result.get('total_count'),result.get('incomplete_results'),
          [(x['number'],x['title']) for x in result.get('items',[])],flush=True)
