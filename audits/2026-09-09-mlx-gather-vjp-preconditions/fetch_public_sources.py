"""Read public source and issue metadata; do not post or use account data."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
def api(path):
    r=subprocess.run(['gh','api',path],capture_output=True,text=True,timeout=20)
    if r.returncode:raise RuntimeError(r.stderr)
    return json.loads(r.stdout)

def main():
    metadata=dict(checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),errors=[],files=[])
    sha=api('repos/ml-explore/mlx/commits/main')['sha']
    metadata['upstream_revision']=sha
    print('main',sha,flush=True)
    for path in ('mlx/primitives.cpp','mlx/ops.cpp','python/tests/test_blas.py','python/tests/test_quantized.py'):
        try:
            url=f'https://raw.githubusercontent.com/ml-explore/mlx/{sha}/{path}'
            p=subprocess.run(['curl','--fail','--silent','--show-error','--max-time','20',url],capture_output=True,timeout=25)
            if p.returncode:raise RuntimeError(p.stderr.decode())
            (ROOT/('upstream-'+path.replace('/','_'))).write_bytes(p.stdout)
            metadata['files'].append(dict(path=path,url=url,sha256=hashlib.sha256(p.stdout).hexdigest()))
            print('saved',path,len(p.stdout),flush=True)
        except Exception as e:
            metadata['errors'].append(dict(path=path,error=str(e)))
        (ROOT/'source-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from urllib.parse import urlencode
    searches=[]
    for query in ('repo:ml-explore/mlx gather_mm gradient','repo:ml-explore/mlx gather_mm vjp',
                  'repo:ml-explore/mlx "left_sorted"','repo:ml-explore/mlx "lhs_indices" "sorted"',
                  'repo:ml-explore/mlx gather_qmm gradient'):
        try:
            result=api('search/issues?'+urlencode(dict(q=query,per_page=100)))
            searches.append(dict(query=query,result=result))
            print(query,result['total_count'],[(x['number'],x['title']) for x in result['items']],flush=True)
        except Exception as e: searches.append(dict(query=query,error=str(e)))
        (ROOT/'duplicate-search.json').write_text(json.dumps(searches,indent=2)+'\n')

if __name__=='__main__':main()
