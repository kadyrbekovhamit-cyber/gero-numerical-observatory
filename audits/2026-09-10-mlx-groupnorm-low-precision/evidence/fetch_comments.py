"""Read comments on the six public search hits; only three have comments."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
rows=[]
for number in [3613,3702,2490]:
    url=f'https://api.github.com/repos/ml-explore/mlx/issues/{number}/comments?per_page=100'
    p=subprocess.run(['curl','--fail','--silent','--show-error','--max-time','12',url],capture_output=True,timeout=15)
    row=dict(number=number,url=url)
    if p.returncode:row['error']=p.stderr.decode()
    else:row['comments']=json.loads(p.stdout)
    rows.append(row)
    (ROOT/'duplicate-comments.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(number,len(row.get('comments',[])),row.get('error'),flush=True)
