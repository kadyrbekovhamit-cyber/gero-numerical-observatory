"""Public GitHub GET requests only."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode
ROOT=Path(__file__).resolve().parent
def get(url):
    p=subprocess.run(["curl","--fail","--silent","--show-error","--max-time","12",url],
                     capture_output=True,timeout=15)
    if p.returncode:raise RuntimeError(p.stderr.decode().strip())
    return p.stdout
meta=dict(checked_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),files=[],errors=[])
try:
    sha=json.loads(get("https://api.github.com/repos/ml-explore/mlx/commits/main"))["sha"]
    meta["upstream_revision"]=sha
    print("main",sha,flush=True)
    for path in ("mlx/primitives.cpp","mlx/ops.cpp","python/tests/test_autograd.py"):
        try:
            url=f"https://raw.githubusercontent.com/ml-explore/mlx/{sha}/{path}"
            data=get(url);(ROOT/("upstream-"+path.replace("/","_"))).write_bytes(data)
            meta["files"].append(dict(path=path,url=url,sha256=hashlib.sha256(data).hexdigest()))
        except Exception as e:meta["errors"].append(dict(path=path,error=str(e)))
except Exception as e:meta["errors"].append(dict(path="main",error=str(e)))
(ROOT/"source-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
rows=[]
for q in ('repo:ml-explore/mlx "divide" "gradient"','repo:ml-explore/mlx "division" "gradient"',
          'repo:ml-explore/mlx "Divide::vjp"','repo:ml-explore/mlx "division" "overflow"'):
    url="https://api.github.com/search/issues?"+urlencode(dict(q=q,per_page=100))
    try:
        result=json.loads(get(url));rows.append(dict(query=q,url=url,result=result))
        print(q,result["total_count"],[(x["number"],x["title"]) for x in result["items"]],flush=True)
    except Exception as e:
        rows.append(dict(query=q,url=url,error=str(e)));print(q,str(e),flush=True)
    (ROOT/"duplicate-search.json").write_text(json.dumps(rows,indent=2)+"\n")
