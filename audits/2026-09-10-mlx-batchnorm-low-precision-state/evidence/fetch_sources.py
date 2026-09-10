"""Read public GitHub sources and issue search results only."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode
ROOT = Path(__file__).resolve().parent
def get(url):
    p = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--max-time", "12", url],
                       capture_output=True, timeout=15)
    if p.returncode: raise RuntimeError(p.stderr.decode())
    return p.stdout
meta = dict(checked_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), files=[], errors=[])
try:
    sha = json.loads(get("https://api.github.com/repos/ml-explore/mlx/commits/main"))["sha"]
    meta["upstream_revision"] = sha
    for path in ("python/mlx/nn/layers/normalization.py", "python/tests/test_nn.py"):
        url = f"https://raw.githubusercontent.com/ml-explore/mlx/{sha}/{path}"
        data = get(url)
        (ROOT / ("upstream-"+path.replace("/","_"))).write_bytes(data)
        meta["files"].append(dict(path=path, url=url, sha256=hashlib.sha256(data).hexdigest()))
except Exception as e:
    meta["errors"].append(str(e))
(ROOT / "source-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta),flush=True)
rows = []
for query in ('repo:ml-explore/mlx "BatchNorm"', 'repo:ml-explore/mlx "running_var"',
              'repo:ml-explore/mlx "normalization" "float16"',
              'repo:ml-explore/mlx "BatchNorm" "overflow"'):
    row = dict(query=query, attempts=[])
    for sort in (None,"updated"):
        params = dict(q=query, per_page=100)
        if sort: params["sort"] = sort
        url = "https://api.github.com/search/issues?"+urlencode(params)
        try:
            result = json.loads(get(url))
            row["attempts"].append(dict(url=url,result=result))
            row["result"] = result
            if not result.get("incomplete_results"): break
        except Exception as e:
            row["attempts"].append(dict(url=url,error=str(e)))
    rows.append(row)
    (ROOT / "duplicate-search.json").write_text(json.dumps(rows,indent=2)+"\n")
    result = row.get("result",{})
    print(query,result.get("total_count"),result.get("incomplete_results"),
          [(i["number"],i["title"]) for i in result.get("items",[])],flush=True)
