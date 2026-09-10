"""Retry public source reads after the initial DNS timeout, without repeating search."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT = Path(__file__).resolve().parent
initial = ROOT / "source-metadata-first.json"
if not initial.exists(): initial.write_bytes((ROOT / "source-metadata.json").read_bytes())
def get(url):
    p = subprocess.run(["curl","--fail","--silent","--show-error","--max-time","12",url], capture_output=True,timeout=15)
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
        meta["files"].append(dict(path=path,url=url,sha256=hashlib.sha256(data).hexdigest()))
except Exception as e: meta["errors"].append(str(e))
(ROOT / "source-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
