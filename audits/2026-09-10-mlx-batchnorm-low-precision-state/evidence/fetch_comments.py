"""Read a bounded set of public comments closest to the BatchNorm finding."""
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parent
rows = []
for path in ("issues/3817/comments", "issues/1960/comments", "issues/217/comments", "pulls/3817/comments"):
    url = "https://api.github.com/repos/ml-explore/mlx/"+path+"?per_page=100"
    p = subprocess.run(["curl","--fail","--silent","--show-error","--max-time","12",url], capture_output=True,timeout=15)
    row = dict(path=path,url=url)
    if p.returncode: row["error"] = p.stderr.decode()
    else: row["comments"] = json.loads(p.stdout)
    rows.append(row)
    (ROOT / "duplicate-comments.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(path, len(row.get("comments",[])),row.get("error"),flush=True)
