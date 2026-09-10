"""Retry the single GitHub response marked incomplete; retain the first response."""
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode
ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "duplicate-search.json").read_text())
(ROOT / "duplicate-search-first.json").write_text(json.dumps(rows, indent=2)+"\n")
for row in rows:
    if not row.get("result", {}).get("incomplete_results"):
        continue
    url = "https://api.github.com/search/issues?" + urlencode(dict(q=row["query"], per_page=100, sort="updated"))
    p = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--max-time", "12", url],
                       capture_output=True, timeout=15)
    retry = dict(query=row["query"], url=url)
    if p.returncode:
        retry["error"] = p.stderr.decode()
    else:
        retry["result"] = json.loads(p.stdout)
        if not retry["result"]["incomplete_results"]:
            row.update(retry)
    (ROOT / "duplicate-search-retry.json").write_text(json.dumps(retry, indent=2)+"\n")
    print(json.dumps(retry), flush=True)
(ROOT / "duplicate-search.json").write_text(json.dumps(rows, indent=2)+"\n")
