"""Read the small set of comments on relevant public GitHub search hits."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
saved = ROOT / "duplicate-comments.json"
results = json.loads(saved.read_text()) if saved.exists() else []
seen = {row["number"] for row in results if "error" not in row}
for query in json.loads((ROOT / "duplicate-search.json").read_text()):
    for item in query.get("result", {}).get("items", []):
        if not item["comments"] or item["number"] in seen:
            continue
        seen.add(item["number"])
        url = item["comments_url"] + "?per_page=100"
        p = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--max-time", "12", url],
                           capture_output=True, timeout=15)
        row = dict(number=item["number"], url=url)
        if p.returncode:
            row["error"] = p.stderr.decode()
        else:
            row["comments"] = json.loads(p.stdout)
        results.append(row)
        (ROOT / "duplicate-comments.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
