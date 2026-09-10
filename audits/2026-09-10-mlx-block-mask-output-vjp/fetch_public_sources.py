"""Fetch public source and public issue metadata, without account access."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent


def get(url):
    r = subprocess.run(["curl", "--fail", "--silent", "--show-error",
                        "--max-time", "12", url], capture_output=True, timeout=15)
    if r.returncode:
        raise RuntimeError(r.stderr.decode().strip())
    return r.stdout


def main():
    meta = dict(checked_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                files=[], errors=[])
    try:
        sha = json.loads(get("https://api.github.com/repos/ml-explore/mlx/commits/main"))["sha"]
        meta["upstream_revision"] = sha
        print("main", sha, flush=True)
        for path in ("mlx/primitives.cpp", "python/tests/test_blas.py"):
            try:
                url = f"https://raw.githubusercontent.com/ml-explore/mlx/{sha}/{path}"
                data = get(url)
                (ROOT / ("upstream-" + path.replace("/", "_"))).write_bytes(data)
                meta["files"].append(dict(path=path, url=url,
                                         sha256=hashlib.sha256(data).hexdigest()))
            except Exception as error:
                meta["errors"].append(dict(path=path, error=str(error)))
    except Exception as error:
        meta["errors"].append(dict(path="commits/main", error=str(error)))
    (ROOT / "source-metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    searches = []
    for query in (
        'repo:ml-explore/mlx "block_masked_mm" "gradient"',
        'repo:ml-explore/mlx "block_masked_mm" "out_mask"',
        'repo:ml-explore/mlx "BlockMaskedMM" "vjp"',
    ):
        try:
            url = "https://api.github.com/search/issues?" + urlencode(dict(q=query, per_page=100))
            result = json.loads(get(url))
            searches.append(dict(query=query, url=url, result=result))
            print(query, result["total_count"],
                  [(x["number"], x["title"]) for x in result["items"]], flush=True)
        except Exception as error:
            searches.append(dict(query=query, error=str(error)))
            print(query, str(error), flush=True)
        (ROOT / "duplicate-search.json").write_text(json.dumps(searches, indent=2)+"\n")


if __name__ == "__main__":
    main()
