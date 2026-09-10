"""Verify the published manifest without importing MLX or running numerical work."""
import hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256SUMS.json').read_text())
failures=[name for name,sha in manifest.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha]
print(json.dumps({'files':len(manifest),'failures':failures}))
raise SystemExit(bool(failures))
