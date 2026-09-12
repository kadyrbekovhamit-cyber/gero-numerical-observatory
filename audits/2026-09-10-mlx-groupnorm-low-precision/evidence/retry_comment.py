"""Retry the single unavailable public comment request; retain its first result."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
path=ROOT/'duplicate-comments.json'
rows=json.loads(path.read_text())
(ROOT/'duplicate-comments-first.json').write_bytes(path.read_bytes())
for row in rows:
    if 'error' not in row:continue
    p=subprocess.run(['curl','--fail','--silent','--show-error','--max-time','12',row['url']],capture_output=True,timeout=15)
    if p.returncode:
        print(row['number'],p.stderr.decode(),flush=True)
        continue
    row['comments']=json.loads(p.stdout)
    row.pop('error',None)
    print(row['number'],[(c['html_url'],c['body']) for c in row['comments']],flush=True)
path.write_text(json.dumps(rows,indent=2)+'\n')
