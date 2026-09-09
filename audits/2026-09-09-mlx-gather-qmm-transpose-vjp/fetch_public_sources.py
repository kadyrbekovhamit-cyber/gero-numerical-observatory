"""Read public MLX source and issue metadata; no submission or account content."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent


def api(path):
    result = subprocess.run(
        ['curl', '--fail', '--silent', '--show-error', '--max-time', '20',
         'https://api.github.com/' + path], capture_output=True, text=True,
        timeout=25)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def main():
    metadata = dict(checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    files=[], errors=[])
    try:
        sha = api('repos/ml-explore/mlx/commits/main')['sha']
    except Exception as error:
        metadata['errors'].append(dict(path='commits/main', error=str(error)))
        (ROOT / 'source-fetch-error.json').write_text(json.dumps(metadata, indent=2) + '\n')
        raise
    metadata['upstream_revision'] = sha
    print('main', sha, flush=True)
    for path in ('mlx/primitives.cpp', 'mlx/ops.cpp', 'python/tests/test_quantized.py'):
        try:
            url = f'https://raw.githubusercontent.com/ml-explore/mlx/{sha}/{path}'
            result = subprocess.run(['curl', '--fail', '--silent', '--show-error',
                                     '--max-time', '20', url], capture_output=True,
                                    timeout=25, check=True)
            (ROOT / ('upstream-' + path.replace('/', '_'))).write_bytes(result.stdout)
            metadata['files'].append(dict(path=path, url=url,
                sha256=hashlib.sha256(result.stdout).hexdigest()))
        except Exception as error:
            metadata['errors'].append(dict(path=path, error=str(error)))
        (ROOT / 'source-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    searches = []
    for query in (
        'repo:ml-explore/mlx "gather_qmm" "gradient"',
        'repo:ml-explore/mlx "gather_qmm" "transpose"',
        'repo:ml-explore/mlx "GatherQMM" "vjp"',
        'repo:ml-explore/mlx "gather_qmm" "scales"',
        'repo:ml-explore/mlx "gather_qmm" "biases"',
    ):
        try:
            result = api('search/issues?' + urlencode(dict(q=query, per_page=100)))
            searches.append(dict(query=query, result=result))
            print(query, result['total_count'],
                  [(x['number'], x['title']) for x in result['items']], flush=True)
        except Exception as error:
            searches.append(dict(query=query, error=str(error)))
        (ROOT / 'duplicate-search.json').write_text(json.dumps(searches, indent=2) + '\n')


if __name__ == '__main__':
    main()
