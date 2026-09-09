"""Read public nntrainer issue/PR evidence, sequentially. No writes to GitHub."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
gh = '/opt/homebrew/bin/gh'


def api(endpoint, *args):
    return json.loads(subprocess.check_output(
        [gh, 'api', endpoint, *args], text=True, timeout=15))


target = root / 'duplicate-search.json'
record = (json.loads(target.read_text()) if target.exists() else
          {'checked_at': datetime.now(timezone.utc).isoformat(), 'queries': []})
record.setdefault('errors', [])
if 'head' not in record:
    record['head'] = api('repos/nntrainer/nntrainer/commits/main')
queries = [
    'repo:nntrainer/nntrainer pooling padding',
    'repo:nntrainer/nntrainer pooling backward',
    'repo:nntrainer/nntrainer pooling gradient',
    'repo:nntrainer/nntrainer "average" "padding"',
    'repo:nntrainer/nntrainer "height_stride_end"',
]
for query in queries:
    if any(item['query'] == query for item in record['queries']):
        continue
    try:
        result = api('search/issues', '-X', 'GET', '-f', 'q=' + query,
                     '-f', 'per_page=100')
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        record['errors'].append({'query': query, 'error': str(exc)})
        target.write_text(json.dumps(record, indent=2) + '\n')
        print('Unavailable:', query, flush=True)
        continue
    for page in range(2, (min(result['total_count'], 1000) + 99) // 100 + 1):
        extra = api('search/issues', '-X', 'GET', '-f', 'q=' + query,
                    '-f', 'per_page=100', '-f', 'page=' + str(page))
        result['items'].extend(extra['items'])
        result['incomplete_results'] |= extra['incomplete_results']
    record['queries'].append({'query': query, **result})
    (root / 'duplicate-search.json').write_text(json.dumps(record, indent=2) + '\n')
    print(query, result['total_count'], result['incomplete_results'], flush=True)

review_path = root / 'reviewed-public-reports.json'
reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
for number in (1045, 1051, 1360, 1577, 4085):
    item = reviews.get(str(number), {})
    try:
        if 'issue' not in item:
            item['issue'] = api('repos/nntrainer/nntrainer/issues/' + str(number))
        if 'pull_request' in item['issue']:
            if 'pull' not in item:
                item['pull'] = api('repos/nntrainer/nntrainer/pulls/' + str(number))
            if 'files' not in item:
                item['files'] = api('repos/nntrainer/nntrainer/pulls/' + str(number) + '/files',
                                    '-f', 'per_page=100', '-X', 'GET')
        item.pop('error', None)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        item['error'] = str(exc)
    reviews[str(number)] = item
    review_path.write_text(json.dumps(reviews, indent=2) + '\n')
    print(number, item.get('issue', {}).get('title', 'Unavailable'),
          item.get('error', ''), flush=True)
