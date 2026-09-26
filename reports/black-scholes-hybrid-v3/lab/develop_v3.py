"""Screen v3 against 604 previously disclosed inputs and cached references.

Saved reference strings retain 35 significant digits; sufficient for these
mixed practical gates, not a new independent high-precision confirmation.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import mpmath as mp
from lab.black_scholes_v3 import prices
from lab.corpus import arguments
from lab.metrics import round_binary64

ROOT = Path(__file__).resolve().parents[1]

def main():
    output = ROOT/'evidence/development-v3-draft-01.json'
    if output.exists():
        raise FileExistsError(output)
    rows = []
    with mp.workdps(80):
        for n in (1, 2):
            old = json.loads((ROOT/f'evidence/benchmark-v{n}.json').read_text())['rows']
            for case in old:
                row = {'source': f'v{n}', 'id': case['id']}
                try:
                    got = prices(*arguments(case))
                    ref = mp.mpf(case['reference_otm'])
                    logref = mp.mpf(case['reference_log_otm'])
                    value = getattr(got, case['otm_option'])
                    limits = (max(mp.mpf('1e-10')*ref, mp.mpf(math.ulp(round_binary64(ref)))),
                              max(mp.mpf('2e-11'), 16*mp.mpf(math.ulp(float(logref)))))
                    row.update(method=got.method,
                               price_pass=bool(abs(mp.mpf(value)-ref) <= limits[0]),
                               log_pass=bool(abs(mp.mpf(got.log_otm_value)-logref) <= limits[1]),
                               correct_side=got.otm_option == case['otm_option'],
                               relative_error=mp.nstr(abs(mp.mpf(value)-ref)/ref,25),
                               log_absolute_error=mp.nstr(abs(mp.mpf(got.log_otm_value)-logref),25))
                except Exception as exc:
                    row['exception'] = f'{type(exc).__name__}: {exc}'
                rows.append(row)
    failures = [r for r in rows if 'exception' in r or not all(r[k] for k in ('price_pass','log_pass','correct_side'))]
    report = {'candidate_sha256':hashlib.sha256((ROOT/'lab/black_scholes_v3.py').read_bytes()).hexdigest(),
              'status':'development; cached rounded decimal references',
              'summary':{'cases':len(rows),'failures':len(failures),
                         'methods':dict(Counter(r.get('method','exception') for r in rows))},
              'failures':failures,'rows':rows}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('summary','failures')},indent=2))

if __name__ == '__main__':
    main()
