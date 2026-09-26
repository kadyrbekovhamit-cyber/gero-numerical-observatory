"""Frozen confirmation runner, stores all results including failed gates."""
import argparse,json,datetime,time
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
from flint import ctx
from lab.iv_interval import Contract,QuoteInterval,invert_quote
from lab.iv_reference_v2 import check_witness,witnesses,number,value
from lab.iv_corpus_v2 import verify,ROOT


def run(output,portable=False):
    assert not output.exists(),'Use a fresh output path'
    freeze=verify(portable);ctx.threads=1
    cases=json.loads((ROOT/'iv-evidence/corpus-v2.json').read_text());rows=[];start=time.monotonic()
    for case in cases:
        c=Contract(**case['contract']);q=QuoteInterval(**case['quote'])
        result=invert_quote(c,q,**case['kwargs'])
        controls=[check_witness(c.as_dict(),w) for w in witnesses(result)]
        gates={'status':result['status']==case['expected_status'],
               'resolution':result['resolved']==case['expected_resolved'],
               'witnesses':all(x.get('pass',True) for x in controls)}
        if 'expected_endpoint_state' in case:
            gates['budget_state']=all(result[k]['state']==case['expected_endpoint_state']
                                      for k in ('lower_endpoint','upper_endpoint'))
        for key in ('lower_endpoint','upper_endpoint'):
            e=result.get(key,{})
            if e.get('state')=='converged':gates[key+'_tolerance']=F(e['upper'])-F(e['lower'])<=F(result['tolerance'])
        if 'compatible_sigma' in case:
            seed=F(case['compatible_sigma'])
            gates['seed_in_outer']=(F(result['outer_lower'])<=seed and
                                   (result['outer_upper'] is None or seed<=F(result['outer_upper'])))
            with mp.workdps(260):
                price=value(c.as_dict(),seed)
                gates['independent_seed_in_quote']=bool(number(q.lower)<price<number(q.upper))
        rows.append({'id':case['id'],'family':case['family'],'result':result,'controls':controls,'gates':gates})
    groups=defaultdict(list)
    for case,row in zip(cases,rows):
        if 'symmetry_group' in case:groups[case['symmetry_group']].append((case,row))
    symmetry=[]
    for group,entries in groups.items():
        for key in ('lower_endpoint','upper_endpoint'):
            brackets=[]
            for case,row in entries:
                e=row['result'][key];m=F(case['sigma_multiplier'])
                brackets.append((m*F(e['lower']),m*F(e['upper'])))
            passed=max(a for a,b in brackets)<=min(b for a,b in brackets)
            symmetry.append({'group':group,'endpoint':key,'pass':passed})
    summary={'scenarios':len(rows),'all_gates_pass':sum(all(r['gates'].values()) for r in rows),
             'families':dict(Counter(r['family'] for r in rows)),
             'statuses':dict(Counter(r['result']['status'] for r in rows)),
             'resolved':sum(r['result']['resolved'] for r in rows),
             'expected_budget_limited':sum(not c['expected_resolved'] for c in cases),
             'checked_witnesses':sum(x['checked'] for r in rows for x in r['controls']),
             'failed_witnesses':sum(not x.get('pass',True) for r in rows for x in r['controls']),
             'uncertain_comparisons':sum(not x['checked'] for r in rows for x in r['controls']),
             'symmetry_checks':len(symmetry),'symmetry_pass':sum(x['pass'] for x in symmetry),
             'highest_precision_bits':max(r['result']['highest_precision_bits'] for r in rows)}
    data={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'elapsed_seconds':time.monotonic()-start,'freeze_time_utc':freeze['time_utc'],
          'portable_dependency_check':portable,'summary':summary,'symmetry':symmetry,'rows':rows}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if summary['all_gates_pass']==len(rows) and summary['symmetry_pass']==len(symmetry) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--portable',action='store_true',help='check versions but allow platform-specific flint binaries')
    args=p.parse_args();raise SystemExit(run(args.output,args.portable))
