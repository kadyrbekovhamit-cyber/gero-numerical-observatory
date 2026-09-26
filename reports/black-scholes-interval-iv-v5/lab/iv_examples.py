"""Post-confirmation illustrations, not additional predeclared confirmation."""
import json
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
from flint import ctx
from lab.iv_interval import Contract,QuoteInterval,invert_quote,rounding_cell
from lab.iv_reference_v3 import value,witnesses,check_witness


def main():
    ctx.threads=1;rows=[]
    def add(name,c,q,**kwargs):
        result=invert_quote(c,q,**kwargs)
        checks=[check_witness(c.as_dict(),w) for w in witnesses(result)]
        assert all(x.get('pass',True) for x in checks)
        rows.append({'name':name,'result':result,'controls':checks})
    add('rounded_ITM_call_100',Contract(200,100,1),rounding_cell(100.0),tolerance='1e-24')
    put=Contract(200,100,1,option='put')
    with mp.workdps(260):p=float(F(mp.nstr(value(put.as_dict(),'1/20'),260)))
    add('rounded_OTM_put_same_contract_sigma_5pct',put,rounding_cell(p),tolerance='1e-20')
    add('ATM_price_band_7_to_9',Contract(100,100,1),QuoteInterval(7,9))
    c=Contract(113,113,1)
    with mp.workdps(260):price=F(mp.nstr(value(c.as_dict(),'1/2'),100))
    add('adaptive_precision_near_sigma_half',c,QuoteInterval(price,price),precisions=(64,128,256,512,1024))
    path=Path(__file__).resolve().parents[1]/'iv-evidence/illustrations.json'
    assert not path.exists();path.write_text(json.dumps({'purpose':'post-confirmation illustrations','rows':rows},indent=2)+'\n')
    for row in rows:
        z=row['result'];print(row['name'],z['resolved'],z['highest_precision_bits'],
                             float(F(z['outer_lower'])),None if z['outer_upper'] is None else float(F(z['outer_upper'])))


if __name__=='__main__':main()
