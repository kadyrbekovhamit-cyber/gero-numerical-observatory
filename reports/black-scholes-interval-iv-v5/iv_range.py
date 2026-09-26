"""Command-line quote interval inversion; JSON preserves exact rational endpoints."""
import argparse,json
from flint import ctx
from lab.iv_interval import Contract,QuoteInterval,invert_quote,rounding_cell


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spot','strike','time'):p.add_argument('--'+n,required=True)
    p.add_argument('--rate',default='0');p.add_argument('--yield-rate',default='0')
    p.add_argument('--option',choices=('call','put'),default='call')
    p.add_argument('--rounded-price',help='finite binary64 observation, nearest-even rounding assumed')
    p.add_argument('--bid',help='exact decimal/rational lower price');p.add_argument('--ask')
    p.add_argument('--tolerance',default='1e-12',help='absolute sigma root bracket tolerance')
    a=p.parse_args()
    if a.rounded_price is not None:
        if a.bid is not None or a.ask is not None:p.error('choose a rounded price OR bid and ask')
        q=rounding_cell(float(a.rounded_price))
    else:
        if a.bid is None or a.ask is None:p.error('provide --rounded-price OR both --bid and --ask')
        q=QuoteInterval(a.bid,a.ask)
    ctx.threads=1
    c=Contract(a.spot,a.strike,a.time,a.rate,a.yield_rate,a.option)
    print(json.dumps(invert_quote(c,q,tolerance=a.tolerance),indent=2))


if __name__=='__main__':main()
