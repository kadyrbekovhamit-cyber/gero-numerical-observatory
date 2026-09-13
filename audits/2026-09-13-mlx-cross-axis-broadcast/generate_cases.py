"""Small exact-arithmetic cross-product oracle; NumPy is an independent check."""
from pathlib import Path
import itertools, json, math, warnings
import numpy as np

R=Path(__file__).resolve().parent
def product(s): return math.prod(s)
def coordinates(s): return itertools.product(*(range(x) for x in s))
def offset(c,s):
    out=0
    for x,n in zip(c,s): out=out*n+x
    return out
def shape(batch,k,axis):
    r=len(batch)+1; ax=axis%r
    return list(batch[:ax])+[k]+list(batch[ax:])
def broadcast(a,b):
    n=max(len(a),len(b));a=[1]*(n-len(a))+a;b=[1]*(n-len(b))+b
    return [y if x==1 else x for x,y in zip(a,b)]
def oracle(av,ash,bv,bsh,axis,w=None):
    aa=axis%len(ash);ba=axis%len(bsh)
    ab=ash[:aa]+ash[aa+1:];bb=bsh[:ba]+bsh[ba+1:]
    batch=broadcast(ab,bb);outsh=shape(batch,3,axis);oa=axis%len(outsh)
    out=[0.]*product(outsh);ga=[0.]*len(av);gb=[0.]*len(bv)
    def source(bc,bs,s,ax,k):
        if k>=s[ax]: return None
        q=list(bc[len(batch)-len(bs):]);q=[0 if n==1 else i for i,n in zip(q,bs)]
        q.insert(ax,k);return offset(q,s)
    # Levi-Civita expansion, independent of MLX's reshape/split implementation.
    for bc in coordinates(batch):
        for i,j,k,sgn in [(0,1,2,1),(0,2,1,-1),(1,2,0,1),(1,0,2,-1),(2,0,1,1),(2,1,0,-1)]:
            ai=source(bc,ab,ash,aa,j);bi=source(bc,bb,bsh,ba,k)
            if ai is None or bi is None: continue
            oc=list(bc);oc.insert(oa,i);oi=offset(oc,outsh)
            out[oi]+=sgn*av[ai]*bv[bi]
            if w is not None:ga[ai]+=sgn*w[oi]*bv[bi];gb[bi]+=sgn*w[oi]*av[ai]
    return out,outsh,ga,gb
def numpy_oracle(av,ash,bv,bsh,axis):
    a=np.moveaxis(np.array(av).reshape(ash),axis,-1)
    b=np.moveaxis(np.array(bv).reshape(bsh),axis,-1)
    if a.shape[-1]==2:a=np.concatenate([a,np.zeros(a.shape[:-1]+(1,))],axis=-1)
    if b.shape[-1]==2:b=np.concatenate([b,np.zeros(b.shape[:-1]+(1,))],axis=-1)
    return np.moveaxis(np.cross(a,b),-1,axis)

pairs=[([],[]),([],[3]),([3],[]),([],[2]),([2],[]),([],[2,3]),([2,3],[]),
       ([1],[3]),([3],[1]),([3],[2,3]),([2,3],[3]),([1,3],[2,1]),([2,1],[1,3]),
       ([1,2,1],[3,1,4]),([3,1,4],[1,2,1]),([],[0]),([0],[]),([1],[0]),([0],[1]),
       ([0,3],[1,3]),([2,1],[3])]
cases=[];finite_difference_checks=0
for pi,(ab,bb) in enumerate(pairs):
    rank=min(len(ab),len(bb))+1
    for axis in list(range(rank))+list(range(-rank,0)):
        for ka,kb in itertools.product((2,3),repeat=2):
            ash=shape(ab,ka,axis);bsh=shape(bb,kb,axis)
            av=[float((i*3+pi)%7-3) for i in range(product(ash))]
            bv=[float((i*5+pi+1)%7-3) for i in range(product(bsh))]
            expected,outsh,_,_=oracle(av,ash,bv,bsh,axis)
            nr=numpy_oracle(av,ash,bv,bsh,axis)
            assert list(nr.shape)==outsh and nr.ravel().tolist()==expected
            w=[float((i*2+1)%5-2) for i in range(len(expected))]
            _,_,ga,gb=oracle(av,ash,bv,bsh,axis,w)
            ta=[float(i%3-1) for i in range(len(av))];tb=[float((i+1)%3-1) for i in range(len(bv))]
            ja=oracle(ta,ash,bv,bsh,axis)[0];jb=oracle(av,ash,tb,bsh,axis)[0]
            if pi in (0,1,2,3,9,11) and ka==kb==3 and axis in (0,-1):
                for which in (0,1):
                    values=(av,bv)[which];g=(ga,gb)[which]
                    for idx in range(len(values)):
                        plus=list(values);minus=list(values);plus[idx]+=1/1024;minus[idx]-=1/1024
                        args1=(plus,ash,bv,bsh,axis) if which==0 else (av,ash,plus,bsh,axis)
                        args2=(minus,ash,bv,bsh,axis) if which==0 else (av,ash,minus,bsh,axis)
                        fp=sum(x*y for x,y in zip(oracle(*args1)[0],w));fm=sum(x*y for x,y in zip(oracle(*args2)[0],w))
                        assert (fp-fm)*512==g[idx]
                        finite_difference_checks+=1
            cases.append(dict(name=f'p{pi}_axis{axis}_k{ka}{kb}',axis=axis,ashape=ash,bshape=bsh,
                a=av,b=bv,expected=expected,shape=outsh,w=w,ga=ga,gb=gb,ta=ta,tb=tb,
                jvp=[x+y for x,y in zip(ja,jb)],autodiff=ka==kb==3 and pi<15,
                control=(len(ash)==len(bsh) or axis<0)))
(R/'cases.json').write_text(json.dumps(cases,separators=(',',':'))+'\n')
(R/'oracle-validation.json').write_text(json.dumps(dict(numpy_version=np.__version__,
    cases=len(cases),numpy_cross_agreements=len(cases),host_finite_difference_checks=finite_difference_checks,
    note='All two-component inputs are explicitly padded with a zero third component for NumPy, preserving MLX output semantics.'),indent=2)+'\n')
print((R/'oracle-validation.json').read_text())
