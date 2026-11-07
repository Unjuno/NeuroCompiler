"""Exact rewrite properties, non-collapse controls, independent autodiff checks."""
import random,json
from fractions import Fraction
from pathlib import Path
import numpy as np,pandas as pd,torch
from differential_ir import leaf,add,mul,diff,scale,canonical,ir_hash,evaluate
ROOT=Path(__file__).resolve().parents[1]
DESC=[(-.6,.25),(-.15,.325),(.3,.4),(.65,.3)]
def ev(e,x):
    if e[0]=='leaf':
        mu,sig=DESC[e[1]];return float(e[2])*torch.exp(-.5*((x-mu)/sig)**2)
    if e[0]=='scale':return float(e[1])*ev(e[2],x)
    if e[0]=='add':return ev(e[1],x)+ev(e[2],x)
    if e[0]=='mul':return ev(e[1],x)*ev(e[2],x)
    if e[0]=='D':return torch.autograd.grad(ev(e[1],x).sum(),x,create_graph=True)[0]
    raise ValueError(e[0])
r=random.Random(718);rows=[]
for i in range(200):
    a,b,c=[leaf(r.randrange(4),Fraction(r.choice([-1,1])*r.randrange(1,16),r.randrange(2,11))) for _ in range(3)]
    laws={
      'linearity':(diff(add(a,b)),add(diff(a),diff(b))),
      'product':(diff(mul(a,b)),add(mul(diff(a),b),mul(a,diff(b)))),
      'second_product':(diff(diff(mul(a,b))),add(add(mul(diff(diff(a)),b),scale(mul(diff(a),diff(b)),2)),mul(a,diff(diff(b))))),
      'nested':(diff(mul(a,add(b,c))),add(add(mul(diff(a),b),mul(a,diff(b))),add(mul(diff(a),c),mul(a,diff(c))))),
      'rational':(add(scale(diff(a),Fraction(1,3)),scale(diff(a),Fraction(2,3))),diff(a)),
      'zero':(add(diff(a),scale(diff(a),-1)),scale(a,0))}
    for name,(aa,bb) in laws.items():
        pa,pb=canonical(aa),canonical(bb);assert pa==pb and ir_hash(pa)==ir_hash(pb)
        x=torch.linspace(-.97,.97,97,dtype=torch.float64,requires_grad=True)
        err=float(np.max(np.abs(ev(aa,x).detach().numpy()-evaluate(pa,x.detach().numpy(),DESC))))
        rows.append({'trial':i,'law':name,'identical_ir':True,'max_autodiff_error':err})
wrong=0
for i in range(200):
    a,b,c=[leaf(j,Fraction(r.randrange(2,12),7)) for j in (0,1,2)]
    wrong+=canonical(diff(mul(a,b)))==canonical(mul(diff(a),diff(b)))
    wrong+=canonical(add(mul(a,b),c))==canonical(mul(a,add(b,c)))
assert wrong==0
frame=pd.DataFrame(rows);assert frame.max_autodiff_error.max()<1e-9
out={'exact_rewrite_pairs':len(rows),'non_equivalent_controls':400,'false_collapses':wrong,'max_ir_vs_autodiff_abs_error_float64':float(frame.max_autodiff_error.max()),'scope':'commutative-ring and derivation rules; not all Gaussian identities'}
frame.to_csv(ROOT/'checks'/'differential_identity_tests.csv',index=False)
(ROOT/'checks'/'differential_ir_summary.json').write_text(json.dumps(out,indent=2));print(out)
