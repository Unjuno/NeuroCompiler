"""Exact rational differential-polynomial IR. Not a general identity oracle."""
from __future__ import annotations
from fractions import Fraction
from collections import defaultdict
import hashlib,json
import numpy as np
MAX_TERMS=4096;MAX_DEGREE=32

def leaf(i,c=1):return ('leaf',int(i),Fraction(c))
def add(a,b):return ('add',a,b)
def mul(a,b):return ('mul',a,b)
def diff(a):return ('D',a)
def scale(a,c):return ('scale',Fraction(c),a)
def clean(p):
    p={k:v for k,v in p.items() if v}
    if len(p)>MAX_TERMS or any(len(k)>MAX_DEGREE for k in p):raise ValueError('IR expansion budget exceeded')
    return dict(sorted(p.items()))
def p_add(a,b):
    p=defaultdict(Fraction)
    for k,v in list(a.items())+list(b.items()):p[k]+=v
    return clean(p)
def p_mul(a,b):
    p=defaultdict(Fraction)
    for ka,va in a.items():
        for kb,vb in b.items():
            p[tuple(sorted(ka+kb))]+=va*vb
            if len(p)>MAX_TERMS:raise ValueError('IR expansion budget exceeded')
    return clean(p)
def p_diff(a):
    p=defaultdict(Fraction)
    for key,v in a.items():
        for j,(i,n) in enumerate(key):
            k=list(key);k[j]=(i,n+1);p[tuple(sorted(k))]+=v
    return clean(p)
def canonical(e):
    if e[0]=='leaf':return clean({((e[1],0),):Fraction(e[2])})
    if e[0]=='scale':return clean({k:e[1]*v for k,v in canonical(e[2]).items()})
    if e[0]=='D':return p_diff(canonical(e[1]))
    if e[0]=='add':return p_add(canonical(e[1]),canonical(e[2]))
    if e[0]=='mul':return p_mul(canonical(e[1]),canonical(e[2]))
    raise ValueError(e[0])
def serialize(p):return [[list(map(list,k)),v.numerator,v.denominator] for k,v in p.items()]
def ir_hash(p):return hashlib.sha256(json.dumps(serialize(p),separators=(',',':')).encode()).hexdigest()
def gaussian_jet(x,mu,sigma,n):
    z=(np.asarray(x,dtype=np.float64)-mu)/sigma;hm,h=np.ones_like(z),z.copy()
    if n==0:h=hm
    else:
        for k in range(1,n):hm,h=h,z*h-k*hm
    return (-1./sigma)**n*h*np.exp(-.5*z*z)
def evaluate(p,x,descriptors):
    x=np.asarray(x,dtype=np.float64);out=np.zeros_like(x);cache={}
    for factors,c in p.items():
        v=np.full_like(x,float(c))
        for atom in factors:
            if atom not in cache:
                i,n=atom;mu,sig=descriptors[i];cache[atom]=gaussian_jet(x,mu,sig,n)
            v*=cache[atom]
        out+=v
    return out
