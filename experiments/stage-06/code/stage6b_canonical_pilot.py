import math,random,time,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage6b_pilot'));OUT.mkdir(parents=True,exist_ok=True);SEED=int(os.getenv('SEED','0'))
H=32;P=3*H+1;E=48;STEPS=2600;BATCH=8;Q=48;N=100;E_MIN=.02;SEP_MIN=.02;EPS=1e-8
ALL=[(-.8,.18),(-.4,.18),(0.,.18),(.4,.18),(.8,.18),(-.8,.30),(-.4,.30),(0.,.30),(.4,.30),(.8,.30),(-.2,.24),(.2,.24),(.6,.24),(-.6,.24)]
TRAIN=list(range(10));DESC=torch.tensor([[m,s,(s-.24)/.12,m*m] for m,s in ALL],dtype=torch.float32)
def leaf(i,c):return ('leaf',int(i),float(c))
def add(a,b):return ('add',a,b)
def mul(a,b):return ('mul',a,b)
def adds(xs):
 z=xs[0]
 for q in xs[1:]:z=add(z,q)
 return z
def prim(i,x):
 m,s=ALL[i];z=x.squeeze(-1);return torch.exp(-.5*((z-m)/s)**2).view(*x.shape[:-1],1)
def ev(e,x):
 if e[0]=='leaf':return e[2]*prim(e[1],x)
 a=ev(e[1],x);b=ev(e[2],x);return a+b if e[0]=='add' else a*b
# Exact polynomial normal form over primitive-function symbols.
def poly(e):
 if e[0]=='leaf':return {(e[1],):float(e[2])}
 A=poly(e[1]);B=poly(e[2]);out={}
 if e[0]=='add':
  for k,v in A.items():out[k]=out.get(k,0.0)+v
  for k,v in B.items():out[k]=out.get(k,0.0)+v
 else:
  for ka,va in A.items():
   for kb,vb in B.items():
    k=tuple(sorted(ka+kb));out[k]=out.get(k,0.0)+va*vb
 return {k:v for k,v in out.items() if abs(v)>1e-12}
def canon(e):
 p=poly(e);terms=[]
 for key in sorted(p):
  c=p[key]
  fs=[leaf(key[0],c)]+[leaf(i,1.0) for i in key[1:]]
  t=fs[0]
  for f in fs[1:]:t=mul(t,f)
  terms.append(t)
 if not terms:return leaf(0,0.0)
 return adds(terms)
def rc(r):return (1 if r.random()>.5 else -1)*(.55+.65*r.random())
TT=['leaf','add2','mul2','add_mul_leaf','mul_add_leaf']
def train_expr(name,r):
 n={'leaf':1,'add2':2,'mul2':2,'add_mul_leaf':3,'mul_add_leaf':3}[name];ids=r.sample(TRAIN,n);L=[leaf(i,rc(r)) for i in ids]
 if name=='leaf':return L[0]
 if name=='add2':return add(L[0],L[1])
 if name=='mul2':return mul(L[0],L[1])
 if name=='add_mul_leaf':return add(mul(L[0],L[1]),L[2])
 return mul(add(L[0],L[1]),L[2])
class GR:
 def random(self):return random.random()
 def sample(self,p,n):return random.sample(p,n)
def sample_train(B):
 r=GR();return [train_expr(random.choice(TT),r) for _ in range(B)]
REW=['distribute3','mul_assoc3','distribute4','product_expand4','product_expand5']
def rw_pair(k,L):
 if k=='distribute3':a,b,c=L;return mul(a,add(b,c)),add(mul(a,b),mul(a,c))
 if k=='mul_assoc3':a,b,c=L;return mul(mul(a,b),c),mul(a,mul(b,c))
 if k=='distribute4':a,b,c,d=L;return add(mul(add(a,b),c),d),add(add(mul(a,c),mul(b,c)),d)
 if k=='product_expand4':a,b,c,d=L;return mul(add(a,b),add(c,d)),adds([mul(a,c),mul(a,d),mul(b,c),mul(b,d)])
 if k=='product_expand5':a,b,c,d,e=L;return add(mul(add(a,b),add(c,d)),e),adds([mul(a,c),mul(a,d),mul(b,c),mul(b,d),e])
X=torch.linspace(-1,1,256).view(-1,1)
def rewrite_bank(kind,n,seed):
 r=random.Random(seed);nl={'distribute3':3,'mul_assoc3':3,'distribute4':4,'product_expand4':4,'product_expand5':5}[kind];pairs=[]
 while len(pairs)<n:
  ids=r.sample(TRAIN,nl);L=[leaf(i,rc(r)) for i in ids];A,B=rw_pair(kind,L);y=ev(A,X)
  if float((y*y).mean())<E_MIN:continue
  pairs.append((A,B))
 return pairs
# Non-equivalent paired structural controls, same leaves/coefficients.
def structural_pair(kind,L):
 a,b,c,d=L
 if kind=='placement':return add(mul(a,b),add(c,d)),mul(add(a,b),add(c,d))
 if kind=='rewire':return add(mul(a,b),mul(c,d)),add(mul(a,c),mul(b,d))
def structural_bank(kind,n,seed):
 r=random.Random(seed);pairs=[]
 while len(pairs)<n:
  ids=r.sample(TRAIN,4);L=[leaf(i,rc(r)) for i in ids];A,B=structural_pair(kind,L);ya,yb=ev(A,X),ev(B,X)
  if min(float((ya*ya).mean()),float((yb*yb).mean()))<E_MIN or float(((ya-yb)**2).mean())<SEP_MIN:continue
  pairs.append((A,B))
 return pairs
def tf(th,x):
 B=th.shape[0];i=0;W1=th[:,i:i+H].view(B,H,1);i+=H;b1=th[:,i:i+H].view(B,1,H);i+=H;W2=th[:,i:i+H].view(B,1,H);i+=H;b2=th[:,i:i+1].view(B,1,1);return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2
def grad(th,x,y):
 B=th.shape[0];i=0;W1=th[:,i:i+H];i+=H;b1=th[:,i:i+H];i+=H;W2=th[:,i:i+H];i+=H;b2=th[:,i:i+1];xx=x.squeeze(-1);yy=y.squeeze(-1);z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:];a=torch.tanh(z);pred=(a*W2[:,None,:]).sum(2)+b2;d=2*(pred-yy)/xx.shape[1];gb2=d.sum(1,keepdim=True);gW2=(d[:,:,None]*a).sum(1);gz=d[:,:,None]*W2[:,None,:]*(1-a*a);gb1=gz.sum(1);gW1=(gz*xx[:,:,None]).sum(1);return torch.cat([gW1,gb1,gW2,gb2],1)
class LE(torch.nn.Module):
 def __init__(self):super().__init__();self.register_buffer('d',DESC.clone());self.l1=torch.nn.Linear(4,E);self.l2=torch.nn.Linear(E,E)
 def emb(self,i):return torch.tanh(self.l2(torch.tanh(self.l1(self.d[i]))))
class Tree(torch.nn.Module):
 def __init__(self,canonical=False):
  super().__init__();self.canonical=canonical;self.le=LE();self.m1=torch.nn.Linear(3*E,E);self.m2=torch.nn.Linear(E,E);self.base=torch.nn.Parameter(torch.randn(P)*.12);self.rho=torch.nn.Linear(E,P);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
 def enc(self,e):
  if e[0]=='leaf':return e[2]*self.le.emb(e[1])
  a=self.enc(e[1]);b=self.enc(e[2])
  if e[0]=='add':return a+b
  return torch.tanh(self.m2(torch.tanh(self.m1(torch.cat([a+b,a*b,torch.abs(a-b)])))))
 def forward(self,es):
  if self.canonical:es=[canon(e) for e in es]
  return torch.stack([self.base+self.rho(torch.tanh(self.enc(e))) for e in es])
def sample_xy(es,n):
 B=len(es);x=2*torch.rand(B,n,1)-1;y=torch.stack([ev(es[i],x[i]) for i in range(B)]);return x,y
def train(canonical,off):
 torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);m=Tree(canonical);o=torch.optim.Adam(m.parameters(),lr=1e-3)
 for _ in range(STEPS):
  es=sample_train(BATCH);x,y=sample_xy(es,Q);th=m(es);g=grad(th,x,y).detach()/BATCH;o.zero_grad();torch.autograd.backward(th,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);o.step()
 return m
rw={n:rewrite_bank(n,N,20268000+i) for i,n in enumerate(REW)};controls={n:structural_bank(n,N,20268100+i) for i,n in enumerate(['placement','rewire'])}
rows=[];tim=[]
for fam,can,off in [('Syntax',False,800000),('Canonical',True,800000)]: # identical initialization/training RNG seed
 t=time.time();m=train(can,off);tim.append({'seed':SEED,'family':fam,'train_seconds':time.time()-t})
 with torch.no_grad():
  for name,pairs in rw.items():
   es=[z for p in pairs for z in p];xq=X.view(1,256,1).repeat(len(es),1,1);y=torch.stack([ev(e,xq[i]) for i,e in enumerate(es)]);th=m(es);pred=tf(th,xq)
   for pid in range(N):
    a,b=2*pid,2*pid+1;energy=float((y[a]**2).mean());rows.append({'seed':SEED,'family':fam,'test_type':'rewrite','name':name,'pair_id':pid,'mse_a':float(((pred[a]-y[a])**2).mean()),'mse_b':float(((pred[b]-y[b])**2).mean()),'pred_disagreement':float(((pred[a]-pred[b])**2).mean()),'pred_disagreement_over_energy':float(((pred[a]-pred[b])**2).mean())/(energy+EPS),'max_weight_diff':float(torch.max(torch.abs(th[a]-th[b]))),'target_disagreement':float(((y[a]-y[b])**2).mean())})
  for name,pairs in controls.items():
   es=[z for p in pairs for z in p];xq=X.view(1,256,1).repeat(len(es),1,1);y=torch.stack([ev(e,xq[i]) for i,e in enumerate(es)]);th=m(es);pred=tf(th,xq)
   for pid in range(N):
    a,b=2*pid,2*pid+1;sep=float(((y[a]-y[b])**2).mean());rows.append({'seed':SEED,'family':fam,'test_type':'structural_control','name':name,'pair_id':pid,'mse_a':float(((pred[a]-y[a])**2).mean()),'mse_b':float(((pred[b]-y[b])**2).mean()),'pred_disagreement':float(((pred[a]-pred[b])**2).mean()),'pred_disagreement_over_energy':float(((pred[a]-pred[b])**2).mean())/(sep+EPS),'max_weight_diff':float(torch.max(torch.abs(th[a]-th[b]))),'target_disagreement':sep})
pd.DataFrame(rows).to_csv(OUT/f'canonical_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
print(pd.DataFrame(rows).groupby(['test_type','name','family'])[['mse_a','mse_b','pred_disagreement','pred_disagreement_over_energy','max_weight_diff','target_disagreement']].median().to_string());print(tim)
