import math,random,time,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage6_pilot'));OUT.mkdir(parents=True,exist_ok=True);SEED=int(os.getenv('SEED','0'))
H=32;P=3*H+1;E=48;DIRECT_STEPS=2600;BATCH=8;QUERY_N=48;FINAL_N=100;E_MIN=.02;EPS=1e-8
ALL=[(-.8,.18),(-.4,.18),(0.,.18),(.4,.18),(.8,.18),(-.8,.30),(-.4,.30),(0.,.30),(.4,.30),(.8,.30),(-.2,.24),(.2,.24),(.6,.24),(-.6,.24)]
TRAIN_IDS=list(range(10));DESC=torch.tensor([[mu,sig,(sig-.24)/.12,mu*mu] for mu,sig in ALL],dtype=torch.float32)
def leaf(i,c):return ('leaf',int(i),float(c))
def add(a,b):return ('add',a,b)
def mul(a,b):return ('mul',a,b)
def adds(xs):
 z=xs[0]
 for q in xs[1:]:z=add(z,q)
 return z
def prim(i,x):
 mu,sig=ALL[i];z=x.squeeze(-1);return torch.exp(-.5*((z-mu)/sig)**2).view(*x.shape[:-1],1)
def ev(e,x):
 if e[0]=='leaf':return e[2]*prim(e[1],x)
 a=ev(e[1],x);b=ev(e[2],x);return a+b if e[0]=='add' else a*b
def rc(r):return (1 if r.random()>.5 else -1)*(.55+.65*r.random())
TRAIN_T=['leaf','add2','mul2','add_mul_leaf','mul_add_leaf']
def train_expr(name,r):
 n={'leaf':1,'add2':2,'mul2':2,'add_mul_leaf':3,'mul_add_leaf':3}[name];ids=r.sample(TRAIN_IDS,n);L=[leaf(i,rc(r)) for i in ids]
 if name=='leaf':return L[0]
 if name=='add2':return add(L[0],L[1])
 if name=='mul2':return mul(L[0],L[1])
 if name=='add_mul_leaf':return add(mul(L[0],L[1]),L[2])
 return mul(add(L[0],L[1]),L[2])
class GR:
 def random(self):return random.random()
 def sample(self,p,n):return random.sample(p,n)
def sample_train(B):
 r=GR();return [train_expr(random.choice(TRAIN_T),r) for _ in range(B)]
REWRITES=['distribute3','mul_assoc3','distribute4','product_expand4','product_expand5']
def rewrite_pair(kind,L):
 if kind=='distribute3':
  a,b,c=L;return mul(a,add(b,c)),add(mul(a,b),mul(a,c))
 if kind=='mul_assoc3':
  a,b,c=L;return mul(mul(a,b),c),mul(a,mul(b,c))
 if kind=='distribute4':
  a,b,c,d=L;return add(mul(add(a,b),c),d),add(add(mul(a,c),mul(b,c)),d)
 if kind=='product_expand4':
  a,b,c,d=L;return mul(add(a,b),add(c,d)),adds([mul(a,c),mul(a,d),mul(b,c),mul(b,d)])
 if kind=='product_expand5':
  a,b,c,d,e=L;return add(mul(add(a,b),add(c,d)),e),adds([mul(a,c),mul(a,d),mul(b,c),mul(b,d),e])
 raise ValueError(kind)
X=torch.linspace(-1,1,256).view(-1,1)
def bank(kind,n,seed):
 r=random.Random(seed);pairs=[];meta=[];attempt=0;nleaf={'distribute3':3,'mul_assoc3':3,'distribute4':4,'product_expand4':4,'product_expand5':5}[kind]
 while len(pairs)<n:
  attempt+=1;ids=r.sample(TRAIN_IDS,nleaf);L=[leaf(i,rc(r)) for i in ids];A,B=rewrite_pair(kind,L);ya,yb=ev(A,X),ev(B,X);energy=float((ya**2).mean());eq=float(((ya-yb)**2).mean())
  if energy<E_MIN:continue
  pid=len(pairs);pairs.append((A,B));meta.append({'pair_id':pid,'target_energy':energy,'target_equiv_mse':eq})
 expr=[z for p in pairs for z in p];xq=X.view(1,256,1).repeat(len(expr),1,1);yq=torch.stack([ev(e,xq[i]) for i,e in enumerate(expr)])
 return {'exprs':expr,'xq':xq,'yq':yq,'meta':pd.DataFrame(meta),'attempts':attempt}
def tf(th,x):
 B=th.shape[0];i=0;W1=th[:,i:i+H].view(B,H,1);i+=H;b1=th[:,i:i+H].view(B,1,H);i+=H;W2=th[:,i:i+H].view(B,1,H);i+=H;b2=th[:,i:i+1].view(B,1,1);return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2
def grad(th,x,y):
 B=th.shape[0];i=0;W1=th[:,i:i+H];i+=H;b1=th[:,i:i+H];i+=H;W2=th[:,i:i+H];i+=H;b2=th[:,i:i+1];xx=x.squeeze(-1);yy=y.squeeze(-1);z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:];a=torch.tanh(z);pred=(a*W2[:,None,:]).sum(2)+b2;d=2*(pred-yy)/xx.shape[1];gb2=d.sum(1,keepdim=True);gW2=(d[:,:,None]*a).sum(1);gz=d[:,:,None]*W2[:,None,:]*(1-a*a);gb1=gz.sum(1);gW1=(gz*xx[:,:,None]).sum(1);return torch.cat([gW1,gb1,gW2,gb2],1)
class LE(torch.nn.Module):
 def __init__(self):super().__init__();self.register_buffer('d',DESC.clone());self.l1=torch.nn.Linear(4,E);self.l2=torch.nn.Linear(E,E)
 def emb(self,i):return torch.tanh(self.l2(torch.tanh(self.l1(self.d[i]))))
class Tree(torch.nn.Module):
 def __init__(self):
  super().__init__();self.le=LE();self.m1=torch.nn.Linear(3*E,E);self.m2=torch.nn.Linear(E,E);self.base=torch.nn.Parameter(torch.randn(P)*.12);self.rho=torch.nn.Linear(E,P);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
 def enc(self,e):
  if e[0]=='leaf':return e[2]*self.le.emb(e[1])
  a=self.enc(e[1]);b=self.enc(e[2])
  if e[0]=='add':return a+b
  return torch.tanh(self.m2(torch.tanh(self.m1(torch.cat([a+b,a*b,torch.abs(a-b)])))))
 def forward(self,es):return torch.stack([self.base+self.rho(torch.tanh(self.enc(e))) for e in es])
def sample_xy(es,n):
 B=len(es);x=2*torch.rand(B,n,1)-1;y=torch.stack([ev(es[i],x[i]) for i in range(B)]);return x,y
def train_direct():
 torch.manual_seed(700000+SEED);np.random.seed(700000+SEED);random.seed(700000+SEED);m=Tree();o=torch.optim.Adam(m.parameters(),lr=1e-3)
 for _ in range(DIRECT_STEPS):
  es=sample_train(BATCH);x,y=sample_xy(es,QUERY_N);th=m(es);g=grad(th,x,y).detach()/BATCH;o.zero_grad();torch.autograd.backward(th,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);o.step()
 return m
banks={n:bank(n,FINAL_N,20267000+i) for i,n in enumerate(REWRITES)}
t=time.time();model=train_direct();train_s=time.time()-t;rows=[]
with torch.no_grad():
 for name,b in banks.items():
  th=model(b['exprs']);pred=tf(th,b['xq'])
  for pid in range(FINAL_N):
   ia,ib=2*pid,2*pid+1;ya=b['yq'][ia];yb=b['yq'][ib];pa=pred[ia];pb=pred[ib];ma=float(((pa-ya)**2).mean());mb=float(((pb-yb)**2).mean());dis=float(((pa-pb)**2).mean());wd=float(torch.max(torch.abs(th[ia]-th[ib])));energy=float(b['meta'].loc[pid,'target_energy']);eq=float(b['meta'].loc[pid,'target_equiv_mse'])
   rows.append({'seed':SEED,'rewrite':name,'pair_id':pid,'mse_compact':ma,'mse_rewrite':mb,'prediction_disagreement':dis,'disagreement_over_energy':dis/(energy+EPS),'max_weight_diff':wd,'target_energy':energy,'target_equiv_mse':eq,'log_mse_ratio_rewrite_compact':math.log((mb+EPS)/(ma+EPS))})
df=pd.DataFrame(rows);df.to_csv(OUT/f'rewrite_seed{SEED:02d}.csv',index=False);pd.DataFrame([{'seed':SEED,'train_seconds':train_s}]).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
print(df.groupby('rewrite')[['mse_compact','mse_rewrite','prediction_disagreement','disagreement_over_energy','max_weight_diff','target_equiv_mse','log_mse_ratio_rewrite_compact']].median().to_string());print('train_s',train_s)
