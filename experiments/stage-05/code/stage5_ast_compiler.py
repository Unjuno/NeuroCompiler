import math,random,time,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage5_results'));OUT.mkdir(parents=True,exist_ok=True);SEED=int(os.getenv('SEED','0'))
H=32;P=3*H+1;E=48;META_STEPS=int(os.getenv('META_STEPS','650'));DIRECT_STEPS=int(os.getenv('DIRECT_STEPS','2600'));BATCH=int(os.getenv('BATCH','8'));META_H=10;TRAIN_INNER_LR=.02
SUPPORT_N=16;QUERY_N=48;CAL_N=int(os.getenv('CAL_N','100'));FINAL_N=int(os.getenv('FINAL_N','100'));VAL_STEPS=20;EVAL=[0,1,2,5,10,20,50,100];LR_CAND=[.005,.01,.02,.05,.1];CKE=10;EPS=1e-8
# Same smooth semantic primitive family. First 10 train identities; last 4 completely held from training.
ALL=[(-.8,.18),(-.4,.18),(0.,.18),(.4,.18),(.8,.18),(-.8,.30),(-.4,.30),(0.,.30),(.4,.30),(.8,.30),(-.2,.24),(.2,.24),(.6,.24),(-.6,.24)]
D=len(ALL);TRAIN_IDS=list(range(10));HELD_IDS=list(range(10,14));DESC=torch.tensor([[mu,sig,(sig-.24)/.12,mu*mu] for mu,sig in ALL],dtype=torch.float32)

def leaf(i,c):return ('leaf',int(i),float(c))
def add(a,b):return ('add',a,b)
def mul(a,b):return ('mul',a,b)
def desc_expr(e):
 if e[0]=='leaf':return f'{e[2]:+.2f}*g{e[1]}'
 return f'({desc_expr(e[1])}{"+" if e[0]=="add" else "*"}{desc_expr(e[2])})'

def prim(i,x):
 mu,sig=ALL[i];z=x.squeeze(-1);return torch.exp(-.5*((z-mu)/sig)**2).view(*x.shape[:-1],1)
def eval_expr(e,x):
 if e[0]=='leaf':return e[2]*prim(e[1],x)
 a=eval_expr(e[1],x);b=eval_expr(e[2],x);return a+b if e[0]=='add' else a*b

def rc():return (1 if random.random()>.5 else -1)*(.55+.65*random.random())
def sample_ids(n,pool=TRAIN_IDS):return random.sample(pool,n)
def make_template(name,pool=TRAIN_IDS):
 if name=='leaf':
  i=sample_ids(1,pool)[0];return leaf(i,rc())
 if name in ['add2','mul2']:
  i,j=sample_ids(2,pool);return (add if name=='add2' else mul)(leaf(i,rc()),leaf(j,rc()))
 if name=='add_mul_leaf':
  i,j,k=sample_ids(3,pool);return add(mul(leaf(i,rc()),leaf(j,rc())),leaf(k,rc()))
 if name=='mul_add_leaf':
  i,j,k=sample_ids(3,pool);return mul(add(leaf(i,rc()),leaf(j,rc())),leaf(k,rc()))
 if name=='sum_products':
  i,j,k,l=sample_ids(4,pool);return add(mul(leaf(i,rc()),leaf(j,rc())),mul(leaf(k,rc()),leaf(l,rc())))
 if name=='product_sums':
  i,j,k,l=sample_ids(4,pool);return mul(add(leaf(i,rc()),leaf(j,rc())),add(leaf(k,rc()),leaf(l,rc())))
 if name=='triple_product':
  i,j,k=sample_ids(3,pool);return mul(mul(leaf(i,rc()),leaf(j,rc())),leaf(k,rc()))
 if name=='nested_mix':
  i,j,k,l=sample_ids(4,pool);return add(mul(add(leaf(i,rc()),leaf(j,rc())),leaf(k,rc())),leaf(l,rc()))
 if name=='deep_mix':
  i,j,k,l,m=sample_ids(5,pool);return mul(add(mul(leaf(i,rc()),leaf(j,rc())),leaf(k,rc())),add(leaf(l,rc()),leaf(m,rc())))
 raise ValueError(name)
TRAIN_TEMPLATES=['leaf','add2','mul2','add_mul_leaf','mul_add_leaf']
FINAL_TEMPLATES=['sum_products','product_sums','triple_product','nested_mix','deep_mix']

def sample_train_exprs(B):return [make_template(random.choice(TRAIN_TEMPLATES)) for _ in range(B)]

def target_forward(th,x):
 if th.ndim==1:
  i=0;W1=th[i:i+H].view(H,1);i+=H;b1=th[i:i+H].view(1,H);i+=H;W2=th[i:i+H].view(1,H);i+=H;b2=th[i:i+1].view(1,1);return torch.tanh(x@W1.T+b1)@W2.T+b2
 B=th.shape[0];i=0;W1=th[:,i:i+H].view(B,H,1);i+=H;b1=th[:,i:i+H].view(B,1,H);i+=H;W2=th[:,i:i+H].view(B,1,H);i+=H;b2=th[:,i:i+1].view(B,1,1);return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2
def manual_grad(th,x,y):
 sq=th.ndim==1;tb=th.view(1,-1) if sq else th;xb=x.view(1,*x.shape) if sq else x;yb=y.view(1,*y.shape) if sq else y;B=tb.shape[0];i=0;W1=tb[:,i:i+H].view(B,H);i+=H;b1=tb[:,i:i+H].view(B,H);i+=H;W2=tb[:,i:i+H].view(B,H);i+=H;b2=tb[:,i:i+1].view(B,1);xx=xb.squeeze(-1);yy=yb.squeeze(-1);z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:];a=torch.tanh(z);pred=(a*W2[:,None,:]).sum(2)+b2;d=2*(pred-yy)/xx.shape[1];gb2=d.sum(1,keepdim=True);gW2=(d[:,:,None]*a).sum(1);gz=d[:,:,None]*W2[:,None,:]*(1-a*a);gb1=gz.sum(1);gW1=(gz*xx[:,:,None]).sum(1);g=torch.cat([gW1,gb1,gW2,gb2],1);return g[0] if sq else g

class LeafEncoder(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('desc',DESC.clone());self.l1=torch.nn.Linear(4,E);self.l2=torch.nn.Linear(E,E)
 def emb(self,i):return torch.tanh(self.l2(torch.tanh(self.l1(self.desc[i]))))
class TreeHyper(torch.nn.Module):
 def __init__(self):
  super().__init__();self.leafenc=LeafEncoder();self.mul1=torch.nn.Linear(3*E,E);self.mul2=torch.nn.Linear(E,E);self.base=torch.nn.Parameter(torch.randn(P)*.12);self.rho=torch.nn.Linear(E,P);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
 def enc_one(self,e):
  if e[0]=='leaf':return e[2]*self.leafenc.emb(e[1])
  a=self.enc_one(e[1]);b=self.enc_one(e[2])
  if e[0]=='add':return a+b
  z=torch.cat([a+b,a*b,torch.abs(a-b)],0);return torch.tanh(self.mul2(torch.tanh(self.mul1(z))))
 def forward(self,exprs):return torch.stack([self.base+self.rho(torch.tanh(self.enc_one(e))) for e in exprs],0)
class BagHyper(torch.nn.Module):
 def __init__(self):
  super().__init__();self.leafenc=LeafEncoder();self.base=torch.nn.Parameter(torch.randn(P)*.12);self.rho=torch.nn.Linear(E,P);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
 def enc_one(self,e):
  if e[0]=='leaf':return e[2]*self.leafenc.emb(e[1])
  return self.enc_one(e[1])+self.enc_one(e[2]) # deliberately ignores + vs × and tree structure
 def forward(self,exprs):return torch.stack([self.base+self.rho(torch.tanh(self.enc_one(e))) for e in exprs],0)

def sample_xy(exprs,n):
 B=len(exprs);x=2*torch.rand(B,n,1)-1;y=torch.stack([eval_expr(exprs[i],x[i]) for i in range(B)],0);return x,y

def train_meta(factory,off):
 torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
 for _ in range(META_STEPS):
  es=sample_train_exprs(BATCH);t0=m(es);th=t0
  for j in range(META_H):
   sx,sy=sample_xy(es,SUPPORT_N);th=th-TRAIN_INNER_LR*manual_grad(th,sx,sy).detach()
  qx,qy=sample_xy(es,QUERY_N);g=manual_grad(th,qx,qy).detach()/BATCH;opt.zero_grad();torch.autograd.backward(t0,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
 return m
def train_direct(factory,off):
 torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3);ck={0:copy.deepcopy(m.state_dict())}
 for s in range(1,DIRECT_STEPS+1):
  es=sample_train_exprs(BATCH);qx,qy=sample_xy(es,QUERY_N);th=m(es);g=manual_grad(th,qx,qy).detach()/BATCH;opt.zero_grad();torch.autograd.backward(th,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();
  if s%CKE==0:ck[s]=copy.deepcopy(m.state_dict())
 return m,ck

def make_bank(template,n,seed,pool=TRAIN_IDS,q=256):
 st=random.getstate();random.seed(seed);es=[make_template(template,pool) for _ in range(n)];random.setstate(st);g=torch.Generator().manual_seed(seed+91);sx=2*torch.rand(n,100,SUPPORT_N,1,generator=g)-1;sy=torch.stack([torch.stack([eval_expr(es[i],sx[i,j]) for j in range(100)],0) for i in range(n)],0);xq=torch.linspace(-1,1,q).view(1,q,1).repeat(n,1,1);yq=torch.stack([eval_expr(es[i],xq[i]) for i in range(n)],0);return {'exprs':es,'sx':sx,'sy':sy,'xq':xq,'yq':yq}
def zero(th,b):return float(((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2)).median())
def curve(th,b,lr):
 th=th.detach().clone();o={}
 for s in range(101):
  if s in EVAL:o[s]=((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2)).detach().cpu()
  if s<100:th=th-lr*manual_grad(th,b['sx'][:,s],b['sy'][:,s])
 return o
def val(th,b,lr):
 th=th.detach().clone();a=[]
 for s in range(VAL_STEPS+1):
  a.append(torch.log(((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2))+EPS))
  if s<VAL_STEPS:th=th-lr*manual_grad(th,b['sx'][:,s],b['sy'][:,s])
 return float(torch.stack(a,1).mean(1).median())
def choose_lr(th,b):return min((val(th,b,l),l) for l in LR_CAND)[1]
def select_ck(ck,meta,b,factory):
 with torch.no_grad():tar=zero(meta(b['exprs']),b)
 tmp=factory();best=None
 for s,st in ck.items():
  tmp.load_state_dict(st)
  with torch.no_grad():med=zero(tmp(b['exprs']),b)
  rec=(abs(math.log(med+EPS)-math.log(tar+EPS)),-s,s,med)
  if best is None or rec<best:best=rec
 _,_,s,med=best;tmp.load_state_dict(ck[s]);return copy.deepcopy(tmp),s,tar,med
def metric(c):
 m0=c[0].numpy();rel=np.stack([np.log((c[k].numpy()+EPS)/(m0+EPS)) for k in EVAL if k>0],1).mean(1);return float(np.median(rel)),float(np.median(m0)),float(np.median(c[100].numpy()))
def addrow(rows,sm,meth,c,lr,name):
 r,z,h=metric(c);sm.append({'seed':SEED,'method':meth,'template':name,'selected_lr':lr,'rel_log_auc':r,'zero_median':z,'step100_median':h})
 for k in EVAL:
  for i,v in enumerate(c[k].numpy()):rows.append({'seed':SEED,'method':meth,'template':name,'adapt_steps':k,'task_index':i,'mse':float(v),'selected_lr':lr})

# Calibration uses a TRAINED template only; final structural templates are untouched.
cal=make_bank('add_mul_leaf',CAL_N,20262001,TRAIN_IDS,q=128)
banks={n:make_bank(n,FINAL_N,20262020+i,TRAIN_IDS) for i,n in enumerate(FINAL_TEMPLATES)}
# secondary: new structure + unseen primitive identities
banks['nested_mix_unseen_leaves']=make_bank('nested_mix',FINAL_N,20262100,HELD_IDS)
rows=[];sm=[];matches=[];tim=[]
for fam,factory,off in [('Tree',TreeHyper,100000),('Bag',BagHyper,200000)]:
 t=time.time();direct,ck=train_direct(factory,off);tim.append((fam+'_Direct',time.time()-t));t=time.time();meta=train_meta(factory,off+50000);tim.append((fam+'_Meta',time.time()-t));matched,step,mz,dz=select_ck(ck,meta,cal,factory);matches.append({'seed':SEED,'family':fam,'match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))})
 with torch.no_grad():cc=[(fam+'_Meta',meta(cal['exprs'])),(fam+'_Matched',matched(cal['exprs'])),(fam+'_DirectFinal',direct(cal['exprs']))]
 lrs={n:choose_lr(th,cal) for n,th in cc}
 for name,b in banks.items():
  with torch.no_grad():ths=[meta(b['exprs']),matched(b['exprs']),direct(b['exprs'])]
  for n,th in zip([fam+'_Meta',fam+'_Matched',fam+'_DirectFinal'],ths):addrow(rows,sm,n,curve(th,b,lrs[n]),lrs[n],name)
pd.DataFrame(rows).to_csv(OUT/f'task_seed{SEED:02d}.csv',index=False);pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_seed{SEED:02d}.csv',index=False);pd.DataFrame(matches).to_csv(OUT/f'matching_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim,columns=['method','wall_seconds']).assign(seed=SEED).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
print(json.dumps({'seed':SEED,'matches':matches,'metrics':sm},indent=2))
