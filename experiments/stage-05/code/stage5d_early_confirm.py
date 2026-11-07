import math,random,time,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage5d_confirm')); OUT.mkdir(parents=True,exist_ok=True); SEED=int(os.getenv('SEED','0'))
H=32;P=3*H+1;E=48; META_STEPS=650; DIRECT_STEPS=2600; BATCH=8; META_H=10; TRAIN_INNER_LR=.02
SUPPORT_N=16;QUERY_N=48;CAL_N=100;FINAL_N=int(os.getenv('FINAL_N','60'));VAL_STEPS=20;EVAL=[0,1,2,5,10,20,50,100];LR_CAND=[.005,.01,.02,.05,.1];CKE=10;EPS=1e-8
E_MIN=.02;SEP_MIN=.02
ALL=[(-.8,.18),(-.4,.18),(0.,.18),(.4,.18),(.8,.18),(-.8,.30),(-.4,.30),(0.,.30),(.4,.30),(.8,.30),(-.2,.24),(.2,.24),(.6,.24),(-.6,.24)]
TRAIN_IDS=list(range(10));DESC=torch.tensor([[mu,sig,(sig-.24)/.12,mu*mu] for mu,sig in ALL],dtype=torch.float32)
def leaf(i,c):return ('leaf',int(i),float(c))
def add(a,b):return ('add',a,b)
def mul(a,b):return ('mul',a,b)
def prim(i,x):
 mu,sig=ALL[i];z=x.squeeze(-1);return torch.exp(-.5*((z-mu)/sig)**2).view(*x.shape[:-1],1)
def ev(e,x):
 if e[0]=='leaf':return e[2]*prim(e[1],x)
 a=ev(e[1],x);b=ev(e[2],x);return a+b if e[0]=='add' else a*b
def rc(r):return (1 if r.random()>.5 else -1)*(.55+.65*r.random())
def train_expr(name,r):
 n={'leaf':1,'add2':2,'mul2':2,'add_mul_leaf':3,'mul_add_leaf':3}[name];ids=r.sample(TRAIN_IDS,n);L=[leaf(i,rc(r)) for i in ids]
 if name=='leaf':return L[0]
 if name=='add2':return add(L[0],L[1])
 if name=='mul2':return mul(L[0],L[1])
 if name=='add_mul_leaf':return add(mul(L[0],L[1]),L[2])
 return mul(add(L[0],L[1]),L[2])
TRAIN_T=['leaf','add2','mul2','add_mul_leaf','mul_add_leaf']
class GR:
 def random(self):return random.random()
 def sample(self,p,n):return random.sample(p,n)
def sample_train(B):
 g=GR();return [train_expr(random.choice(TRAIN_T),g) for _ in range(B)]
PAIR_NAMES=['fresh_sum3prod','fresh_scope','fresh_nested','fresh_opmix']
def build_pair(kind,L):
 a,b,c,d,e,f=L
 if kind=='fresh_sum3prod':return add(add(mul(a,b),mul(c,d)),mul(e,f)),add(add(mul(a,c),mul(b,e)),mul(d,f))
 if kind=='fresh_scope':return add(mul(add(a,b),add(c,d)),add(e,f)),mul(add(a,b),add(add(c,d),add(e,f)))
 if kind=='fresh_nested':return add(mul(add(mul(a,b),c),d),add(e,f)),add(mul(add(a,c),e),mul(add(b,d),f))
 if kind=='fresh_opmix':return mul(add(add(a,b),c),add(d,mul(e,f))),add(mul(add(a,d),b),mul(c,add(e,f)))
 raise ValueError(kind)

X=torch.linspace(-1,1,256).view(-1,1)
def paired_bank(kind,n,seed,q=256):
 r=random.Random(seed);pairs=[];md=[];attempt=0
 while len(pairs)<n:
  attempt+=1;ids=r.sample(TRAIN_IDS,6);cs=[rc(r) for _ in range(6)];L=[leaf(i,c) for i,c in zip(ids,cs)];A,B=build_pair(kind,L);ya,yb=ev(A,X),ev(B,X);ea=float((ya**2).mean());eb=float((yb**2).mean());sep=float(((ya-yb)**2).mean())
  if min(ea,eb)<E_MIN or sep<SEP_MIN:continue
  pid=len(pairs);pairs.append((A,B));md.extend([{'pair_id':pid,'variant':'A','target_energy':ea,'target_pair_sep':sep},{'pair_id':pid,'variant':'B','target_energy':eb,'target_pair_sep':sep}])
 expr=[e for p in pairs for e in p];N=len(expr);g=torch.Generator().manual_seed(seed+771);sx=2*torch.rand(N,100,SUPPORT_N,1,generator=g)-1;sy=torch.stack([torch.stack([ev(expr[i],sx[i,j]) for j in range(100)],0) for i in range(N)],0);xq=torch.linspace(-1,1,q).view(1,q,1).repeat(N,1,1);yq=torch.stack([ev(expr[i],xq[i]) for i in range(N)],0)
 return {'exprs':expr,'sx':sx,'sy':sy,'xq':xq,'yq':yq,'meta':pd.DataFrame(md),'attempts':attempt}
def cal_bank(n,seed,q=128):
 r=random.Random(seed);es=[train_expr('add_mul_leaf',r) for _ in range(n)];g=torch.Generator().manual_seed(seed+91);sx=2*torch.rand(n,100,SUPPORT_N,1,generator=g)-1;sy=torch.stack([torch.stack([ev(es[i],sx[i,j]) for j in range(100)],0) for i in range(n)],0);xq=torch.linspace(-1,1,q).view(1,q,1).repeat(n,1,1);yq=torch.stack([ev(es[i],xq[i]) for i in range(n)],0);return {'exprs':es,'sx':sx,'sy':sy,'xq':xq,'yq':yq}
def tf(th,x):
 if th.ndim==1:
  i=0;W1=th[i:i+H].view(H,1);i+=H;b1=th[i:i+H].view(1,H);i+=H;W2=th[i:i+H].view(1,H);i+=H;b2=th[i:i+1].view(1,1);return torch.tanh(x@W1.T+b1)@W2.T+b2
 B=th.shape[0];i=0;W1=th[:,i:i+H].view(B,H,1);i+=H;b1=th[:,i:i+H].view(B,1,H);i+=H;W2=th[:,i:i+H].view(B,1,H);i+=H;b2=th[:,i:i+1].view(B,1,1);return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2
def grad(th,x,y):
 sq=th.ndim==1;tb=th[None] if sq else th;xb=x[None] if sq else x;yb=y[None] if sq else y;B=tb.shape[0];i=0;W1=tb[:,i:i+H];i+=H;b1=tb[:,i:i+H];i+=H;W2=tb[:,i:i+H];i+=H;b2=tb[:,i:i+1];xx=xb.squeeze(-1);yy=yb.squeeze(-1);z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:];a=torch.tanh(z);pred=(a*W2[:,None,:]).sum(2)+b2;d=2*(pred-yy)/xx.shape[1];gb2=d.sum(1,keepdim=True);gW2=(d[:,:,None]*a).sum(1);gz=d[:,:,None]*W2[:,None,:]*(1-a*a);gb1=gz.sum(1);gW1=(gz*xx[:,:,None]).sum(1);g=torch.cat([gW1,gb1,gW2,gb2],1);return g[0] if sq else g
class LeafEnc(torch.nn.Module):
 def __init__(self):super().__init__();self.register_buffer('d',DESC.clone());self.l1=torch.nn.Linear(4,E);self.l2=torch.nn.Linear(E,E)
 def emb(self,i):return torch.tanh(self.l2(torch.tanh(self.l1(self.d[i]))))
class Tree(torch.nn.Module):
 def __init__(self):
  super().__init__();self.le=LeafEnc();self.m1=torch.nn.Linear(3*E,E);self.m2=torch.nn.Linear(E,E);self.base=torch.nn.Parameter(torch.randn(P)*.12);self.rho=torch.nn.Linear(E,P);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
 def enc(self,e):
  if e[0]=='leaf':return e[2]*self.le.emb(e[1])
  a=self.enc(e[1]);b=self.enc(e[2])
  if e[0]=='add':return a+b
  return torch.tanh(self.m2(torch.tanh(self.m1(torch.cat([a+b,a*b,torch.abs(a-b)])))))
 def forward(self,es):return torch.stack([self.base+self.rho(torch.tanh(self.enc(e))) for e in es])
def sample_xy(es,n):
 B=len(es);x=2*torch.rand(B,n,1)-1;y=torch.stack([ev(es[i],x[i]) for i in range(B)]);return x,y
def train_meta():
 torch.manual_seed(500000+SEED);np.random.seed(500000+SEED);random.seed(500000+SEED);m=Tree();o=torch.optim.Adam(m.parameters(),lr=1e-3)
 for _ in range(META_STEPS):
  es=sample_train(BATCH);t0=m(es);th=t0
  for j in range(META_H):sx,sy=sample_xy(es,SUPPORT_N);th=th-TRAIN_INNER_LR*grad(th,sx,sy).detach()
  qx,qy=sample_xy(es,QUERY_N);g=grad(th,qx,qy).detach()/BATCH;o.zero_grad();torch.autograd.backward(t0,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);o.step()
 return m
def train_direct():
 torch.manual_seed(510000+SEED);np.random.seed(510000+SEED);random.seed(510000+SEED);m=Tree();o=torch.optim.Adam(m.parameters(),lr=1e-3);ck={0:copy.deepcopy(m.state_dict())}
 for s in range(1,DIRECT_STEPS+1):
  es=sample_train(BATCH);qx,qy=sample_xy(es,QUERY_N);th=m(es);g=grad(th,qx,qy).detach()/BATCH;o.zero_grad();torch.autograd.backward(th,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);o.step();
  if s%CKE==0:ck[s]=copy.deepcopy(m.state_dict())
 return m,ck
def train_fo():
 torch.manual_seed(520000+SEED);random.seed(520000+SEED);th=torch.nn.Parameter(torch.randn(P)*.12);o=torch.optim.Adam([th],lr=1e-3)
 for _ in range(META_STEPS):
  es=sample_train(BATCH);t=th[None].repeat(BATCH,1)
  for j in range(META_H):sx,sy=sample_xy(es,SUPPORT_N);t=t-TRAIN_INNER_LR*grad(t,sx,sy).detach()
  qx,qy=sample_xy(es,QUERY_N);g=grad(t,qx,qy).detach().mean(0);o.zero_grad();th.grad=g;torch.nn.utils.clip_grad_norm_([th],1.0);o.step()
 return th.detach()
def zero(th,b):return float(((tf(th,b['xq'])-b['yq'])**2).mean((1,2)).median())
def val(th,b,lr):
 th=th.detach().clone();a=[]
 for s in range(VAL_STEPS+1):
  a.append(torch.log(((tf(th,b['xq'])-b['yq'])**2).mean((1,2))+EPS));
  if s<VAL_STEPS:th=th-lr*grad(th,b['sx'][:,s],b['sy'][:,s])
 return float(torch.stack(a,1).mean(1).median())
def choose(th,b):return min((val(th,b,l),l) for l in LR_CAND)[1]
def curve(th,b,lr):
 th=th.detach().clone();o={}
 for s in range(101):
  if s in EVAL:o[s]=((tf(th,b['xq'])-b['yq'])**2).mean((1,2)).detach().cpu()
  if s<100:th=th-lr*grad(th,b['sx'][:,s],b['sy'][:,s])
 return o
def match(ck,meta,b):
 with torch.no_grad():tar=zero(meta(b['exprs']),b)
 tmp=Tree();best=None
 for s,st in ck.items():
  tmp.load_state_dict(st)
  with torch.no_grad():m=zero(tmp(b['exprs']),b)
  rec=(abs(math.log(m+EPS)-math.log(tar+EPS)),-s,s,m)
  if best is None or rec<best:best=rec
 _,_,s,m=best;tmp.load_state_dict(ck[s]);return copy.deepcopy(tmp),s,tar,m
def addrows(rows,sm,name,c,lr,tpl,b):
 m0=c[0].numpy();rel=np.mean([np.log((c[k].numpy()+EPS)/(m0+EPS)) for k in EVAL if k>0],axis=0);sm.append({'seed':SEED,'method':name,'template':tpl,'lr':lr,'rel_log_auc':float(np.median(rel)),'zero_median':float(np.median(m0)),'step100_median':float(np.median(c[100].numpy()))});md=b['meta'].reset_index(drop=True)
 for k in EVAL:
  for i,v in enumerate(c[k].numpy()):rows.append({'seed':SEED,'method':name,'template':tpl,'adapt_steps':k,'task_index':i,'mse':float(v),'pair_id':int(md.loc[i,'pair_id']),'variant':md.loc[i,'variant'],'target_energy':float(md.loc[i,'target_energy']),'target_pair_sep':float(md.loc[i,'target_pair_sep']),'lr':lr})
cal=cal_bank(CAL_N,20264001);banks={n:paired_bank(n,FINAL_N,20266100+i) for i,n in enumerate(PAIR_NAMES)}
tim=[];print('START direct',flush=True);t=time.time();direct,ck=train_direct();tim.append(('Direct',time.time()-t));print('DONE direct',tim[-1],flush=True);print('START meta',flush=True);t=time.time();meta=train_meta();tim.append(('Meta',time.time()-t));print('DONE meta',tim[-1],flush=True);print('START match',flush=True);matched,step,mz,dz=match(ck,meta,cal);print('DONE match',step,mz,dz,flush=True)
with torch.no_grad():ct={'Tree_Meta':meta(cal['exprs']),'Tree_Matched':matched(cal['exprs']),'Tree_DirectFinal':direct(cal['exprs'])}
lrs={n:choose(th,cal) for n,th in ct.items()};rows=[];sm=[]
for tpl,b in banks.items():
 print('EVAL',tpl,flush=True)
 with torch.no_grad():ths={'Tree_Meta':meta(b['exprs']),'Tree_Matched':matched(b['exprs']),'Tree_DirectFinal':direct(b['exprs'])}
 for n,th in ths.items():addrows(rows,sm,n,curve(th,b,lrs[n]),lrs[n],tpl,b)
pd.DataFrame(rows).to_csv(OUT/f'task_seed{SEED:02d}.csv',index=False);pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_seed{SEED:02d}.csv',index=False);pd.DataFrame([{'seed':SEED,'match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))}]).to_csv(OUT/f'matching_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim,columns=['method','wall_seconds']).assign(seed=SEED).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False);pd.DataFrame([{'template':n,'attempts':b['attempts'],'accept_rate':FINAL_N/b['attempts']} for n,b in banks.items()]).to_csv(OUT/f'bank_seed{SEED:02d}.csv',index=False)
print(json.dumps({'seed':SEED,'match':{'step':step,'gap':abs(math.log(mz+EPS)-math.log(dz+EPS))},'metrics':sm,'timing':tim},indent=2))
