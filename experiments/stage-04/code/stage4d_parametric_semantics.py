import math,random,time,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage4d_results'));OUT.mkdir(parents=True,exist_ok=True)
SEED=int(os.getenv('SEED','0')); SPLIT=os.getenv('SPLIT','dev'); FAMILIES=[x for x in os.getenv('FAMILIES','Semantic,Fingerprint').split(',') if x]
H=32;P=3*H+1;E=48;META_STEPS=600;DIRECT_STEPS=2400;META_BATCH=8;META_H=10;TRAIN_INNER_LR=.02
SUPPORT_N=16;QUERY_N=48;CALIB_N=120;FINAL_N=120;VAL_STEPS=20;EVAL=[0,1,2,5,10,20,50,100];LR_CAND=[.005,.01,.02,.05,.1];CKE=10;EPS=1e-8
# Parametric primitive family: Gaussian g_{mu,sigma}(x). Train and held identities are disjoint descriptor points.
train_desc=[(-.8,.18),(-.4,.18),(0.,.18),(.4,.18),(.8,.18),(-.8,.30),(-.4,.30),(0.,.30),(.4,.30),(.8,.30)]
dev_desc=[(-.6,.18),(.2,.18),(-.6,.24),(.2,.24)]
conf_desc=[(-.2,.18),(.6,.18),(-.2,.24),(.6,.24)]
extra_desc=[(-1.0,.24),(1.0,.24)]
ALL=train_desc+dev_desc+conf_desc+extra_desc;D=len(ALL);TRAIN_IDS=list(range(len(train_desc)));DEV_IDS=list(range(10,14));CONF_IDS=list(range(14,18));EXTRA_IDS=[18,19]
DESC=torch.tensor([[mu,sig,(sig-.24)/.12,mu*mu] for mu,sig in ALL],dtype=torch.float32)
ANCH=torch.linspace(-1,1,41)
FP=torch.stack([torch.exp(-.5*((ANCH-mu)/sig)**2) for mu,sig in ALL],0)
# normalize fingerprints but append original RMS so amplitude/width information remains
rms=torch.sqrt((FP*FP).mean(1,keepdim=True)+1e-8);FPRINT=torch.cat([FP/rms,rms],1)

def mask(*ids):m=torch.zeros(D);m[list(ids)]=1;return m
TRAIN_MASKS=[mask(i) for i in TRAIN_IDS]
for i in range(4): TRAIN_MASKS += [mask(i,i+1),mask(5+i,6+i)]
for t in [(0,2,5),(1,3,7),(2,4,9),(0,6),(3,8),(1,5,9),(0,4,7)]:TRAIN_MASKS.append(mask(*t))
TRAIN_MASKS=torch.stack(TRAIN_MASKS); CALIB_MASK=mask(1,3,6,8)
if SPLIT=='dev': ids=DEV_IDS
else: ids=CONF_IDS
FINAL_MASKS={
 'held_single_a':mask(ids[0]),
 'held_single_b':mask(ids[1]),
 'held_width_interp':mask(ids[2]),
 'held_pair':mask(ids[0],ids[3]),
 'held_mix_known':mask(ids[1],2,7),
 'held_all4':mask(*ids),
}
if SPLIT=='confirm':
 FINAL_MASKS['descriptor_extrap']=mask(EXTRA_IDS[0],EXTRA_IDS[1])

def primitive_values(x):
 # x [...,1] -> [...,D]
 z=x.squeeze(-1);arr=[]
 for mu,sig in ALL:arr.append(torch.exp(-.5*((z-mu)/sig)**2))
 return torch.stack(arr,-1)
def true_function(c,x):
 ph=primitive_values(x)
 if c.ndim==1:return (ph*c).sum(-1,keepdim=True)
 return (ph*c[:,None,:]).sum(-1,keepdim=True)

def target_forward(th,x):
 if th.ndim==1:
  i=0;W1=th[i:i+H].view(H,1);i+=H;b1=th[i:i+H].view(1,H);i+=H;W2=th[i:i+H].view(1,H);i+=H;b2=th[i:i+1].view(1,1);return torch.tanh(x@W1.T+b1)@W2.T+b2
 B=th.shape[0];i=0;W1=th[:,i:i+H].view(B,H,1);i+=H;b1=th[:,i:i+H].view(B,1,H);i+=H;W2=th[:,i:i+H].view(B,1,H);i+=H;b2=th[:,i:i+1].view(B,1,1);return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2
def manual_grad(th,x,y):
 sq=th.ndim==1;tb=th.view(1,-1) if sq else th;xb=x.view(1,*x.shape) if sq else x;yb=y.view(1,*y.shape) if sq else y
 B=tb.shape[0];i=0;W1=tb[:,i:i+H].view(B,H);i+=H;b1=tb[:,i:i+H].view(B,H);i+=H;W2=tb[:,i:i+H].view(B,H);i+=H;b2=tb[:,i:i+1].view(B,1)
 xx=xb.squeeze(-1);yy=yb.squeeze(-1);z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:];a=torch.tanh(z);pred=(a*W2[:,None,:]).sum(2)+b2;d=2*(pred-yy)/xx.shape[1]
 gb2=d.sum(1,keepdim=True);gW2=(d[:,:,None]*a).sum(1);gz=d[:,:,None]*W2[:,None,:]*(1-a*a);gb1=gz.sum(1);gW1=(gz*xx[:,:,None]).sum(1);g=torch.cat([gW1,gb1,gW2,gb2],1);return g[0] if sq else g

def sample_c(B):
 idx=torch.randint(0,len(TRAIN_MASKS),(B,));m=TRAIN_MASKS[idx];sgn=torch.where(torch.rand(B,D)>.5,1.,-1.);mag=.5+torch.rand(B,D);return m*sgn*mag
def sample_xy(c,n):B=c.shape[0];x=2*torch.rand(B,n,1)-1;return x,true_function(c,x)

class TokenHyper(torch.nn.Module):
 def __init__(self,desc):
  super().__init__();self.register_buffer('desc',desc.clone());din=desc.shape[1];self.base=torch.nn.Parameter(torch.randn(P)*.12);self.phi1=torch.nn.Linear(din,E);self.phi2=torch.nn.Linear(E,E);self.pre_bias=torch.nn.Parameter(torch.zeros(E));self.rho=torch.nn.Linear(E,P);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
 def emb(self):return torch.tanh(self.phi2(torch.tanh(self.phi1(self.desc))))
 def forward(self,c):return self.base+self.rho(torch.tanh(c@self.emb()+self.pre_bias))
class SemanticHyper(TokenHyper):
 def __init__(self):super().__init__(DESC)
class FingerHyper(TokenHyper):
 def __init__(self):super().__init__(FPRINT)
class IDHyper(torch.nn.Module):
 def __init__(self):
  super().__init__();self.base=torch.nn.Parameter(torch.randn(P)*.12);self.emb=torch.nn.Parameter(torch.empty(D,E));torch.nn.init.uniform_(self.emb,-1/math.sqrt(D),1/math.sqrt(D));self.pre_bias=torch.nn.Parameter(torch.zeros(E));self.rho=torch.nn.Linear(E,P);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
 def forward(self,c):return self.base+self.rho(torch.tanh(c@self.emb+self.pre_bias))

def train_meta(factory,off):
 torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
 for _ in range(META_STEPS):
  c=sample_c(META_BATCH);t0=m(c);th=t0;sx=2*torch.rand(META_BATCH,META_H,SUPPORT_N,1)-1;cc=c[:,None,:].expand(META_BATCH,META_H,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(META_BATCH,META_H,SUPPORT_N,1)
  for j in range(META_H):th=th-TRAIN_INNER_LR*manual_grad(th,sx[:,j],sy[:,j]).detach()
  xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).detach()/META_BATCH;opt.zero_grad();torch.autograd.backward(t0,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
 return m
def train_direct(factory,off):
 torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3);ck={0:copy.deepcopy(m.state_dict())}
 for s in range(1,DIRECT_STEPS+1):
  c=sample_c(META_BATCH);xq,yq=sample_xy(c,QUERY_N);th=m(c);g=manual_grad(th,xq,yq).detach()/META_BATCH;opt.zero_grad();torch.autograd.backward(th,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
  if s%CKE==0:ck[s]=copy.deepcopy(m.state_dict())
 return m,ck

def bank(maskv,n,seed,q=256,lo=.5,hi=1.5):
 g=torch.Generator().manual_seed(seed);m=maskv.view(1,D).repeat(n,1);sgn=torch.where(torch.rand(n,D,generator=g)>.5,1.,-1.);mag=lo+(hi-lo)*torch.rand(n,D,generator=g);c=m*sgn*mag;sx=2*torch.rand(n,100,SUPPORT_N,1,generator=g)-1;cc=c[:,None,:].expand(n,100,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(n,100,SUPPORT_N,1);xq=torch.linspace(-1,1,q).view(1,q,1).repeat(n,1,1);yq=true_function(c,xq);return {'c':c,'sx':sx,'sy':sy,'xq':xq,'yq':yq}
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
def choose_lr(th,b):return min((val(th,b,lr),lr) for lr in LR_CAND)[1]
def select_ck(ck,meta,b,factory):
 with torch.no_grad():tar=zero(meta(b['c']),b)
 tmp=factory();best=None
 for s,st in ck.items():
  tmp.load_state_dict(st)
  with torch.no_grad():med=zero(tmp(b['c']),b)
  rec=(abs(math.log(med+EPS)-math.log(tar+EPS)),-s,s,med)
  if best is None or rec<best:best=rec
 _,_,s,med=best;tmp.load_state_dict(ck[s]);return copy.deepcopy(tmp),s,tar,med
def metric(c):
 m0=c[0].numpy();rel=np.stack([np.log((c[k].numpy()+EPS)/(m0+EPS)) for k in EVAL if k>0],1).mean(1);return float(np.median(rel)),float(np.median(m0)),float(np.median(c[100].numpy()))
def add(rows,sm,method,c,lr,mn):
 r,z,h=metric(c);sm.append({'seed':SEED,'split':SPLIT,'method':method,'mask':mn,'selected_lr':lr,'rel_log_auc':r,'zero_median':z,'step100_median':h})
 for k in EVAL:
  for i,v in enumerate(c[k].numpy()):rows.append({'seed':SEED,'split':SPLIT,'method':method,'mask':mn,'adapt_steps':k,'task_index':i,'mse':float(v),'selected_lr':lr})

cal=bank(CALIB_MASK,CALIB_N,20261801,q=128);banks={n:bank(m,FINAL_N,20261820+i,q=256) for i,(n,m) in enumerate(FINAL_MASKS.items())};banks['held_all4_high']=bank(FINAL_MASKS['held_all4'],FINAL_N,20261900,q=256,lo=1.5,hi=2.0)
rows=[];sm=[];matches=[];tim=[];allf={'Semantic':(SemanticHyper,100000),'Fingerprint':(FingerHyper,200000),'ID':(IDHyper,300000)}
for fam in FAMILIES:
 factory,off=allf[fam];t=time.time();direct,ck=train_direct(factory,off);tim.append((fam+'_Direct',time.time()-t));t=time.time();meta=train_meta(factory,off+50000);tim.append((fam+'_Meta',time.time()-t));matched,step,mz,dz=select_ck(ck,meta,cal,factory);matches.append({'seed':SEED,'split':SPLIT,'family':fam,'match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))})
 with torch.no_grad():famcal=[(fam+'_Meta',meta(cal['c'])),(fam+'_Matched',matched(cal['c'])),(fam+'_DirectFinal',direct(cal['c']))]
 lrs={n:choose_lr(th,cal) for n,th in famcal}
 for mn,b in banks.items():
  with torch.no_grad():ths=[meta(b['c']),matched(b['c']),direct(b['c'])]
  for n,th in zip([fam+'_Meta',fam+'_Matched',fam+'_DirectFinal'],ths):add(rows,sm,n,curve(th,b,lrs[n]),lrs[n],mn)
pd.DataFrame(rows).to_csv(OUT/f'task_{SPLIT}_seed{SEED:02d}.csv',index=False);pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_{SPLIT}_seed{SEED:02d}.csv',index=False);pd.DataFrame(matches).to_csv(OUT/f'match_{SPLIT}_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim,columns=['method','wall_seconds']).assign(seed=SEED,split=SPLIT).to_csv(OUT/f'timing_{SPLIT}_seed{SEED:02d}.csv',index=False)
print(json.dumps({'seed':SEED,'split':SPLIT,'matches':matches,'metrics':sm[:18]},indent=2))
