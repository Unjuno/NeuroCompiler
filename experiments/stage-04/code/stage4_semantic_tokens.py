import math,random,time,sys,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage4_results')); OUT.mkdir(exist_ok=True,parents=True)
SEED=int(os.getenv('SEED','0'))
D=12; H=32; P=3*H+1; E=64
META_STEPS=700; DIRECT_STEPS=2800; META_BATCH=8; META_H=10; TRAIN_INNER_LR=0.02
SUPPORT_N=16; QUERY_N=48; CALIB_N=120; FINAL_N=120; VAL_STEPS=20; EVAL=[0,1,2,5,10,20,50,100]
LR_CAND=[0.005,0.01,0.02,0.05,0.10]; CHECKPOINT_EVERY=10; EPS=1e-8
# 0 sin1,1 cos1,2 sin2,3 cos2,4 sin3(HO),5 cos3(HO),6 P1,7 P2,8 P3,9 P4,10 P5,11 P6(HO)
TRAIN_IDS=[0,1,2,3,6,7,8,9,10]
HELD_IDS=[4,5,11]

def mask(*ids):
    m=torch.zeros(D); m[list(ids)]=1; return m
TRAIN_MASKS=[mask(i) for i in TRAIN_IDS]
for a,b in zip(TRAIN_IDS,TRAIN_IDS[1:]+TRAIN_IDS[:1]): TRAIN_MASKS.append(mask(a,b))
for t in [(0,1,2),(1,2,3),(6,7,8),(7,8,9),(8,9,10),(0,3,8)]: TRAIN_MASKS.append(mask(*t))
TRAIN_MASKS=torch.stack(TRAIN_MASKS)
CALIB_MASK=mask(0,2,7,9)  # known primitives only
FINAL_MASKS={
 'held_sin3':mask(4),
 'held_cos3_plus_sin1':mask(5,0),
 'held_P6_plus_P2':mask(11,7),
 'held_all3':mask(4,5,11),
 'held_mix_known':mask(4,1,8,11),
}
# fixed semantic descriptors: [is_sin,is_cos,is_legendre, normalized order]
desc=[]
for i in range(D):
    if i in [0,2,4]:
        k={0:1,2:2,4:3}[i]; desc.append([1,0,0,k/3])
    elif i in [1,3,5]:
        k={1:1,3:2,5:3}[i]; desc.append([0,1,0,k/3])
    else:
        d=i-5; desc.append([0,0,1,d/6])
DESC=torch.tensor(desc,dtype=torch.float32)

def legendre(z,n):
    if n==0:return torch.ones_like(z)
    if n==1:return z
    p0=torch.ones_like(z); p1=z
    for k in range(2,n+1):
        p=((2*k-1)*z*p1-(k-1)*p0)/k; p0,p1=p1,p
    return p1

def basis(x):
    z=x.squeeze(-1)
    return torch.stack([
        torch.sin(math.pi*z),torch.cos(math.pi*z),
        torch.sin(2*math.pi*z),torch.cos(2*math.pi*z),
        torch.sin(3*math.pi*z),torch.cos(3*math.pi*z),
        legendre(z,1),legendre(z,2),legendre(z,3),legendre(z,4),legendre(z,5),legendre(z,6)
    ],dim=-1)

def true_function(c,x):
    ph=basis(x)
    if c.ndim==1:return (ph*c).sum(-1,keepdim=True)
    return (ph*c[:,None,:]).sum(-1,keepdim=True)

def target_forward(theta,x):
    if theta.ndim==1:
        i=0;W1=theta[i:i+H].view(H,1);i+=H;b1=theta[i:i+H].view(1,H);i+=H;W2=theta[i:i+H].view(1,H);i+=H;b2=theta[i:i+1].view(1,1)
        return torch.tanh(x@W1.T+b1)@W2.T+b2
    B=theta.shape[0];i=0;W1=theta[:,i:i+H].view(B,H,1);i+=H;b1=theta[:,i:i+H].view(B,1,H);i+=H;W2=theta[:,i:i+H].view(B,1,H);i+=H;b2=theta[:,i:i+1].view(B,1,1)
    return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2

def manual_grad(theta,x,y):
    sq=theta.ndim==1
    if sq:tb=theta.view(1,-1);xb=x.view(1,*x.shape);yb=y.view(1,*y.shape)
    else:tb=theta;xb=x;yb=y
    B=tb.shape[0];i=0;W1=tb[:,i:i+H].view(B,H);i+=H;b1=tb[:,i:i+H].view(B,H);i+=H;W2=tb[:,i:i+H].view(B,H);i+=H;b2=tb[:,i:i+1].view(B,1)
    xx=xb.squeeze(-1);yy=yb.squeeze(-1);z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:];a=torch.tanh(z);pred=(a*W2[:,None,:]).sum(2)+b2
    d=2*(pred-yy)/xx.shape[1];gb2=d.sum(1,keepdim=True);gW2=(d[:,:,None]*a).sum(1);gz=d[:,:,None]*W2[:,None,:]*(1-a*a);gb1=gz.sum(1);gW1=(gz*xx[:,:,None]).sum(1)
    g=torch.cat([gW1,gb1,gW2,gb2],1);return g[0] if sq else g

def sample_c(B):
    idx=torch.randint(0,len(TRAIN_MASKS),(B,));m=TRAIN_MASKS[idx];sgn=torch.where(torch.rand(B,D)>.5,1.,-1.);mag=.5+torch.rand(B,D);return m*sgn*mag

def sample_xy(c,n):
    B=c.shape[0];x=2*torch.rand(B,n,1)-1;return x,true_function(c,x)

class SemanticHyper(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('desc',DESC.clone());self.base=torch.nn.Parameter(torch.randn(P)*.12)
        self.phi=torch.nn.Linear(4,E); self.pre_bias=torch.nn.Parameter(torch.zeros(E)); self.rho=torch.nn.Linear(E,P)
        torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
    def primitive_embeddings(self): return torch.tanh(self.phi(self.desc))
    def forward(self,c):
        emb=self.primitive_embeddings(); z=c@emb + self.pre_bias; return self.base+self.rho(torch.tanh(z))

class IDHyper(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Parameter(torch.randn(P)*.12);self.emb=torch.nn.Parameter(torch.empty(D,E));self.pre_bias=torch.nn.Parameter(torch.zeros(E));self.rho=torch.nn.Linear(E,P)
        torch.nn.init.uniform_(self.emb,-1/math.sqrt(D),1/math.sqrt(D));torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
    def forward(self,c): return self.base+self.rho(torch.tanh(c@self.emb+self.pre_bias))

def train_meta(factory,off):
    torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for _ in range(META_STEPS):
        c=sample_c(META_BATCH);theta0=m(c);th=theta0
        sx=2*torch.rand(META_BATCH,META_H,SUPPORT_N,1)-1;cc=c[:,None,:].expand(META_BATCH,META_H,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(META_BATCH,META_H,SUPPORT_N,1)
        for j in range(META_H):th=th-TRAIN_INNER_LR*manual_grad(th,sx[:,j],sy[:,j]).detach()
        xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).detach()/META_BATCH
        opt.zero_grad();torch.autograd.backward(theta0,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m

def train_direct(factory,off):
    torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3);ck={0:copy.deepcopy(m.state_dict())}
    for s in range(1,DIRECT_STEPS+1):
        c=sample_c(META_BATCH);xq,yq=sample_xy(c,QUERY_N);theta=m(c);g=manual_grad(theta,xq,yq).detach()/META_BATCH
        opt.zero_grad();torch.autograd.backward(theta,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        if s%CHECKPOINT_EVERY==0:ck[s]=copy.deepcopy(m.state_dict())
    return m,ck

def train_fo(off):
    torch.manual_seed(off+SEED);np.random.seed(off+SEED);random.seed(off+SEED);theta=torch.nn.Parameter(torch.randn(P)*.12);opt=torch.optim.Adam([theta],lr=1e-3)
    for _ in range(META_STEPS):
        c=sample_c(META_BATCH);th=theta.detach().view(1,-1).repeat(META_BATCH,1)
        sx=2*torch.rand(META_BATCH,META_H,SUPPORT_N,1)-1;cc=c[:,None,:].expand(META_BATCH,META_H,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(META_BATCH,META_H,SUPPORT_N,1)
        for j in range(META_H):th=th-TRAIN_INNER_LR*manual_grad(th,sx[:,j],sy[:,j])
        xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).mean(0).detach();opt.zero_grad();theta.grad=g;torch.nn.utils.clip_grad_norm_([theta],1.0);opt.step()
    return theta.detach().clone()

def bank(mask,n,seed,q=256,lo=.5,hi=1.5):
    g=torch.Generator().manual_seed(seed);m=mask.view(1,D).repeat(n,1);sgn=torch.where(torch.rand(n,D,generator=g)>.5,1.,-1.);mag=lo+(hi-lo)*torch.rand(n,D,generator=g);c=m*sgn*mag
    sx=2*torch.rand(n,100,SUPPORT_N,1,generator=g)-1;cc=c[:,None,:].expand(n,100,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(n,100,SUPPORT_N,1)
    xq=torch.linspace(-1,1,q).view(1,q,1).repeat(n,1,1);yq=true_function(c,xq);return {'c':c,'sx':sx,'sy':sy,'xq':xq,'yq':yq}

def zero(th,b):return float(((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2)).median())
def curve(th0,b,lr):
    th=th0.detach().clone();out={}
    for s in range(101):
        if s in EVAL:out[s]=((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2)).detach().cpu()
        if s<100:th=th-lr*manual_grad(th,b['sx'][:,s],b['sy'][:,s])
    return out

def val_score(th,b,lr):
    th=th.detach().clone();vs=[]
    for s in range(VAL_STEPS+1):
        mse=((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2));vs.append(torch.log(mse+EPS))
        if s<VAL_STEPS:th=th-lr*manual_grad(th,b['sx'][:,s],b['sy'][:,s])
    return float(torch.stack(vs,1).mean(1).median())
def choose_lr(th,b):return min((val_score(th,b,lr),lr) for lr in LR_CAND)[1]

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
def add(rows,sm,method,c,lr,maskname):
    r,z,h=metric(c);sm.append({'seed':SEED,'method':method,'selected_lr':lr,'rel_log_auc':r,'zero_median':z,'step100_median':h,'mask':maskname})
    for k in EVAL:
        for i,v in enumerate(c[k].numpy()):rows.append({'seed':SEED,'method':method,'adapt_steps':k,'task_index':i,'mse':float(v),'mask':maskname,'selected_lr':lr})

cal=bank(CALIB_MASK,CALIB_N,20261401,q=128)
banks={name:bank(m,FINAL_N,20261420+i,q=256) for i,(name,m) in enumerate(FINAL_MASKS.items())}
banks['held_all3_high']=bank(FINAL_MASKS['held_all3'],FINAL_N,20261500,q=256,lo=1.5,hi=2.0)
rows=[];sm=[];matches=[];tim=[];t0=time.time()
# Semantic primary
for family,factory,off in [('Semantic',SemanticHyper,100000),('ID',IDHyper,200000)]:
    t=time.time();direct,ck=train_direct(factory,off);tim.append((family+'_Direct',time.time()-t));
    t=time.time();meta=train_meta(factory,off+50000);tim.append((family+'_Meta',time.time()-t));
    matched,step,mz,dz=select_ck(ck,meta,cal,factory);matches.append({'seed':SEED,'family':family,'match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))})
    with torch.no_grad(): fam=[(family+'_MetaInit_h10',meta(cal['c'])),(family+'_DirectFit_matched_h10',matched(cal['c'])),(family+'_DirectFit_final',direct(cal['c']))]
    lrs={m:choose_lr(th,cal) for m,th in fam}
    for bname,b in banks.items():
        with torch.no_grad(): objs=[meta(b['c']),matched(b['c']),direct(b['c'])]
        for method,th in zip([family+'_MetaInit_h10',family+'_DirectFit_matched_h10',family+'_DirectFit_final'],objs): add(rows,sm,method,curve(th,b,lrs[method]),lrs[method],bname)
# FO
t=time.time();fo=train_fo(300000);tim.append(('FO',time.time()-t));fo_cal=fo.view(1,-1).repeat(CALIB_N,1);flr=choose_lr(fo_cal,cal)
for bname,b in banks.items(): add(rows,sm,'FO-MAML_shared_h10',curve(fo.view(1,-1).repeat(FINAL_N,1),b,flr),flr,bname)
pd.DataFrame(rows).to_csv(OUT/f'task_seed{SEED:02d}.csv',index=False);pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_seed{SEED:02d}.csv',index=False);pd.DataFrame(matches).to_csv(OUT/f'match_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim,columns=['method','wall_seconds']).assign(seed=SEED).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
meta={'seed':SEED,'wall_seconds':time.time()-t0,'train_ids':TRAIN_IDS,'held_ids':HELD_IDS,'final_masks':list(FINAL_MASKS),'descriptor':'[sin,cos,Legendre,normalized order]','claim':'semantic descriptor extrapolation to unseen primitive identities'};(OUT/f'meta_seed{SEED:02d}.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'meta':meta,'matches':matches,'head':sm[:8]},indent=2))
