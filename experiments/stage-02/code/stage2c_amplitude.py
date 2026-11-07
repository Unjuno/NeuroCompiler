import math,random,time,platform,sys,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage2c_results')); OUT.mkdir(exist_ok=True,parents=True)
SEED=int(os.getenv('SEED','0'))
D=8; H=32; P=3*H+1; GH=64
META_STEPS=600; DIRECT_STEPS=2400; META_BATCH=8; META_H=10
TRAIN_INNER_LR=0.02; SUPPORT_N=16; QUERY_N=48
CALIB_N=100; FINAL_N=100; VAL_STEPS=20; EVAL=[0,1,2,5,10,20,50,100]
LR_CAND=[0.01,0.02,0.05,0.10]; CHECKPOINT_EVERY=10; EPS=1e-8

def mask(*idx):
    m=torch.zeros(D); m[list(idx)]=1; return m
TRAIN_MASKS=[]
for i in range(D): TRAIN_MASKS.append(mask(i))
for i in range(D): TRAIN_MASKS.append(mask(i,(i+1)%D))
for t in [(0,1,2),(2,3,4),(4,5,6),(6,7,0)]: TRAIN_MASKS.append(mask(*t))
TRAIN_MASKS=torch.stack(TRAIN_MASKS)
CALIB_MASK=mask(0,2,4,6) # alternating odd-position primitives
FINAL_MASK=mask(0,2,5,7) # confirmatory nonlocal cross-4, untouched during pilot
ALL8_MASK=torch.ones(D)

def basis(x):
    # x [...,1] -> [...,8], bounded primitive family
    z=x.squeeze(-1)
    return torch.stack([
        torch.sin(math.pi*z), torch.cos(math.pi*z),
        torch.sin(2*math.pi*z), torch.cos(2*math.pi*z),
        z, 0.5*(3*z*z-1), 0.5*(5*z**3-3*z), (35*z**4-30*z*z+3)/8
    ],dim=-1)

def true_function(c,x):
    ph=basis(x)
    if c.ndim==1: return (ph*c).sum(-1,keepdim=True)
    return (ph*c[:,None,:]).sum(-1,keepdim=True)

def target_forward(theta,x):
    if theta.ndim==1:
        i=0; W1=theta[i:i+H].view(H,1);i+=H; b1=theta[i:i+H].view(1,H);i+=H; W2=theta[i:i+H].view(1,H);i+=H; b2=theta[i:i+1].view(1,1)
        return torch.tanh(x@W1.T+b1)@W2.T+b2
    B=theta.shape[0];i=0;W1=theta[:,i:i+H].view(B,H,1);i+=H;b1=theta[:,i:i+H].view(B,1,H);i+=H;W2=theta[:,i:i+H].view(B,1,H);i+=H;b2=theta[:,i:i+1].view(B,1,1)
    return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2

def manual_grad(theta,x,y):
    sq=theta.ndim==1
    if sq: tb=theta.view(1,-1); xb=x.view(1,*x.shape); yb=y.view(1,*y.shape)
    else: tb=theta;xb=x;yb=y
    B=tb.shape[0];i=0;W1=tb[:,i:i+H].view(B,H);i+=H;b1=tb[:,i:i+H].view(B,H);i+=H;W2=tb[:,i:i+H].view(B,H);i+=H;b2=tb[:,i:i+1].view(B,1)
    xx=xb.squeeze(-1);yy=yb.squeeze(-1);z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:];a=torch.tanh(z);pred=(a*W2[:,None,:]).sum(2)+b2
    d=2*(pred-yy)/xx.shape[1];gb2=d.sum(1,keepdim=True);gW2=(d[:,:,None]*a).sum(1);gz=d[:,:,None]*W2[:,None,:]*(1-a*a);gb1=gz.sum(1);gW1=(gz*xx[:,:,None]).sum(1)
    g=torch.cat([gW1,gb1,gW2,gb2],1);return g[0] if sq else g

def sample_c(B):
    idx=torch.randint(0,len(TRAIN_MASKS),(B,));m=TRAIN_MASKS[idx];sgn=torch.where(torch.rand(B,D)>.5,1.,-1.);mag=.5+torch.rand(B,D);return m*sgn*mag

def sample_xy(c,n):
    B=c.shape[0];x=2*torch.rand(B,n,1)-1;return x,true_function(c,x)

class Hyper(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Parameter(torch.randn(P)*.12);self.net=torch.nn.Sequential(torch.nn.Linear(D,GH),torch.nn.Tanh(),torch.nn.Linear(GH,P));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,c):return self.base+self.net(c)

def train_meta():
    m=Hyper();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for _ in range(META_STEPS):
        c=sample_c(META_BATCH);theta0=m(c);th=theta0
        sx=2*torch.rand(META_BATCH,META_H,SUPPORT_N,1)-1
        cc=c[:,None,:].expand(META_BATCH,META_H,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(META_BATCH,META_H,SUPPORT_N,1)
        for j in range(META_H):th=th-TRAIN_INNER_LR*manual_grad(th,sx[:,j],sy[:,j]).detach()
        xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).detach()/META_BATCH
        opt.zero_grad();torch.autograd.backward(theta0,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m

def train_direct():
    m=Hyper();opt=torch.optim.Adam(m.parameters(),lr=1e-3);ck={0:copy.deepcopy(m.state_dict())}
    for s in range(1,DIRECT_STEPS+1):
        c=sample_c(META_BATCH);xq,yq=sample_xy(c,QUERY_N);theta=m(c);g=manual_grad(theta,xq,yq).detach()/META_BATCH
        opt.zero_grad();torch.autograd.backward(theta,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        if s%CHECKPOINT_EVERY==0:ck[s]=copy.deepcopy(m.state_dict())
    return m,ck

def train_fo():
    theta=torch.nn.Parameter(torch.randn(P)*.12);opt=torch.optim.Adam([theta],lr=1e-3)
    for _ in range(META_STEPS):
        c=sample_c(META_BATCH);th=theta.detach().view(1,-1).repeat(META_BATCH,1)
        sx=2*torch.rand(META_BATCH,META_H,SUPPORT_N,1)-1;cc=c[:,None,:].expand(META_BATCH,META_H,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(META_BATCH,META_H,SUPPORT_N,1)
        for j in range(META_H):th=th-TRAIN_INNER_LR*manual_grad(th,sx[:,j],sy[:,j])
        xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).mean(0).detach();opt.zero_grad();theta.grad=g.clone();torch.nn.utils.clip_grad_norm_([theta],1.0);opt.step()
    return theta.detach().clone()

def bank(mask,n,seed,q=256):
    g=torch.Generator().manual_seed(seed);m=mask.view(1,D).repeat(n,1);sgn=torch.where(torch.rand(n,D,generator=g)>.5,1.,-1.);mag=.5+torch.rand(n,D,generator=g);c=m*sgn*mag
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
def choose_lr(th,b):
    r=sorted((val_score(th,b,lr),lr) for lr in LR_CAND);return r[0][1],r

def select_ck(ck,meta,b):
    with torch.no_grad():mt=meta(b['c']);tar=zero(mt,b)
    tmp=Hyper();best=None
    for s,st in ck.items():
        tmp.load_state_dict(st)
        with torch.no_grad():t=tmp(b['c'])
        med=zero(t,b);gap=abs(math.log(med+EPS)-math.log(tar+EPS));rec=(gap,-s,s,med)
        if best is None or rec<best:best=rec
    _,_,s,med=best;tmp.load_state_dict(ck[s]);return copy.deepcopy(tmp),s,tar,med

def metric(c):
    m0=c[0].numpy();rel=np.stack([np.log((c[k].numpy()+EPS)/(m0+EPS)) for k in EVAL if k>0],1).mean(1);return float(np.median(rel)),float(np.median(m0)),float(np.median(c[100].numpy()))
def add(rows,sm,method,c,lr,maskname):
    r,z,h=metric(c);sm.append({'seed':SEED,'method':method,'selected_lr':lr,'rel_log_auc':r,'zero_median':z,'step100_median':h,'mask':maskname})
    for k in EVAL:
        for i,v in enumerate(c[k].numpy()):rows.append({'seed':SEED,'method':method,'adapt_steps':k,'task_index':i,'mse':float(v),'mask':maskname,'selected_lr':lr})

def bank_range(maskv,n,seed,mag_lo,mag_hi,q=256):
    g=torch.Generator().manual_seed(seed)
    m=maskv.view(1,D).repeat(n,1)
    sgn=torch.where(torch.rand(n,D,generator=g)>.5,1.,-1.)
    mag=mag_lo+(mag_hi-mag_lo)*torch.rand(n,D,generator=g)
    c=m*sgn*mag
    sx=2*torch.rand(n,100,SUPPORT_N,1,generator=g)-1
    cc=c[:,None,:].expand(n,100,D).reshape(-1,D)
    sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(n,100,SUPPORT_N,1)
    xq=torch.linspace(-1,1,q).view(1,q,1).repeat(n,1,1)
    yq=true_function(c,xq)
    return {'c':c,'sx':sx,'sy':sy,'xq':xq,'yq':yq}

cal=bank(CALIB_MASK,CALIB_N,20260911,q=128)
AMP_MASK=mask(0,3,6)  # new nonlocal triple, not used in Stage-2a/2b evaluation
banks={
    'amp_low_0.10_0.50': bank_range(AMP_MASK,FINAL_N,20262001,0.10,0.50,q=256),
    'amp_in_0.50_1.50': bank_range(AMP_MASK,FINAL_N,20262002,0.50,1.50,q=256),
    'amp_high_1.50_2.00': bank_range(AMP_MASK,FINAL_N,20262003,1.50,2.00,q=256),
}
torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
rows=[];sm=[];tim=[];lrrows=[];match=[];tall=time.time()
t=time.time();direct,ck=train_direct();tim.append(('DirectFit_final',time.time()-t,0));print('direct',time.time()-t,flush=True)
t=time.time();fo=train_fo();tim.append((f'FO-MAML_shared_h{META_H}',time.time()-t,META_STEPS*META_BATCH*META_H));print('fo',time.time()-t,flush=True)
torch.manual_seed(900000+SEED);np.random.seed(900000+SEED);random.seed(900000+SEED);t=time.time();meta=train_meta();tim.append((f'MetaInit_h{META_H}',time.time()-t,META_STEPS*META_BATCH*META_H));print('meta',time.time()-t,flush=True)
matched,step,mz,dz=select_ck(ck,meta,cal);match=[{'seed':SEED,'match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))}]
with torch.no_grad():
    mcal=meta(cal['c']);dcal=matched(cal['c']);dfcal=direct(cal['c'])
fo_cal=fo.view(1,-1).repeat(CALIB_N,1)
for method,cth in [(f'MetaInit_h{META_H}',mcal),(f'DirectFit_matched_h{META_H}',dcal),('DirectFit_final',dfcal),(f'FO-MAML_shared_h{META_H}',fo_cal)]:
    lr,recs=choose_lr(cth,cal)
    for score,l in recs:lrrows.append({'seed':SEED,'method':method,'lr':l,'score':score,'selected':l==lr})
    if method==f'MetaInit_h{META_H}': mlr=lr
    elif method==f'DirectFit_matched_h{META_H}': dlr=lr
    elif method=='DirectFit_final': dflr=lr
    else: flr=lr
for bname,b in banks.items():
    with torch.no_grad(): objs=(meta(b['c']),matched(b['c']),direct(b['c']),fo.view(1,-1).repeat(FINAL_N,1))
    for method,th,lr in zip([f'MetaInit_h{META_H}',f'DirectFit_matched_h{META_H}','DirectFit_final',f'FO-MAML_shared_h{META_H}'],objs,[mlr,dlr,dflr,flr]):
        add(rows,sm,method,curve(th,b,lr),lr,bname)
pd.DataFrame(rows).to_csv(OUT/f'task_seed{SEED:02d}.csv',index=False);pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_seed{SEED:02d}.csv',index=False);pd.DataFrame(match).to_csv(OUT/f'match_seed{SEED:02d}.csv',index=False);pd.DataFrame(lrrows).to_csv(OUT/f'lr_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim,columns=['method','wall_seconds','inner_gradient_evals']).assign(seed=SEED).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
metaout={'seed':SEED,'wall_seconds':time.time()-tall,'config':{'D':D,'meta_horizon':META_H,'train_inner_lr':TRAIN_INNER_LR,'amp_mask':'0,3,6','ranges':['0.10-0.50','0.50-1.50','1.50-2.00']}};(OUT/f'meta_seed{SEED:02d}.json').write_text(json.dumps(metaout,indent=2));print(json.dumps({'meta':metaout,'metrics':sm,'match':match},indent=2),flush=True);sys.stdout.flush();os._exit(0)
