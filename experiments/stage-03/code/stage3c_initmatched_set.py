import math,random,time,sys,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage3_results')); OUT.mkdir(exist_ok=True,parents=True)
SEED=int(os.getenv('SEED','0'))
D=8; H=32; P=3*H+1
META_STEPS=600; DIRECT_STEPS=2400; META_BATCH=8; META_H=int(os.getenv('META_H','10'))
TRAIN_INNER_LR=float(os.getenv('TRAIN_INNER_LR','0.02')); POOL=os.getenv('POOL','sum'); SUPPORT_N=16; QUERY_N=48
CALIB_N=100; FINAL_N=100; VAL_STEPS=20; EVAL=[0,1,2,5,10,20,50,100]
LR_CAND=[0.01,0.02,0.05,0.10]; CHECKPOINT_EVERY=10; EPS=1e-8
# Parameter-matched-ish generators: Set ~8.9k params; Dense width=82 ~8.9k params.
SET_E=64; DENSE_H=64

def mask(*idx):
    m=torch.zeros(D); m[list(idx)]=1; return m
TRAIN_MASKS=[]
for i in range(D): TRAIN_MASKS.append(mask(i))
for i in range(D): TRAIN_MASKS.append(mask(i,(i+1)%D))
for t in [(0,1,2),(2,3,4),(4,5,6),(6,7,0)]: TRAIN_MASKS.append(mask(*t))
TRAIN_MASKS=torch.stack(TRAIN_MASKS)
CALIB_MASK=mask(0,2,4,6)
# Frozen before confirmatory evaluation; none are train masks and none were used in Stage-2 tests.
FINAL_MASKS={
    'pair_3_7':mask(3,7),
    'triple_0_2_6':mask(0,2,6),
    'cross4_1_4_5_7':mask(1,4,5,7),
    'fiveway_0_2_4_5_6':mask(0,2,4,5,6),
    'sixway_1_2_3_5_6_7':mask(1,2,3,5,6,7),
}

def basis(x):
    z=x.squeeze(-1)
    return torch.stack([
        torch.sin(math.pi*z), torch.cos(math.pi*z),
        torch.sin(2*math.pi*z), torch.cos(2*math.pi*z),
        z, 0.5*(3*z*z-1), 0.5*(5*z**3-3*z), (35*z**4-30*z*z+3)/8
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

class DenseHyper(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Parameter(torch.randn(P)*.12);self.net=torch.nn.Sequential(torch.nn.Linear(D,DENSE_H),torch.nn.Tanh(),torch.nn.Linear(DENSE_H,P));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,c):return self.base+self.net(c)

class SetHyper(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Parameter(torch.randn(P)*.12);self.emb=torch.nn.Parameter(torch.empty(D,SET_E));self.pre_bias=torch.nn.Parameter(torch.empty(SET_E));self.rho=torch.nn.Linear(SET_E,P)
        bound=1/math.sqrt(D);torch.nn.init.uniform_(self.emb,-bound,bound);torch.nn.init.uniform_(self.pre_bias,-bound,bound);torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
    def forward_tokens(self,ids,vals,active):
        e=self.emb[ids];z=(e*vals[...,None]*active[...,None]).sum(1)+self.pre_bias;return self.base+self.rho(torch.tanh(z))
    def forward(self,c):
        B=c.shape[0];ids=torch.arange(D,device=c.device).view(1,D).expand(B,D);active=(c!=0).to(c.dtype);return self.forward_tokens(ids,c,active)

def param_count(m):return sum(p.numel() for p in m.parameters())

def train_meta(factory,seed_offset,inner_lr):
    torch.manual_seed(seed_offset+SEED);np.random.seed(seed_offset+SEED);random.seed(seed_offset+SEED)
    m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for _ in range(META_STEPS):
        c=sample_c(META_BATCH);theta0=m(c);th=theta0
        sx=2*torch.rand(META_BATCH,META_H,SUPPORT_N,1)-1;cc=c[:,None,:].expand(META_BATCH,META_H,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(META_BATCH,META_H,SUPPORT_N,1)
        for j in range(META_H):th=th-inner_lr*manual_grad(th,sx[:,j],sy[:,j]).detach()
        xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).detach()/META_BATCH
        opt.zero_grad();torch.autograd.backward(theta0,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m

def train_direct(factory,seed_offset):
    torch.manual_seed(seed_offset+SEED);np.random.seed(seed_offset+SEED);random.seed(seed_offset+SEED)
    m=factory();opt=torch.optim.Adam(m.parameters(),lr=1e-3);ck={0:copy.deepcopy(m.state_dict())}
    for s in range(1,DIRECT_STEPS+1):
        c=sample_c(META_BATCH);xq,yq=sample_xy(c,QUERY_N);theta=m(c);g=manual_grad(theta,xq,yq).detach()/META_BATCH
        opt.zero_grad();torch.autograd.backward(theta,grad_tensors=g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        if s%CHECKPOINT_EVERY==0:ck[s]=copy.deepcopy(m.state_dict())
    return m,ck

def train_fo(seed_offset,inner_lr):
    torch.manual_seed(seed_offset+SEED);np.random.seed(seed_offset+SEED);random.seed(seed_offset+SEED)
    theta=torch.nn.Parameter(torch.randn(P)*.12);opt=torch.optim.Adam([theta],lr=1e-3)
    for _ in range(META_STEPS):
        c=sample_c(META_BATCH);th=theta.detach().view(1,-1).repeat(META_BATCH,1)
        sx=2*torch.rand(META_BATCH,META_H,SUPPORT_N,1)-1;cc=c[:,None,:].expand(META_BATCH,META_H,D).reshape(-1,D);sy=true_function(cc,sx.reshape(-1,SUPPORT_N,1)).reshape(META_BATCH,META_H,SUPPORT_N,1)
        for j in range(META_H):th=th-inner_lr*manual_grad(th,sx[:,j],sy[:,j])
        xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).mean(0).detach();opt.zero_grad();theta.grad=g.clone();torch.nn.utils.clip_grad_norm_([theta],1.0);opt.step()
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
def choose_lr(th,b):
    r=sorted((val_score(th,b,lr),lr) for lr in LR_CAND);return r[0][1],r

def select_ck(ck,meta,b,factory):
    with torch.no_grad():tar=zero(meta(b['c']),b)
    tmp=factory();best=None
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

def perm_invariance(model,c):
    # Compare canonical token order with one random permutation for each task.
    B=c.shape[0];ids=torch.arange(D).view(1,D).expand(B,D);active=(c!=0).float();base=model.forward_tokens(ids,c,active)
    g=torch.Generator().manual_seed(777000+SEED);perms=torch.stack([torch.randperm(D,generator=g) for _ in range(B)])
    ids2=torch.gather(ids,1,perms);c2=torch.gather(c,1,perms);a2=torch.gather(active,1,perms);alt=model.forward_tokens(ids2,c2,a2)
    return float((base-alt).abs().max()),float((base-alt).abs().mean())

cal=bank(CALIB_MASK,CALIB_N,20261101,q=128)
banks={name:bank(m,FINAL_N,20261120+i,q=256) for i,(name,m) in enumerate(FINAL_MASKS.items())}
# coefficient OOD on a sixth untouched structure, secondary only
banks['coeff_high_triple_0_1_4']=bank(mask(0,1,4),FINAL_N,20261340,q=256,lo=1.5,hi=2.0)
rows=[];sm=[];tim=[];lrrows=[];match=[];tall=time.time()
# independent but fixed random streams for architectures
for label,factory,off in [('Set',SetHyper,100000)]:
    t=time.time();direct,ck=train_direct(factory,off);tim.append((f'{label}_DirectFit_final',time.time()-t,0));print(label,'direct',time.time()-t,flush=True)
    family_inner_lr=0.02
    t=time.time();meta=train_meta(factory,off+50000,family_inner_lr);tim.append((f'{label}_MetaInit_h{META_H}',time.time()-t,META_STEPS*META_BATCH*META_H));print(label,'meta',time.time()-t,flush=True)
    matched,step,mz,dz=select_ck(ck,meta,cal,factory);match.append({'seed':SEED,'family':label,'match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))})
    with torch.no_grad():mcal=meta(cal['c']);dcal=matched(cal['c']);dfcal=direct(cal['c'])
    fam_methods=[(f'{label}_MetaInit_h{META_H}',mcal),(f'{label}_DirectFit_matched_h{META_H}',dcal),(f'{label}_DirectFit_final',dfcal)]
    lrs={}
    for method,cth in fam_methods:
        lr,recs=choose_lr(cth,cal);lrs[method]=lr
        for score,l in recs:lrrows.append({'seed':SEED,'method':method,'lr':l,'score':score,'selected':l==lr})
    if label=='Set':
        mx,ma=perm_invariance(meta,cal['c']);match[-1]['perm_max_abs']=mx;match[-1]['perm_mean_abs']=ma
    for bname,b in banks.items():
        with torch.no_grad():objs=[meta(b['c']),matched(b['c']),direct(b['c'])]
        for method,th in zip([f'{label}_MetaInit_h{META_H}',f'{label}_DirectFit_matched_h{META_H}',f'{label}_DirectFit_final'],objs):add(rows,sm,method,curve(th,b,lrs[method]),lrs[method],bname)
# FO baseline
t=time.time();fo=train_fo(300000,0.02);tim.append((f'FO-MAML_shared_h{META_H}',time.time()-t,META_STEPS*META_BATCH*META_H));print('fo',time.time()-t,flush=True)
fo_cal=fo.view(1,-1).repeat(CALIB_N,1);flr,recs=choose_lr(fo_cal,cal)
for score,l in recs:lrrows.append({'seed':SEED,'method':f'FO-MAML_shared_h{META_H}','lr':l,'score':score,'selected':l==flr})
for bname,b in banks.items():add(rows,sm,f'FO-MAML_shared_h{META_H}',curve(fo.view(1,-1).repeat(FINAL_N,1),b,flr),flr,bname)
pd.DataFrame(rows).to_csv(OUT/f'task_seed{SEED:02d}.csv',index=False);pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_seed{SEED:02d}.csv',index=False);pd.DataFrame(match).to_csv(OUT/f'match_seed{SEED:02d}.csv',index=False);pd.DataFrame(lrrows).to_csv(OUT/f'lr_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim,columns=['method','wall_seconds','inner_gradient_evals']).assign(seed=SEED).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
metaout={'seed':SEED,'wall_seconds':time.time()-tall,'config':{'D':D,'H':H,'P':P,'meta_horizon':META_H,'train_inner_lr':TRAIN_INNER_LR,'set_params':param_count(SetHyper()),'dense_params':param_count(DenseHyper()),'set_representation':'algebra-aware tokens: sum(coefficient * primitive_embedding) then tanh','final_masks':list(FINAL_MASKS),'secondary':'coeff_high_triple_0_1_4','initialization':'Dense-first-layer matched uniform bound 1/sqrt(D)','set_train_inner_lr':0.02,'dense_train_inner_lr':0.02,'fo_train_inner_lr':0.02}};(OUT/f'meta_seed{SEED:02d}.json').write_text(json.dumps(metaout,indent=2));print(json.dumps({'meta':metaout,'match':match,'metrics':sm[:8]},indent=2),flush=True);sys.stdout.flush()
