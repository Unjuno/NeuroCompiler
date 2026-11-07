import math,random,time,sys,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage4b_results')); OUT.mkdir(exist_ok=True,parents=True)
SEED=int(os.getenv('SEED','0'))
# primitive dictionary: sin/cos k=1..6 and Legendre degree 1..9.
# Fully held from training: sin6, cos6, P9.
SIN_K=list(range(1,7)); COS_K=list(range(1,7)); LEG_D=list(range(1,10))
D=21
SIN_IDS={k:2*(k-1) for k in SIN_K}; COS_IDS={k:2*(k-1)+1 for k in COS_K}; LEG_IDS={d:12+(d-1) for d in LEG_D}
HELD_IDS=[SIN_IDS[6],COS_IDS[6],LEG_IDS[9]]
TRAIN_IDS=[i for i in range(D) if i not in HELD_IDS]
H=40; P=3*H+1; E=72
META_STEPS=int(os.getenv('META_STEPS','900')); DIRECT_STEPS=int(os.getenv('DIRECT_STEPS','3600'))
META_BATCH=8; META_H=10; TRAIN_INNER_LR=0.015
SUPPORT_N=16; QUERY_N=48; CALIB_N=120; FINAL_N=120; VAL_STEPS=20; EVAL=[0,1,2,5,10,20,50,100]
LR_CAND=[0.0025,0.005,0.01,0.02,0.05]; CHECKPOINT_EVERY=10; EPS=1e-8
FINGER_N=33

def mask(*ids):
    m=torch.zeros(D); m[list(ids)]=1; return m

def legendre(z,n):
    if n==0:return torch.ones_like(z)
    if n==1:return z
    p0=torch.ones_like(z); p1=z
    for k in range(2,n+1):
        p=((2*k-1)*z*p1-(k-1)*p0)/k; p0,p1=p1,p
    return p1

def basis(x):
    z=x.squeeze(-1); arr=[]
    for k in range(1,7): arr += [torch.sin(k*math.pi*z),torch.cos(k*math.pi*z)]
    for d in range(1,10): arr.append(legendre(z,d))
    return torch.stack(arr,dim=-1)

def true_function(c,x):
    ph=basis(x)
    if c.ndim==1:return (ph*c).sum(-1,keepdim=True)
    return (ph*c[:,None,:]).sum(-1,keepdim=True)

# Training structures use only known primitives.
TRAIN_MASKS=[mask(i) for i in TRAIN_IDS]
# within-family adjacent pairs
for k in range(1,5):
    TRAIN_MASKS += [mask(SIN_IDS[k],SIN_IDS[k+1]), mask(COS_IDS[k],COS_IDS[k+1])]
for d in range(1,8): TRAIN_MASKS.append(mask(LEG_IDS[d],LEG_IDS[d+1]))
# cross-family pairs/triples and moderate-cardinality mixtures
for t in [
    (SIN_IDS[1],COS_IDS[2]),(SIN_IDS[3],LEG_IDS[2]),(COS_IDS[4],LEG_IDS[5]),
    (SIN_IDS[1],COS_IDS[1],LEG_IDS[1]),(SIN_IDS[2],COS_IDS[3],LEG_IDS[4]),
    (SIN_IDS[4],COS_IDS[5],LEG_IDS[7]),(SIN_IDS[5],LEG_IDS[2],LEG_IDS[8]),
    (SIN_IDS[1],SIN_IDS[3],COS_IDS[2],LEG_IDS[6]),
]: TRAIN_MASKS.append(mask(*t))
TRAIN_MASKS=torch.stack(TRAIN_MASKS)
CALIB_MASK=mask(SIN_IDS[2],COS_IDS[4],LEG_IDS[3],LEG_IDS[7])
# Fresh final structures containing identities never active in training.
FINAL_MASKS={
 'held_sin6':mask(SIN_IDS[6]),
 'held_cos6_plus_sin2':mask(COS_IDS[6],SIN_IDS[2]),
 'held_P9_plus_P3':mask(LEG_IDS[9],LEG_IDS[3]),
 'held_sin6_P9_mix':mask(SIN_IDS[6],LEG_IDS[9],COS_IDS[2]),
 'held_all3':mask(SIN_IDS[6],COS_IDS[6],LEG_IDS[9]),
 'held_mix_known':mask(SIN_IDS[6],COS_IDS[3],LEG_IDS[5],LEG_IDS[9]),
}

# Semantic descriptor and functional fingerprint for each primitive.
sem=[]
for i in range(D):
    if i<12:
        k=i//2+1; sem.append([1.0 if i%2==0 else 0.0,1.0 if i%2 else 0.0,0.0,k/6.0])
    else:
        d=i-11; sem.append([0.0,0.0,1.0,d/9.0])
SEM=torch.tensor(sem,dtype=torch.float32)
# Chebyshev-Lobatto anchors help resolve oscillatory structure near boundaries.
j=torch.arange(FINGER_N,dtype=torch.float32)
ANCHORS=torch.cos(math.pi*j/(FINGER_N-1)).flip(0).view(-1,1)
FP=basis(ANCHORS).T.contiguous()  # [D, FINGER_N]
# Per-primitive RMS normalization, preserving sign/shape; append RMS as scale feature.
rms=torch.sqrt((FP*FP).mean(1,keepdim=True)+1e-8)
FPN=FP/rms
FPRINT=torch.cat([FPN,rms],1)

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

class TokenHyper(torch.nn.Module):
    def __init__(self,desc):
        super().__init__(); self.register_buffer('desc',desc.clone()); din=desc.shape[1]
        self.base=torch.nn.Parameter(torch.randn(P)*.12)
        self.phi1=torch.nn.Linear(din,E); self.phi2=torch.nn.Linear(E,E)
        self.pre_bias=torch.nn.Parameter(torch.zeros(E)); self.rho=torch.nn.Linear(E,P)
        torch.nn.init.zeros_(self.rho.weight);torch.nn.init.zeros_(self.rho.bias)
    def primitive_embeddings(self): return torch.tanh(self.phi2(torch.tanh(self.phi1(self.desc))))
    def forward(self,c):
        emb=self.primitive_embeddings(); z=c@emb+self.pre_bias; return self.base+self.rho(torch.tanh(z))
class FingerHyper(TokenHyper):
    def __init__(self): super().__init__(FPRINT)
class SemanticHyper(TokenHyper):
    def __init__(self): super().__init__(SEM)
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
        for jj in range(META_H):th=th-TRAIN_INNER_LR*manual_grad(th,sx[:,jj],sy[:,jj]).detach()
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
        for jj in range(META_H):th=th-TRAIN_INNER_LR*manual_grad(th,sx[:,jj],sy[:,jj])
        xq,yq=sample_xy(c,QUERY_N);g=manual_grad(th,xq,yq).mean(0).detach();opt.zero_grad();theta.grad=g;torch.nn.utils.clip_grad_norm_([theta],1.0);opt.step()
    return theta.detach().clone()

def bank(maskv,n,seed,q=256,lo=.5,hi=1.5):
    g=torch.Generator().manual_seed(seed);m=maskv.view(1,D).repeat(n,1);sgn=torch.where(torch.rand(n,D,generator=g)>.5,1.,-1.);mag=lo+(hi-lo)*torch.rand(n,D,generator=g);c=m*sgn*mag
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

cal=bank(CALIB_MASK,CALIB_N,20261601,q=128)
banks={name:bank(m,FINAL_N,20261620+i,q=256) for i,(name,m) in enumerate(FINAL_MASKS.items())}
banks['held_all3_high']=bank(FINAL_MASKS['held_all3'],FINAL_N,20261700,q=256,lo=1.5,hi=2.0)
rows=[];sm=[];matches=[];tim=[];t0=time.time()
all_fams={'Fingerprint':(FingerHyper,100000),'Semantic':(SemanticHyper,200000),'ID':(IDHyper,300000)}
FAMILY_LIST=[x for x in os.getenv('FAMILIES','Fingerprint,Semantic,ID').split(',') if x]
for family in FAMILY_LIST:
    factory,off=all_fams[family]
    t=time.time();direct,ck=train_direct(factory,off);tim.append((family+'_Direct',time.time()-t));
    t=time.time();meta=train_meta(factory,off+50000);tim.append((family+'_Meta',time.time()-t));
    matched,step,mz,dz=select_ck(ck,meta,cal,factory);matches.append({'seed':SEED,'family':family,'match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))})
    with torch.no_grad(): fam=[(family+'_MetaInit_h10',meta(cal['c'])),(family+'_DirectFit_matched_h10',matched(cal['c'])),(family+'_DirectFit_final',direct(cal['c']))]
    lrs={m:choose_lr(th,cal) for m,th in fam}
    for bname,b in banks.items():
        with torch.no_grad(): objs=[meta(b['c']),matched(b['c']),direct(b['c'])]
        for method,th in zip([family+'_MetaInit_h10',family+'_DirectFit_matched_h10',family+'_DirectFit_final'],objs): add(rows,sm,method,curve(th,b,lrs[method]),lrs[method],bname)
# shared FO baseline
if os.getenv('SKIP_FO','0')!='1':
    t=time.time();fo=train_fo(400000);tim.append(('FO',time.time()-t));fo_cal=fo.view(1,-1).repeat(CALIB_N,1);flr=choose_lr(fo_cal,cal)
    for bname,b in banks.items(): add(rows,sm,'FO-MAML_shared_h10',curve(fo.view(1,-1).repeat(FINAL_N,1),b,flr),flr,bname)

pd.DataFrame(rows).to_csv(OUT/f'task_seed{SEED:02d}.csv',index=False);pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_seed{SEED:02d}.csv',index=False);pd.DataFrame(matches).to_csv(OUT/f'match_seed{SEED:02d}.csv',index=False);pd.DataFrame(tim,columns=['method','wall_seconds']).assign(seed=SEED).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
# descriptor geometry diagnostics: nearest cosine similarity of held to training descriptors
def cos(a,b): return float(torch.dot(a,b)/(torch.norm(a)*torch.norm(b)+1e-12))
diag=[]
for name,des in [('semantic',SEM),('fingerprint',FPRINT)]:
    for hid in HELD_IDS:
        vals=[(cos(des[hid],des[jj]),jj) for jj in TRAIN_IDS]; vals.sort(reverse=True)
        diag.append({'descriptor':name,'held_id':hid,'nearest_train_id':vals[0][1],'nearest_cosine':vals[0][0]})
pd.DataFrame(diag).to_csv(OUT/f'descriptor_diag_seed{SEED:02d}.csv',index=False)
meta={'seed':SEED,'wall_seconds':time.time()-t0,'train_ids':TRAIN_IDS,'held_ids':HELD_IDS,'final_masks':list(FINAL_MASKS),'fingerprint_anchors':FINGER_N,'claim':'functional fingerprint extrapolation to unseen primitive identities'};(OUT/f'meta_seed{SEED:02d}.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'meta':meta,'matches':matches,'head':sm[:12],'diag':diag},indent=2))
