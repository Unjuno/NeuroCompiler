import math,random,time,copy,os,json
from pathlib import Path
import numpy as np,pandas as pd,torch

torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(True)
OUT=Path(os.getenv('OUT','/mnt/data/k2w_stage5_confirm')); OUT.mkdir(parents=True,exist_ok=True)
SEED=int(os.getenv('SEED','0'))
H=32; P=3*H+1; E=48
# Freeze the development-pilot budget; no retuning on confirmatory ASTs.
META_STEPS=int(os.getenv('META_STEPS','220')); DIRECT_STEPS=int(os.getenv('DIRECT_STEPS','700')); BATCH=8
META_H=10; TRAIN_INNER_LR=.02; SUPPORT_N=16; QUERY_N=48; CAL_N=100; FINAL_N=int(os.getenv('FINAL_N','80'))
VAL_STEPS=20; EVAL=[0,1,2,5,10,20,50,100]; LR_CAND=[.005,.01,.02,.05,.1]; CKE=10; EPS=1e-8
E_MIN=float(os.getenv('E_MIN','.02')); SEP_MIN=float(os.getenv('SEP_MIN','.02'))
# First 10 train identities; last 4 held out of Stage-5 training.
ALL=[(-.8,.18),(-.4,.18),(0.,.18),(.4,.18),(.8,.18),(-.8,.30),(-.4,.30),(0.,.30),(.4,.30),(.8,.30),(-.2,.24),(.2,.24),(.6,.24),(-.6,.24)]
TRAIN_IDS=list(range(10)); HELD_IDS=list(range(10,14))
DESC=torch.tensor([[mu,sig,(sig-.24)/.12,mu*mu] for mu,sig in ALL],dtype=torch.float32)

def leaf(i,c): return ('leaf',int(i),float(c))
def add(a,b): return ('add',a,b)
def mul(a,b): return ('mul',a,b)
def prim(i,x):
    mu,sig=ALL[i]; z=x.squeeze(-1)
    return torch.exp(-.5*((z-mu)/sig)**2).view(*x.shape[:-1],1)
def eval_expr(e,x):
    if e[0]=='leaf': return e[2]*prim(e[1],x)
    a=eval_expr(e[1],x); b=eval_expr(e[2],x)
    return a+b if e[0]=='add' else a*b

def rc(rng): return (1 if rng.random()>.5 else -1)*(.55+.65*rng.random())

def make_train_template(name,rng,pool=TRAIN_IDS):
    ids=rng.sample(pool, {'leaf':1,'add2':2,'mul2':2,'add_mul_leaf':3,'mul_add_leaf':3}[name])
    L=[leaf(i,rc(rng)) for i in ids]
    if name=='leaf': return L[0]
    if name=='add2': return add(L[0],L[1])
    if name=='mul2': return mul(L[0],L[1])
    if name=='add_mul_leaf': return add(mul(L[0],L[1]),L[2])
    if name=='mul_add_leaf': return mul(add(L[0],L[1]),L[2])
    raise ValueError(name)
TRAIN_TEMPLATES=['leaf','add2','mul2','add_mul_leaf','mul_add_leaf']

def sample_train_exprs(B):
    class GlobalRNG:
        def random(self): return random.random()
        def sample(self,pool,n): return random.sample(pool,n)
    g=GlobalRNG()
    return [make_train_template(random.choice(TRAIN_TEMPLATES),g) for _ in range(B)]

# Confirmatory paired ASTs: exactly same 4 leaves and coefficients within a pair.
# Only operator placement / tree wiring changes.
PAIR_NAMES=['placement','nested_scope','product_rewire','balanced_rewire']
def build_pair(kind,L):
    a,b,c,d=L
    if kind=='placement':
        return add(mul(a,b),add(c,d)), mul(add(a,b),add(c,d))
    if kind=='nested_scope':
        return add(mul(add(a,b),c),d), mul(add(a,b),add(c,d))
    if kind=='product_rewire':
        return add(mul(a,b),mul(c,d)), add(mul(a,c),mul(b,d))
    if kind=='balanced_rewire':
        return mul(add(a,b),add(c,d)), mul(add(a,c),add(b,d))
    raise ValueError(kind)

XENER=torch.linspace(-1,1,256).view(-1,1)
def make_paired_bank(kind,n,seed,pool=TRAIN_IDS,q=256):
    rng=random.Random(seed); pairs=[]; meta=[]; attempts=0
    while len(pairs)<n:
        attempts+=1
        if attempts>200000: raise RuntimeError(f'acceptance failure {kind}')
        ids=rng.sample(pool,4); cs=[rc(rng) for _ in range(4)]
        L=[leaf(i,c) for i,c in zip(ids,cs)]; ea,eb=build_pair(kind,L)
        ya,yb=eval_expr(ea,XENER),eval_expr(eb,XENER)
        en_a=float((ya*ya).mean()); en_b=float((yb*yb).mean()); sep=float(((ya-yb)**2).mean())
        if min(en_a,en_b)<E_MIN or sep<SEP_MIN: continue
        pid=len(pairs); pairs.append((ea,eb)); meta.append((pid,ids,cs,en_a,en_b,sep))
    exprs=[e for p in pairs for e in p]
    N=len(exprs); g=torch.Generator().manual_seed(seed+991)
    sx=2*torch.rand(N,100,SUPPORT_N,1,generator=g)-1
    sy=torch.stack([torch.stack([eval_expr(exprs[i],sx[i,j]) for j in range(100)],0) for i in range(N)],0)
    xq=torch.linspace(-1,1,q).view(1,q,1).repeat(N,1,1)
    yq=torch.stack([eval_expr(exprs[i],xq[i]) for i in range(N)],0)
    pair_id=np.repeat(np.arange(n),2); variant=np.tile(np.array(['A','B']),n)
    md=[]
    for pid,ids,cs,en_a,en_b,sep in meta:
        for v,en in [('A',en_a),('B',en_b)]:
            md.append({'pair_id':pid,'variant':v,'target_energy':en,'target_pair_sep':sep,
                       **{f'id{i}':ids[i] for i in range(4)},**{f'c{i}':cs[i] for i in range(4)}})
    return {'exprs':exprs,'sx':sx,'sy':sy,'xq':xq,'yq':yq,'pair_id':pair_id,'variant':variant,'meta':pd.DataFrame(md),'attempts':attempts}

def make_cal_bank(n,seed,q=128):
    rng=random.Random(seed); es=[make_train_template('add_mul_leaf',rng) for _ in range(n)]
    g=torch.Generator().manual_seed(seed+91); sx=2*torch.rand(n,100,SUPPORT_N,1,generator=g)-1
    sy=torch.stack([torch.stack([eval_expr(es[i],sx[i,j]) for j in range(100)],0) for i in range(n)],0)
    xq=torch.linspace(-1,1,q).view(1,q,1).repeat(n,1,1); yq=torch.stack([eval_expr(es[i],xq[i]) for i in range(n)],0)
    return {'exprs':es,'sx':sx,'sy':sy,'xq':xq,'yq':yq}

def target_forward(th,x):
    if th.ndim==1:
        i=0;W1=th[i:i+H].view(H,1);i+=H;b1=th[i:i+H].view(1,H);i+=H;W2=th[i:i+H].view(1,H);i+=H;b2=th[i:i+1].view(1,1)
        return torch.tanh(x@W1.T+b1)@W2.T+b2
    B=th.shape[0]; i=0; W1=th[:,i:i+H].view(B,H,1);i+=H;b1=th[:,i:i+H].view(B,1,H);i+=H;W2=th[:,i:i+H].view(B,1,H);i+=H;b2=th[:,i:i+1].view(B,1,1)
    return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1),W2.transpose(1,2))+b2

def manual_grad(th,x,y):
    sq=th.ndim==1; tb=th.view(1,-1) if sq else th; xb=x.view(1,*x.shape) if sq else x; yb=y.view(1,*y.shape) if sq else y
    B=tb.shape[0]; i=0; W1=tb[:,i:i+H].view(B,H);i+=H;b1=tb[:,i:i+H].view(B,H);i+=H;W2=tb[:,i:i+H].view(B,H);i+=H;b2=tb[:,i:i+1].view(B,1)
    xx=xb.squeeze(-1); yy=yb.squeeze(-1); z=xx[:,:,None]*W1[:,None,:]+b1[:,None,:]; a=torch.tanh(z); pred=(a*W2[:,None,:]).sum(2)+b2
    d=2*(pred-yy)/xx.shape[1]; gb2=d.sum(1,keepdim=True); gW2=(d[:,:,None]*a).sum(1); gz=d[:,:,None]*W2[:,None,:]*(1-a*a); gb1=gz.sum(1); gW1=(gz*xx[:,:,None]).sum(1)
    g=torch.cat([gW1,gb1,gW2,gb2],1); return g[0] if sq else g

class LeafEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('desc',DESC.clone()); self.l1=torch.nn.Linear(4,E); self.l2=torch.nn.Linear(E,E)
    def emb(self,i): return torch.tanh(self.l2(torch.tanh(self.l1(self.desc[i]))))
class TreeHyper(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.leafenc=LeafEncoder(); self.mul1=torch.nn.Linear(3*E,E); self.mul2=torch.nn.Linear(E,E); self.base=torch.nn.Parameter(torch.randn(P)*.12); self.rho=torch.nn.Linear(E,P)
        torch.nn.init.zeros_(self.rho.weight); torch.nn.init.zeros_(self.rho.bias)
    def enc_one(self,e):
        if e[0]=='leaf': return e[2]*self.leafenc.emb(e[1])
        a=self.enc_one(e[1]); b=self.enc_one(e[2])
        if e[0]=='add': return a+b
        z=torch.cat([a+b,a*b,torch.abs(a-b)],0); return torch.tanh(self.mul2(torch.tanh(self.mul1(z))))
    def forward(self,exprs): return torch.stack([self.base+self.rho(torch.tanh(self.enc_one(e))) for e in exprs],0)
class BagHyper(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.leafenc=LeafEncoder(); self.base=torch.nn.Parameter(torch.randn(P)*.12); self.rho=torch.nn.Linear(E,P)
        torch.nn.init.zeros_(self.rho.weight); torch.nn.init.zeros_(self.rho.bias)
    def enc_one(self,e):
        if e[0]=='leaf': return e[2]*self.leafenc.emb(e[1])
        return self.enc_one(e[1])+self.enc_one(e[2])
    def forward(self,exprs): return torch.stack([self.base+self.rho(torch.tanh(self.enc_one(e))) for e in exprs],0)

def sample_xy(exprs,n):
    B=len(exprs); x=2*torch.rand(B,n,1)-1; y=torch.stack([eval_expr(exprs[i],x[i]) for i in range(B)],0); return x,y

def train_meta(factory,off):
    torch.manual_seed(off+SEED); np.random.seed(off+SEED); random.seed(off+SEED); m=factory(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for _ in range(META_STEPS):
        es=sample_train_exprs(BATCH); t0=m(es); th=t0
        for j in range(META_H):
            sx,sy=sample_xy(es,SUPPORT_N); th=th-TRAIN_INNER_LR*manual_grad(th,sx,sy).detach()
        qx,qy=sample_xy(es,QUERY_N); g=manual_grad(th,qx,qy).detach()/BATCH; opt.zero_grad(); torch.autograd.backward(t0,grad_tensors=g); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m

def train_direct(factory,off):
    torch.manual_seed(off+SEED); np.random.seed(off+SEED); random.seed(off+SEED); m=factory(); opt=torch.optim.Adam(m.parameters(),lr=1e-3); ck={0:copy.deepcopy(m.state_dict())}
    for s in range(1,DIRECT_STEPS+1):
        es=sample_train_exprs(BATCH); qx,qy=sample_xy(es,QUERY_N); th=m(es); g=manual_grad(th,qx,qy).detach()/BATCH; opt.zero_grad(); torch.autograd.backward(th,grad_tensors=g); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if s%CKE==0: ck[s]=copy.deepcopy(m.state_dict())
    return m,ck

def zero(th,b): return float(((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2)).median())
def curve(th,b,lr):
    th=th.detach().clone(); o={}
    for s in range(101):
        if s in EVAL: o[s]=((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2)).detach().cpu()
        if s<100: th=th-lr*manual_grad(th,b['sx'][:,s],b['sy'][:,s])
    return o

def val(th,b,lr):
    th=th.detach().clone(); a=[]
    for s in range(VAL_STEPS+1):
        a.append(torch.log(((target_forward(th,b['xq'])-b['yq'])**2).mean((1,2))+EPS))
        if s<VAL_STEPS: th=th-lr*manual_grad(th,b['sx'][:,s],b['sy'][:,s])
    return float(torch.stack(a,1).mean(1).median())
def choose_lr(th,b): return min((val(th,b,l),l) for l in LR_CAND)[1]
def select_ck(ck,meta,b,factory):
    with torch.no_grad(): tar=zero(meta(b['exprs']),b)
    tmp=factory(); best=None
    for s,st in ck.items():
        tmp.load_state_dict(st)
        with torch.no_grad(): med=zero(tmp(b['exprs']),b)
        rec=(abs(math.log(med+EPS)-math.log(tar+EPS)),-s,s,med)
        if best is None or rec<best: best=rec
    _,_,s,med=best; tmp.load_state_dict(ck[s]); return copy.deepcopy(tmp),s,tar,med

def metric(c):
    m0=c[0].numpy(); rel=np.stack([np.log((c[k].numpy()+EPS)/(m0+EPS)) for k in EVAL if k>0],1).mean(1)
    return rel,m0,c[100].numpy()

def add_rows(rows,sm,method,c,lr,name,b):
    rel,z,h=metric(c); sm.append({'seed':SEED,'method':method,'template':name,'selected_lr':lr,'rel_log_auc':float(np.median(rel)),'zero_median':float(np.median(z)),'step100_median':float(np.median(h))})
    md=b['meta'].reset_index(drop=True)
    for k in EVAL:
        vals=c[k].numpy()
        for i,v in enumerate(vals):
            rec={'seed':SEED,'method':method,'template':name,'adapt_steps':k,'task_index':i,'mse':float(v),'selected_lr':lr}
            for col in ['pair_id','variant','target_energy','target_pair_sep']: rec[col]=md.loc[i,col]
            rows.append(rec)

def pair_diagnostics(model,b,family,name):
    with torch.no_grad():
        th=model(b['exprs']); pred=target_forward(th,b['xq'])
    recs=[]
    for pid in range(FINAL_N):
        ia,ib=2*pid,2*pid+1; targ_sep=float(((b['yq'][ia]-b['yq'][ib])**2).mean()); pred_sep=float(((pred[ia]-pred[ib])**2).mean()); wdiff=float(torch.max(torch.abs(th[ia]-th[ib])))
        ma=float(((pred[ia]-b['yq'][ia])**2).mean()); mb=float(((pred[ib]-b['yq'][ib])**2).mean()); avg=.5*(ma+mb); lb=.25*targ_sep
        recs.append({'seed':SEED,'family':family,'template':name,'pair_id':pid,'target_pair_sep':targ_sep,'pred_pair_sep':pred_sep,'max_weight_diff':wdiff,'pair_avg_zero_mse':avg,'bag_common_prediction_lower_bound':lb,'mse_over_sep4':avg/(lb+EPS)})
    return recs

cal=make_cal_bank(CAL_N,20263001)
banks={n:make_paired_bank(n,FINAL_N,20263100+i,TRAIN_IDS) for i,n in enumerate(PAIR_NAMES)}
# Secondary combines unseen semantic leaves with a new operator wiring.
banks['unseen_balanced_rewire']=make_paired_bank('balanced_rewire',FINAL_N,20263200,HELD_IDS)
rows=[]; sm=[]; matches=[]; tim=[]; pdiag=[]
# Tree: Direct + Meta + loss-matched Direct
factory=TreeHyper; off=300000
t=time.time(); tree_direct,tree_ck=train_direct(factory,off); tim.append(('Tree_Direct',time.time()-t)); t=time.time(); tree_meta=train_meta(factory,off+50000); tim.append(('Tree_Meta',time.time()-t)); tree_matched,step,mz,dz=select_ck(tree_ck,tree_meta,cal,factory); matches.append({'seed':SEED,'family':'Tree','match_step':step,'meta_calib_zero':mz,'direct_calib_zero':dz,'calib_log_gap':abs(math.log(mz+EPS)-math.log(dz+EPS))})
with torch.no_grad(): cths=[('Tree_Meta',tree_meta(cal['exprs'])),('Tree_Matched',tree_matched(cal['exprs'])),('Tree_DirectFinal',tree_direct(cal['exprs']))]
lrs={n:choose_lr(th,cal) for n,th in cths}
# Bag direct baseline only; no confirmatory tuning.
t=time.time(); bag_direct,_=train_direct(BagHyper,400000); tim.append(('Bag_Direct',time.time()-t))
with torch.no_grad(): bag_cal=bag_direct(cal['exprs'])
lrs['Bag_DirectFinal']=choose_lr(bag_cal,cal)
for name,b in banks.items():
    with torch.no_grad(): ths=[tree_meta(b['exprs']),tree_matched(b['exprs']),tree_direct(b['exprs']),bag_direct(b['exprs'])]
    for n,th in zip(['Tree_Meta','Tree_Matched','Tree_DirectFinal','Bag_DirectFinal'],ths): add_rows(rows,sm,n,curve(th,b,lrs[n]),lrs[n],name,b)
    pdiag.extend(pair_diagnostics(tree_direct,b,'Tree',name)); pdiag.extend(pair_diagnostics(bag_direct,b,'Bag',name))

pd.DataFrame(rows).to_csv(OUT/f'task_seed{SEED:02d}.csv',index=False)
pd.DataFrame(sm).to_csv(OUT/f'seed_metrics_seed{SEED:02d}.csv',index=False)
pd.DataFrame(matches).to_csv(OUT/f'matching_seed{SEED:02d}.csv',index=False)
pd.DataFrame(tim,columns=['method','wall_seconds']).assign(seed=SEED).to_csv(OUT/f'timing_seed{SEED:02d}.csv',index=False)
pd.DataFrame(pdiag).to_csv(OUT/f'pair_diag_seed{SEED:02d}.csv',index=False)
pd.DataFrame([{'template':n,'accepted_pairs':FINAL_N,'attempts':b['attempts'],'accept_rate':FINAL_N/b['attempts'],'E_MIN':E_MIN,'SEP_MIN':SEP_MIN} for n,b in banks.items()]).to_csv(OUT/f'bank_diag_seed{SEED:02d}.csv',index=False)
print(json.dumps({'seed':SEED,'match':matches[0],'metrics':sm,'timing':tim},indent=2))
