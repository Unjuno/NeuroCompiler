import math, random, time, platform, sys, copy, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

OUT = Path('/mnt/data/k2w_stage1_results')
OUT.mkdir(parents=True, exist_ok=True)
TAG = os.getenv('STAGE1_TAG','all')
RESULTS_PATH = OUT/f'stage1_task_results_{TAG}.csv'
SUMMARY_PATH = OUT/f'stage1_summary_{TAG}.csv'
SEED_PATH = OUT/f'stage1_seed_metrics_{TAG}.csv'
LR_PATH = OUT/f'stage1_lr_selection_{TAG}.csv'
MATCH_PATH = OUT/f'stage1_matching_{TAG}.csv'
PLOT_PATH = OUT/f'stage1_learning_curve_{TAG}.png'

# Stage-1 preregistered-like configuration
SEED_START=int(os.getenv('SEED_START','0')); SEED_END=int(os.getenv('SEED_END','20')); SEEDS=list(range(SEED_START,SEED_END))
H = 16
P = H + H + H + 1
META_STEPS = 400
META_BATCH = 4
META_INNER_STEPS = 2
TRAIN_INNER_LR = 0.05
SUPPORT_N = 12
QUERY_N = 32
VAL_TASKS_PER_MASK = 40
TEST_TASKS_PER_MASK = 200
VAL_ADAPT_STEPS = 20
EVAL_STEPS = [0,1,2,5,10,20,50,100]
LR_CANDIDATES = [0.01,0.02,0.05,0.10]
CHECKPOINT_EVERY = 10
EPS = 1e-8

TRAIN_MASKS = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1]], dtype=torch.float32)
TEST_MASKS = torch.tensor([[1,0,1],[1,1,1]], dtype=torch.float32)
TEST_NAMES = ['A+C','A+B+C']

def target_forward(theta, x):
    # theta: [P] or [B,P]; x: [N,1] or [B,N,1]
    if theta.ndim == 1:
        i=0
        W1=theta[i:i+H].view(H,1); i+=H
        b1=theta[i:i+H].view(1,H); i+=H
        W2=theta[i:i+H].view(1,H); i+=H
        b2=theta[i:i+1].view(1,1)
        return torch.tanh(x @ W1.T + b1) @ W2.T + b2
    B=theta.shape[0]
    i=0
    W1=theta[:,i:i+H].view(B,H,1); i+=H
    b1=theta[:,i:i+H].view(B,1,H); i+=H
    W2=theta[:,i:i+H].view(B,1,H); i+=H
    b2=theta[:,i:i+1].view(B,1,1)
    return torch.bmm(torch.tanh(torch.bmm(x,W1.transpose(1,2))+b1), W2.transpose(1,2)) + b2

def true_function(c,x):
    if c.ndim == 1:
        return c[0]*torch.sin(math.pi*x) + c[1]*x + c[2]*(2*x*x-2/3)
    cc=c[:,:,None]
    return cc[:,0:1]*torch.sin(math.pi*x) + cc[:,1:2]*x + cc[:,2:3]*(2*x*x-2/3)

def sample_coeff(mask):
    signs=torch.where(torch.rand(3)>0.5,1.0,-1.0)
    mags=0.5+torch.rand(3)
    return mask*signs*mags

def sample_xy(c,n):
    x=2*torch.rand(n,1)-1
    return x,true_function(c,x)


def manual_theta_grad(theta, x, y):
    # Exact gradient of taskwise mean MSE w.r.t. target-network theta.
    squeeze = theta.ndim == 1
    if squeeze:
        theta_b = theta.view(1,-1); x_b=x.view(1,*x.shape); y_b=y.view(1,*y.shape)
    else:
        theta_b=theta; x_b=x; y_b=y
    B=theta_b.shape[0]; i=0
    W1=theta_b[:,i:i+H].view(B,H); i+=H
    b1=theta_b[:,i:i+H].view(B,H); i+=H
    W2=theta_b[:,i:i+H].view(B,H); i+=H
    b2=theta_b[:,i:i+1].view(B,1)
    xx=x_b.squeeze(-1)  # [B,N]
    yy=y_b.squeeze(-1)
    z=xx[:,:,None]*W1[:,None,:] + b1[:,None,:]
    a=torch.tanh(z)
    pred=(a*W2[:,None,:]).sum(dim=2)+b2
    N=xx.shape[1]
    d=2.0*(pred-yy)/N  # [B,N]
    gb2=d.sum(dim=1,keepdim=True)
    gW2=(d[:,:,None]*a).sum(dim=1)
    gz=d[:,:,None]*W2[:,None,:]*(1.0-a*a)
    gb1=gz.sum(dim=1)
    gW1=(gz*xx[:,:,None]).sum(dim=1)
    g=torch.cat([gW1,gb1,gW2,gb2],dim=1)
    return g[0] if squeeze else g

def inner_adapt(theta0,c,steps,lr=TRAIN_INNER_LR,create_graph=False):
    th=theta0
    for _ in range(steps):
        x,y=sample_xy(c,SUPPORT_N)
        g=manual_theta_grad(th,x,y).detach()
        th=th-lr*g
    return th

class HyperInit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Parameter(torch.randn(P)*0.15)
        self.net=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.Tanh(),torch.nn.Linear(32,P))
        torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,c):
        return self.base+self.net(c)

def train_conditional(objective='meta'):
    model=HyperInit(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    checkpoints={0:copy.deepcopy(model.state_dict())}
    for step in range(1,META_STEPS+1):
        opt.zero_grad(); losses=[]
        for _ in range(META_BATCH):
            c=sample_coeff(TRAIN_MASKS[random.randrange(len(TRAIN_MASKS))])
            if objective=='meta':
                th=inner_adapt(model(c),c,META_INNER_STEPS,create_graph=False)
                xq,yq=sample_xy(c,QUERY_N)
                losses.append(((target_forward(th,xq)-yq)**2).mean())
            else:
                xq,yq=sample_xy(c,QUERY_N)
                losses.append(((target_forward(model(c),xq)-yq)**2).mean())
        loss=torch.stack(losses).mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%CHECKPOINT_EVERY==0:
            checkpoints[step]=copy.deepcopy(model.state_dict())
    return model,checkpoints

def train_fo_maml():
    theta=torch.nn.Parameter(torch.randn(P)*0.15); opt=torch.optim.Adam([theta],lr=1e-3)
    for _ in range(META_STEPS):
        opt.zero_grad(); losses=[]
        for _ in range(META_BATCH):
            c=sample_coeff(TRAIN_MASKS[random.randrange(len(TRAIN_MASKS))])
            th=inner_adapt(theta,c,META_INNER_STEPS,create_graph=False)
            xq,yq=sample_xy(c,QUERY_N)
            losses.append(((target_forward(th,xq)-yq)**2).mean())
        loss=torch.stack(losses).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_([theta],1.0); opt.step()
    return theta.detach()

def make_bank(masks,n_per_mask,max_steps,seed,query_n=256):
    g=torch.Generator().manual_seed(seed)
    cs=[]; names=[]
    for mi,mask in enumerate(masks):
        for _ in range(n_per_mask):
            signs=torch.where(torch.rand(3,generator=g)>0.5,1.0,-1.0)
            mags=0.5+torch.rand(3,generator=g)
            cs.append(mask*signs*mags)
            names.append(TEST_NAMES[mi] if len(masks)==2 else f'trainmask{mi}')
    c=torch.stack(cs)
    B=len(cs)
    support_x=2*torch.rand(B,max_steps,SUPPORT_N,1,generator=g)-1
    support_y=true_function(c[:,None,:].expand(B,max_steps,3).reshape(-1,3), support_x.reshape(-1,SUPPORT_N,1)).reshape(B,max_steps,SUPPORT_N,1)
    xq=torch.linspace(-1,1,query_n).view(1,query_n,1).repeat(B,1,1)
    yq=true_function(c,xq)
    return {'c':c,'names':names,'support_x':support_x,'support_y':support_y,'xq':xq,'yq':yq}

def model_theta(model,c):
    with torch.no_grad():
        return model(c)

def eval_curve(theta0,bank,lr,steps_list):
    th=theta0.detach().clone()
    out={}
    maxs=max(steps_list)
    for step in range(maxs+1):
        if step in steps_list:
            mse=((target_forward(th,bank['xq'])-bank['yq'])**2).mean(dim=(1,2)).cpu()
            out[step]=mse
        if step==maxs: break
        x=bank['support_x'][:,step]; y=bank['support_y'][:,step]
        g=manual_theta_grad(th,x,y)
        th=th-lr*g
    return out

def zero_median(theta0,bank):
    with torch.no_grad():
        return float(((target_forward(theta0,bank['xq'])-bank['yq'])**2).mean(dim=(1,2)).median())

def select_directfit_checkpoint(checkpoints,meta_model,val_bank):
    meta_theta=model_theta(meta_model,val_bank['c'])
    target=zero_median(meta_theta,val_bank)
    temp=HyperInit()
    best=None
    for step,state in checkpoints.items():
        temp.load_state_dict(state)
        th=model_theta(temp,val_bank['c'])
        med=zero_median(th,val_bank)
        gap=abs(math.log(med+EPS)-math.log(target+EPS))
        rec=(gap,-step,step,med)
        if best is None or rec<best: best=rec
    _,_,step,med=best
    temp.load_state_dict(checkpoints[step])
    return copy.deepcopy(temp),step,target,med

def choose_lr(theta0,val_bank):
    rec=[]
    for lr in LR_CANDIDATES:
        curve=eval_curve(theta0,val_bank,lr,list(range(VAL_ADAPT_STEPS+1)))
        vals=torch.stack([torch.log(curve[k]+EPS) for k in range(VAL_ADAPT_STEPS+1)],dim=1).mean(dim=1)
        score=float(vals.median())
        rec.append((score,lr))
    rec.sort()
    return rec[0][1],rec

def seed_auc(curve):
    # mean log-MSE over preregistered sparse checkpoints, taskwise then median
    mat=torch.stack([torch.log(curve[k]+EPS) for k in EVAL_STEPS],dim=1)
    return float(mat.mean(dim=1).median())

val_bank=make_bank(TRAIN_MASKS,VAL_TASKS_PER_MASK,100,seed=20260830,query_n=128)
test_bank=make_bank(TEST_MASKS,TEST_TASKS_PER_MASK,100,seed=20260831,query_n=256)

rows=[]; seed_rows=[]; lr_rows=[]; match_rows=[]; times=[]
t_all=time.time()
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    t0=time.time(); meta,meta_ck=train_conditional('meta'); t_meta=time.time()-t0
    t0=time.time(); direct,direct_ck=train_conditional('direct'); t_direct=time.time()-t0
    t0=time.time(); maml=train_fo_maml(); t_maml=time.time()-t0

    direct_match,match_step,meta_val0,direct_val0=select_directfit_checkpoint(direct_ck,meta,val_bank)
    match_rows.append({'seed':seed,'meta_checkpoint':META_STEPS,'directfit_matched_checkpoint':match_step,
                       'meta_val_zero_median':meta_val0,'directfit_matched_val_zero_median':direct_val0,
                       'log_gap':abs(math.log(meta_val0+EPS)-math.log(direct_val0+EPS))})

    # Fixed random initializations per seed/task, shared across LR tuning/test for reproducibility.
    gg=torch.Generator().manual_seed(50000+seed)
    rand_val=torch.randn(len(val_bank['c']),P,generator=gg)*0.15
    rand_test=torch.randn(len(test_bank['c']),P,generator=gg)*0.15

    init_val={
        'MetaInit':model_theta(meta,val_bank['c']),
        'DirectFit_matched':model_theta(direct_match,val_bank['c']),
        'DirectFit_final':model_theta(direct,val_bank['c']),
        'FO-MAML_shared':maml.view(1,-1).repeat(len(val_bank['c']),1),
        'Random':rand_val,
    }
    init_test={
        'MetaInit':model_theta(meta,test_bank['c']),
        'DirectFit_matched':model_theta(direct_match,test_bank['c']),
        'DirectFit_final':model_theta(direct,test_bank['c']),
        'FO-MAML_shared':maml.view(1,-1).repeat(len(test_bank['c']),1),
        'Random':rand_test,
    }
    selected={}
    for method,thv in init_val.items():
        lr,recs=choose_lr(thv,val_bank); selected[method]=lr
        for score,lrc in recs:
            lr_rows.append({'seed':seed,'method':method,'lr':lrc,'val_log_auc_median':score,'selected':lrc==lr})

    for method,th0 in init_test.items():
        curve=eval_curve(th0,test_bank,selected[method],EVAL_STEPS)
        seed_rows.append({'seed':seed,'method':method,'selected_lr':selected[method],
                          'log_auc_median':seed_auc(curve),
                          'zero_median':float(curve[0].median()),
                          'step100_median':float(curve[100].median())})
        for k in EVAL_STEPS:
            vals=curve[k].numpy()
            for idx,mse in enumerate(vals):
                rows.append({'seed':seed,'method':method,'adapt_steps':k,'task_index':idx,
                             'mask':test_bank['names'][idx],'mse':float(mse),'selected_lr':selected[method]})
    times.append((seed,t_meta,t_direct,t_maml))
    print(f'seed {seed:02d}: train meta={t_meta:.2f}s direct={t_direct:.2f}s maml={t_maml:.2f}s match={match_step} ' +
          ' '.join(f'{m}:{selected[m]:.2g}' for m in selected), flush=True)

pd.DataFrame(rows).to_csv(RESULTS_PATH,index=False)
pd.DataFrame(seed_rows).to_csv(SEED_PATH,index=False)
pd.DataFrame(lr_rows).to_csv(LR_PATH,index=False)
pd.DataFrame(match_rows).to_csv(MATCH_PATH,index=False)

df=pd.DataFrame(rows)
summary=(df.groupby(['method','adapt_steps'])['mse']
         .agg(n='count',median='median',mean='mean',q25=lambda s:s.quantile(.25),q75=lambda s:s.quantile(.75))
         .reset_index())
# seed medians for honest independence view
sm=df.groupby(['seed','method','adapt_steps'])['mse'].median().reset_index()
ss=(sm.groupby(['method','adapt_steps'])['mse']
    .agg(seed_median='median',seed_q25=lambda s:s.quantile(.25),seed_q75=lambda s:s.quantile(.75),seed_min='min',seed_max='max').reset_index())
summary=summary.merge(ss,on=['method','adapt_steps'])
summary.to_csv(SUMMARY_PATH,index=False)

fig,ax=plt.subplots(figsize=(7.5,4.8))
for method in ['MetaInit','DirectFit_matched','DirectFit_final','FO-MAML_shared','Random']:
    s=summary[summary.method==method].sort_values('adapt_steps')
    ax.plot(s.adapt_steps,s.seed_median,marker='o',label=method)
ax.set_yscale('log'); ax.set_xlabel('Adaptation gradient steps'); ax.set_ylabel('Median held-out MSE across seeds')
ax.set_title('Stage-1: unseen compositions, fixed adaptation trajectories'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(PLOT_PATH,dpi=180); plt.close(fig)

# paired bootstrap over independent training seeds for key AUC contrast
sr=pd.DataFrame(seed_rows)
piv=sr.pivot(index='seed',columns='method',values='log_auc_median')
diff=(piv['MetaInit']-piv['DirectFit_matched']).to_numpy()
rng=np.random.default_rng(20260831)
boot=np.array([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(20000)])
ci=np.quantile(boot,[.025,.975]); mean_diff=diff.mean(); med_diff=np.median(diff)

print('\n=== Stage-1 completed ===')
print(f'Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()}')
print(f'Platform: {platform.platform()}')
print(f'Total wall time: {time.time()-t_all:.2f} s')
print('Training times (seed, meta, direct, maml):',times)
print('\nMatching summary:')
print(pd.DataFrame(match_rows).describe().to_string())
print('\nSeed-level metrics:')
print(sr.groupby('method')[['log_auc_median','zero_median','step100_median']].median().to_string())
print(f'\nPaired seed bootstrap MetaInit - DirectFit_matched log-AUC: mean={mean_diff:.6f}, median={med_diff:.6f}, 95% CI=[{ci[0]:.6f},{ci[1]:.6f}] (negative favors MetaInit)')
print('\nLearning curve seed-medians:')
print(summary[['method','adapt_steps','seed_median','seed_q25','seed_q75']].to_string(index=False,float_format=lambda z:f'{z:.6f}'))
