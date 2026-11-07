"""Post-hoc support-only Armijo backtracking; no training or retuning."""
import argparse,json
import numpy as np,pandas as pd,torch
from stage7_experiment import ROOT,Hyper,load_bank,loss_values,physics_grad,residual

def safe_curve(th,bank,lr):
    r=torch.Generator().manual_seed(89000);th=th.detach().clone();steps=[0,1,2,5,10,20,50,100];out={0:{k:v.detach().numpy() for k,v in loss_values(th,bank).items()}}
    counts=torch.zeros(len(th),dtype=torch.int64);fallback=torch.zeros_like(counts);maxrise=0.
    for k in range(1,101):
        idx=torch.randint(bank['xs'].shape[1],(len(th),32),generator=r);x=bank['xs'].gather(1,idx);f=bank['fs'].gather(1,idx);grad=physics_grad(th,x,f,bank['lam'],bank['b'],bank['scale'])
        old=residual(th,x,f,bank['lam'],bank['b']).square().mean(1)/bank['scale'];g2=grad.square().sum(1);step=torch.full_like(old,lr);done=torch.zeros(len(th),dtype=torch.bool);accepted=th.clone();acceptedloss=old.clone()
        for bt in range(21):
            trial=th-step[:,None]*grad;new=residual(trial,x,f,bank['lam'],bank['b']).square().mean(1)/bank['scale'];valid=torch.isfinite(new)&(new<=old-1e-4*step*g2);take=valid&~done
            accepted[take]=trial[take];acceptedloss[take]=new[take];counts[take]+=bt;done|=take
            if bool(done.all()):break
            step[~done]*=.5
        fallback[~done]+=1;maxrise=max(maxrise,float((acceptedloss-old).max()));th=accepted
        if k in steps:out[k]={j:v.detach().numpy() for j,v in loss_values(th,bank).items()}
    assert torch.isfinite(th).all() and maxrise<=1e-10
    return out,counts.numpy(),fallback.numpy(),maxrise

def run(seed):
    p=ROOT/'results'/'confirmatory'/f'seed{seed:03d}';bank=load_bank('test');cfg=json.loads((p/'selection.json').read_text());rows=[];diag=[]
    for name,file in [('DirectBest','bestdirect.pt'),('Meta','meta.pt'),('RegMeta','regmeta.pt')]:
        m=Hyper();m.load_state_dict(torch.load(p/file,weights_only=True))
        with torch.no_grad():cv,cnt,fb,rise=safe_curve(m(bank),bank,cfg['lrs'][name]['lr'])
        for k,vals in cv.items():
            for i,t in enumerate(bank['meta']):rows.append({'seed':seed,'method':name,'task':i,'group':t['group'],'lambda':t['lambda'],'step':k,**{v:float(a[i]) for v,a in vals.items()}})
        for i,t in enumerate(bank['meta']):diag.append({'seed':seed,'method':name,'task':i,'group':t['group'],'lambda':t['lambda'],'backtracks_total':int(cnt[i]),'fallback_steps':int(fb[i]),'max_support_loss_rise_all_tasks':rise})
    pd.DataFrame(rows).to_csv(p/'safe_adaptation.csv',index=False);pd.DataFrame(diag).to_csv(p/'safe_adaptation_diag.csv',index=False);print('safe',seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);run(p.parse_args().seed)
