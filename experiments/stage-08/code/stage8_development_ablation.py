"""Stage 8b exploratory: detect whether the meta objective fails from horizon/LR mismatch.
Uses *development*, never Stage 8a confirmatory final. Model with direct best selected by validation residual.
"""
from pathlib import Path
import json, numpy as np, torch, pandas as pd
import stage8_nonlinear as s
s.LR_LIST=[.001,.003,.01,.03,.1,.2]
ROOT=s.ROOT/'development_ablation';ROOT.mkdir(exist_ok=True)
val=s.load('validation');dev=s.load('development',True)

def get_metrics(model):
    with torch.no_grad():
        vlr,selection=s.validation_lr(model(val),val,921800+301)
        cv=s.trajectory(model(s.sub(dev,slice(None))),dev,vlr,950000+301,10)
    return vlr, [float(np.median(cv[k]['solution_mse'])) for k in (0,1,2,5,10)], [float(np.mean(np.log(cv[k]['solution_mse']+1e-12))) for k in (0,1,2,5,10)]

direct,checkpoints,tsec,ex=s.train(301,'direct',direct_steps=1200)
sc=[]
for k,v in checkpoints.items():
    m=s.HyperNL();m.load_state_dict(v)
    with torch.no_grad():loss=float(s.metrics(m(val),val)['norm_residual'].mean())
    sc.append((loss,k))
_,bk=min(sc)
direct.load_state_dict(checkpoints[bk]);lr,vals,logs=get_metrics(direct)
rows=[dict(model='DirectBest',outer_steps=1200,horizon=0,inner_lr=0.,selected_lr=lr,mse0=vals[0],mse10=vals[-1],log_mse10=logs[-1],train_seconds=tsec,train_explosions=ex)]
print(rows[-1],flush=True)
for horizon,innerlr,steps in [(1,.003,300),(1,.01,300),(1,.03,300),(3,.003,300),(3,.01,300),(3,.03,300),(5,.003,300),(5,.01,300),(5,.03,300),(3,.003,900),(3,.01,900),(3,.03,900)]:
    try:
        m,_,sec,expl=s.train(301,'meta',meta_steps=steps,horizon=horizon,inner_lr=innerlr)
        lr,vals,logs=get_metrics(m)
        item=dict(model='Meta',outer_steps=steps,horizon=horizon,inner_lr=innerlr,selected_lr=lr,mse0=vals[0],mse10=vals[-1],log_mse10=logs[-1],train_seconds=sec,train_explosions=expl)
    except Exception as exc:
        item=dict(model='Meta',outer_steps=steps,horizon=horizon,inner_lr=innerlr,error=repr(exc))
    rows.append(item);pd.DataFrame(rows).to_csv(ROOT/'development_results.csv',index=False);print(item,flush=True)
