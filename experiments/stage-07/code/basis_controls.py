"""Falsification controls: shuffled semantics, tuned untrained features, Chebyshev.
All selections use validation residual only, never reference solution labels.
"""
from __future__ import annotations
import argparse,json,time
import numpy as np,pandas as pd,torch
from stage7_experiment import ROOT,H,P,Hyper,load_bank,loss_values
from linear_head_branch import head_solve,choose_rc

def shuf_forcing(bank):
    groups={}
    for i,t in enumerate(bank['meta']):groups.setdefault((t['group'],t['lambda']),[]).append(i)
    ix=torch.arange(len(bank['b']))
    for vv in groups.values():
        shift=max(1,len(vv)//3);ix[vv]=torch.tensor(vv[shift:]+vv[:shift])
    out=bank.copy()
    for k in ('desc','factor_mask','coeff'):out[k]=bank[k][ix]
    return out

def random_features(seed,scale,n):
    g=torch.Generator().manual_seed(300000+seed)
    w=scale*(.5+torch.rand(H,generator=g))*torch.where(torch.rand(H,generator=g)>.5,1.,-1.);centers=2*torch.rand(H,generator=g)-1;b=-w*centers
    return torch.cat([w,b,torch.zeros(H+1)])[None].repeat(n,1)

def cheb_design(x,degree):
    vs=[np.ones_like(x)];ds=[np.zeros_like(x)]
    if degree:vs.append(x.copy());ds.append(np.ones_like(x))
    for n in range(1,degree):vs.append(2*x*vs[-1]-vs[-2]);ds.append(2*vs[n]+2*x*ds[-1]-ds[-2])
    return np.stack(vs,-1),np.stack(ds,-1)

def cheb_solve(bank,degree,dtype):
    x=bank['xs'].numpy().astype(np.float64);f=bank['fs'].numpy().astype(np.float64);lam=bank['lam'].numpy().astype(np.float64);b=bank['b'].numpy().astype(np.float64)
    vs,ds=cheb_design(x,degree);A=(1+lam[:,None]*(x+1))[...,None]*vs+(x+1)[...,None]*ds;rhs=f-(lam*b)[:,None]
    coef=np.stack([np.linalg.lstsq(ai,yi,rcond=1e-10)[0] for ai,yi in zip(A,rhs)]).astype(dtype)
    q=bank['xq'].numpy().astype(dtype);ll=lam.astype(dtype);bb=b.astype(dtype);v,d=cheb_design(q,degree);vp=(v*coef[:,None]).sum(-1);dp=(d*coef[:,None]).sum(-1)
    u=bb[:,None]+(q+1)*vp;r=(1+ll[:,None]*(q+1))*vp+(q+1)*dp+(ll*bb)[:,None]-bank['fq'].numpy().astype(dtype);ref=bank['solution'].numpy().astype(np.float64)
    return {'solution_mse':np.mean((u-ref)**2,1),'norm_residual':np.mean(r*r,1)/bank['scale'].numpy()},coef

def run(seed):
    out=ROOT/'results'/'confirmatory'/f'seed{seed:03d}';val=load_bank('validation');test=load_bank('test');records=[];tim=[];selection={}
    def append(name,vals,**extras):
        for i,t in enumerate(test['meta']):records.append({'seed':seed,'method':name,'task':i,'group':t['group'],'kind':t['kind'],**{k:float(v[i]) for k,v in vals.items()},**extras})
    for name,file in [('DirectBest','bestdirect.pt'),('Meta','meta.pt'),('RegMeta','regmeta.pt')]:
        m=Hyper();m.load_state_dict(torch.load(out/file,weights_only=True))
        with torch.no_grad():tv=m(shuf_forcing(val));tt=m(shuf_forcing(test))
        rc,scores=choose_rc(tv,val,128,torch.float32);t0=time.perf_counter();fit,_=head_solve(tt,test,128,rc);sec=time.perf_counter()-t0
        append(name+'_ForcingShuffled_Head32',loss_values(fit.float(),test),rcond=rc);selection[name]={'rcond':rc,'validation_scores':scores};tim.append({'method':name+'_ForcingShuffled','solve_seconds':sec,'n_tasks':len(tt)})
    choices=[]
    for scale in (.7,1.5,3.,6.):
        tv=random_features(seed,scale,len(val['b']));rc,scores=choose_rc(tv,val,128,torch.float32);choices.append((min(s[0] for s in scores),scale,rc))
    _,sc,rc=min(choices);tt=random_features(seed,sc,len(test['b']));t0=time.perf_counter();fit,_=head_solve(tt,test,128,rc);sec=time.perf_counter()-t0
    append('RandomTuned_Head32',loss_values(fit.float(),test),scale=sc,rcond=rc);selection['RandomTuned']={'validation_candidates':choices,'scale':sc,'rcond':rc};tim.append({'method':'RandomTuned','solve_seconds':sec,'n_tasks':len(tt)})
    for label,dtype in [('FP32',np.float32),('FP64',np.float64)]:
        choices=[]
        for deg in (8,16,32):
            vals,_=cheb_solve(val,deg,dtype);score=float(np.log(vals['norm_residual']+1e-12).mean());choices.append((score,deg))
        _,deg=min(choices);t0=time.perf_counter();vals,_=cheb_solve(test,deg,dtype);sec=time.perf_counter()-t0;append('Chebyshev_'+label,vals,degree=deg)
        selection['Chebyshev_'+label]={'degree':deg,'validation_candidates':choices};tim.append({'method':'Chebyshev_'+label,'solve_and_eval_seconds':sec,'n_tasks':len(tt)})
    pd.DataFrame(records).to_csv(out/'basis_controls.csv',index=False);pd.DataFrame(tim).to_csv(out/'basis_timing.csv',index=False);(out/'basis_selection.json').write_text(json.dumps(selection,indent=2));print('basis',seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);run(p.parse_args().seed)
