"""Fixed-feature physics least squares. FP64 gelsd; validation-only cutoff."""
from __future__ import annotations
import argparse,json,time
import numpy as np,pandas as pd,torch
from stage7_experiment import ROOT,H,P,Hyper,load_bank,loss_values
RCONDS=[1e-3,1e-4,1e-5,1e-6,1e-8,1e-10]
def head_solve(th,bank,n,rcond):
    th=th.detach().double().clone();w,d=th[:,:H],th[:,H:2*H];x=bank['xs'][:,:n].double();f=bank['fs'][:,:n].double();lam=bank['lam'].double();b=bank['b'].double()
    h=torch.tanh(x[...,None]*w[:,None]+d[:,None]);s=1-h*h;A=1+lam[:,None,None]*(x+1)[...,None];basis=A*h+(x+1)[...,None]*w[:,None]*s
    design=torch.cat([basis,A],-1);rhs=f-(lam*b)[:,None];fit=torch.linalg.lstsq(design,rhs[...,None],rcond=rcond,driver='gelsd');th[:,2*H:]=fit.solution.squeeze(-1);sv=fit.singular_values
    return th,{'rank':fit.rank.numpy(),'max_abs_head':th[:,2*H:].abs().max(1).values.numpy(),'smallest_over_largest_singular':(sv[:,-1]/sv[:,0]).numpy()}
def fp_bank(bank,dtype):return {k:v.to(dtype) if isinstance(v,torch.Tensor) and k!='solution' else v for k,v in bank.items()}
def choose_rc(th,bank,n,dtype):
    b=fp_bank(bank,dtype);scores=[]
    for rc in RCONDS:
        fitted,_=head_solve(th,bank,n,rc);vals=loss_values(fitted.to(dtype),b);score=float(torch.log(vals['norm_residual']+1e-12).mean());scores.append((score,rc))
    return min(scores)[1],scores

def run(seed):
    src=ROOT/'results'/'confirmatory'/f'seed{seed:03d}';val=load_bank('validation');test=load_bank('test');rows=[];tim=[];choices={}
    for name,file,blind in [('DirectBest','bestdirect.pt',False),('Meta','meta.pt',False),('RegMeta','regmeta.pt',False),('Random',None,True)]:
        torch.manual_seed(seed);m=Hyper(blind=blind)
        if file:m.load_state_dict(torch.load(src/file,weights_only=True))
        with torch.no_grad():tv=m(val);tt=m(test)
        for n in (32,128):
            for label,dtype in [('FP64',torch.float64),('FP32aware',torch.float32)]:
                t0=time.perf_counter();rc,scores=choose_rc(tv,val,n,dtype);validtime=time.perf_counter()-t0
                t0=time.perf_counter();head,di=head_solve(tt,test,n,rc);fitsec=time.perf_counter()-t0;vals=loss_values(head.to(dtype),fp_bank(test,dtype))
                if label=='FP64':vals32=loss_values(head.float(),fp_bank(test,torch.float32))
                choices[f'{name}_{n}_{label}']={'rcond':rc,'validation_scores':scores};tim.append({'seed':seed,'method':name,'points':n,'deployment':label,'validation_seconds':validtime,'solve_seconds':fitsec,'n_tasks':len(tt)})
                for i,t in enumerate(test['meta']):
                    rec={'seed':seed,'method':name,'points':n,'deployment':label,'task':i,'group':t['group'],'kind':t['kind'],'lambda':t['lambda'],'rcond':rc,**{k:float(v[i]) for k,v in vals.items()},**{k:float(v[i]) for k,v in di.items()}}
                    if label=='FP64':rec.update(naive_fp32_solution_mse=float(vals32['solution_mse'][i]),naive_fp32_norm_residual=float(vals32['norm_residual'][i]))
                    rows.append(rec)
    pd.DataFrame(rows).to_csv(src/'head_results.csv',index=False);pd.DataFrame(tim).to_csv(src/'head_timing.csv',index=False);(src/'head_selection.json').write_text(json.dumps(choices,indent=2));print('heads',seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);run(p.parse_args().seed)
