"""Exploratory amortized latency and classical spectral-grid follow-up, CPU single-thread."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
from fractions import Fraction
import numpy as np,pandas as pd,torch
import stage8_nonlinear as s
from stage8c_correction import newton_correction
from differential_ir import evaluate

root=s.ROOT;test=s.load('test',True)
rows=test['meta'];N=len(rows)
model=s.HyperNL();model.load_state_dict(torch.load(root/'seeds'/'301'/'direct.pt',weights_only=True));model.eval()
# stage8b correction uses same 129 eval grid, available forcing values precomputed.
x=test['xq'].numpy().astype(float)[0];ff=test['fq'].numpy().astype(float)
lam=test['lam'].numpy().astype(float);beta=test['beta'].numpy().astype(float);b=test['b'].numpy().astype(float)
input64=torch.tensor(test['xq'].numpy(),dtype=torch.float64)
ff64=torch.tensor(ff,dtype=torch.float64)
b64=test['b'].double();l64=test['lam'].double();be64=test['beta'].double()

def do_nn_correct(rounds=1):
    with torch.no_grad():
        theta=model(s.sub(test,slice(None))).double()
        yy=s.st.forward_solution(theta,input64,b64).numpy()
        rr=s.residual_nl(theta,input64,ff64,l64,b64,be64).numpy()
    if rounds:return newton_correction(yy,rr,lam,beta,x,rounds)[-1]
    return yy

def do_constant(rounds=2):
    yy=np.broadcast_to(b[:,None],(N,len(x))).copy()
    rr=lam[:,None]*yy+beta[:,None]*yy**3-ff
    return newton_correction(yy,rr,lam,beta,x,rounds)[-1]

functions=[('NN zero-shot',lambda:do_nn_correct(0)),('NN + 1 Newton',lambda:do_nn_correct(1)),('NN + 2 Newton',lambda:do_nn_correct(2)),
           ('Constant + 1 Newton',lambda:do_constant(1)),('Constant + 2 Newton',lambda:do_constant(2)),('Constant + 3 Newton',lambda:do_constant(3))]
bench=[]
for name,fn in functions:
    for _ in range(5):fn()
    samples=[]
    for _ in range(30):
        t=time.perf_counter();fn();samples.append(time.perf_counter()-t)
    bms=dict(method=name,batch_tasks=N,cpu_threads=1,median_total_ms=float(np.median(samples)*1000),
             p10_total_ms=float(np.quantile(samples,.1)*1000),p90_total_ms=float(np.quantile(samples,.9)*1000),
             median_us_per_task=float(np.median(samples)*1e6/N))
    bench.append(bms);print(bms,flush=True)

cla=pd.read_csv(root/'analysis'/'cheb_classical.csv')
bench.append(dict(method='Chebyshev (serial; support 128, degree 32)',batch_tasks=1,cpu_threads=1,
                  median_total_ms=float(cla.wall_s.median()*1000),p10_total_ms=float(cla.wall_s.quantile(.1)*1000),
                  p90_total_ms=float(cla.wall_s.quantile(.9)*1000),median_us_per_task=float(cla.wall_s.median()*1e6)))
pd.DataFrame(bench).to_csv(root/'analysis'/'latency_exploratory.csv',index=False)

# Chebyshev-Lobatto collocation (128 samples) as numerical stability ablation.
node=-np.cos(np.pi*np.arange(128)/127)
results=[]
for i,row in enumerate(rows):
    poly={tuple(tuple(atom) for atom in key):Fraction(num,den) for key,num,den in row['ir']}
    ffgrid=evaluate(poly,node,s.st.DESC)
    t=time.perf_counter()
    pred,diag=s.cheb_solve(node,ffgrid,row['lambda'],row['beta'],row['boundary'],x,32)
    secs=time.perf_counter()-t
    ref=test['solution'][i].numpy()
    results.append(dict(task=i,group=row['group'],mse_fp64=float(np.mean((pred-ref)**2)),
                        mse_fp32_output=float(np.mean((pred.astype(np.float32).astype(float)-ref)**2)),
                        wall_s_including_forcing_eval=secs,success=diag['success']))
res=pd.DataFrame(results)
res.to_csv(root/'analysis'/'cheb_lobatto_followup.csv',index=False)
print('Uniform support baseline group med/max')
print(cla.groupby('group').mse.agg(['median','max']))
print('Cheb-Lobatto nodes group med/max')
print(res.groupby('group').mse_fp64.agg(['median','max']))
