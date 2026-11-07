"""Independent task replication of exploratory correction/solver routing, after method freeze."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
from fractions import Fraction
from pathlib import Path
import json,time,sys
import numpy as np,pandas as pd,torch
import stage8_nonlinear as s
from stage8c_correction import newton_correction
from differential_ir import evaluate
root=s.ROOT;out=root/'stage8d_confirm';out.mkdir(exist_ok=True)
b=s.load('stage8d_test',True);x=b['xq'].numpy().astype(float)[0]
n=len(b['meta']);meta=b['meta'];true=b['solution'].numpy()
lam=b['lam'].numpy().astype(float);beta=b['beta'].numpy().astype(float)
bound=b['b'].numpy().astype(float);f=b['fq'].numpy().astype(float)
records=[]

def register(seed,name,ans):
 for step,y in enumerate(ans):
  ms=np.mean((y-true)**2,axis=1)
  for i,row in enumerate(meta):records.append({'seed':seed,'method':name,'round':step,'group':row['group'],'task':i,'mse':float(ms[i])})

const=np.broadcast_to(bound[:,None],(n,len(x))).copy()
r=lam[:,None]*const+beta[:,None]*const**3-f
register(-1,'Constant',newton_correction(const,r,lam,beta,x,3))
print('Constant registered',flush=True)

for seed in range(301,311):
 for name,filename in [('DirectFinal','direct'),('Meta','meta')]:
  model=s.HyperNL().eval()
  model.load_state_dict(torch.load(root/'seeds'/f'{seed}'/(filename+'.pt'),weights_only=True))
  with torch.no_grad():
   th=model(s.sub(b,slice(None))).double()
   xx=torch.tensor(b['xq'].numpy(),dtype=torch.float64)
   ff=torch.tensor(f,dtype=torch.float64)
   bb=b['b'].double();ll=b['lam'].double();bt=b['beta'].double()
   yy=s.st.forward_solution(th,xx,bb).numpy()
   rr=s.residual_nl(th,xx,ff,ll,bb,bt).numpy()
  register(seed,name,newton_correction(yy,rr,lam,beta,x,2))
 print('MODEL',seed,flush=True)

df=pd.DataFrame(records);df.to_csv(out/'correction_fresh_results.csv',index=False)
rows=[]
nodes=-np.cos(np.pi*np.arange(128)/127)
for i,row in enumerate(meta):
 poly={tuple(tuple(a) for a in atoms):Fraction(num,den) for atoms,num,den in row['ir']}
 fgrid=evaluate(poly,nodes,s.st.DESC)
 y,diag=s.cheb_solve(nodes,fgrid,row['lambda'],row['beta'],row['boundary'],x,degree=32)
 rows.append({'task':i,'group':row['group'],'mse_float64':float(np.mean((y-true[i])**2)),
              'mse_quantized_fp32':float(np.mean((y.astype(np.float32).astype(float)-true[i])**2)),
              'optimizer_success':diag['success']})
pd.DataFrame(rows).to_csv(out/'cheb_lobatto_fresh_results.csv',index=False)
print('NEW held-out summary:')
for group in ['structure','unseen_leaf','coefficient_ood']:
 z=df[df.group==group]
 print(group)
 for name,k in [('Constant',2),('Constant',3),('DirectFinal',1),('Meta',1)]:
  q=z[(z.method==name)&(z['round']==k)]
  print(name,k,'median',q.mse.median(),'hit1e-5',float((q.mse<1e-5).mean()))
 y=pd.DataFrame(rows)
 if group:
  y=pd.read_csv(out/'cheb_lobatto_fresh_results.csv');q=y[y.group==group]
  print('Cheb median',q.mse_float64.median(),'max',q.mse_float64.max())
