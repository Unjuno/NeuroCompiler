"""Exploratory deterministic residual-Newton correction of NN ODE solutions.
Post-hoc branch: samples the 129 evenly spaced eval grid as collocation. This differs
from Stage-8 support-only Chebyshev baseline and is NOT a matched-compute comparison.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd,torch,json,time,sys
from pathlib import Path
import stage8_nonlinear as s
st=s.st
root=s.ROOT

def newton_correction(u0,r0,lam,beta,x,rounds=2):
    """Integrate linear error IVP by exponential midpoint; iterate nonlinear Newton residual.
    r0 is physical residual of initial u0 at all sample nodes; initial boundary holds.
    """
    u=np.asarray(u0,dtype=np.float64).copy()
    r=np.asarray(r0,dtype=np.float64).copy()
    lam=np.asarray(lam,dtype=float);beta=np.asarray(beta,dtype=float)
    h=np.diff(x)
    ans=[u.copy()]
    for it in range(rounds):
        a=lam[:,None]+3*beta[:,None]*u*u
        delta=np.zeros_like(u)
        for j,dx in enumerate(h):
            am=0.5*(a[:,j]+a[:,j+1]);rm=0.5*(r[:,j]+r[:,j+1])
            w=np.exp(-am*dx)
            phi=-np.expm1(-am*dx)/np.maximum(am,1e-15)
            delta[:,j+1]=w*delta[:,j]-rm*phi
        r=beta[:,None]*(3*u*delta*delta+delta**3)
        u=u+delta
        ans.append(u.copy())
    return ans

def calc(seed):
    torch.manual_seed(seed)
    test=s.load('test',True)
    raw=json.loads((root/'banks'/'test.json').read_text())['tasks']
    n=len(raw)
    x=test['xq'].numpy().astype(float)[0]
    lam=test['lam'].numpy().astype(float)
    beta=test['beta'].numpy().astype(float)
    b=test['b'].numpy().astype(float)
    true=test['solution'].numpy()
    out=[]
    for method in ['Constant','DirectFinal','Meta','RegMeta']:
        if method=='Constant':
            u=np.broadcast_to(b[:,None],(n,len(x))).copy()
            r=lam[:,None]*u+beta[:,None]*u**3-test['fq'].numpy().astype(float)
        else:
            model=s.HyperNL().eval()
            state=torch.load(root/'seeds'/str(seed)/({'DirectFinal':'direct','Meta':'meta','RegMeta':'regmeta'}[method]+'.pt'),map_location='cpu',weights_only=True)
            model.load_state_dict(state)
            with torch.no_grad():
                theta=model(s.sub(test,slice(None))).double()
                x64=torch.from_numpy(test['xq'].numpy().astype(float))
                f64=torch.from_numpy(test['fq'].numpy().astype(float))
                b64=test['b'].double();lam64=test['lam'].double();beta64=test['beta'].double()
                u=st.forward_solution(theta,x64,b64).numpy()
                r=s.residual_nl(theta,x64,f64,lam64,b64,beta64).numpy()
        t0=time.perf_counter()
        iterations=newton_correction(u,r,lam,beta,x,rounds=3)
        duration=time.perf_counter()-t0
        for nstep,pred in enumerate(iterations):
            mse=np.mean((pred-true)**2,axis=1)
            for i,z in enumerate(raw):
                out.append({'seed':seed,'method':method,'newton_round':nstep,'task':i,'group':z['group'],
                            'solution_mse':float(mse[i]),'correction_batch_seconds':duration/3 if nstep>0 else 0.0})
        print(seed,method,'med mse:',[round(float(np.median(np.mean((y-true)**2,axis=1))),12) for y in iterations],flush=True)
    df=pd.DataFrame(out); dest=root/'stage8c_exploratory';dest.mkdir(exist_ok=True)
    df.to_csv(dest/f'correction_seed{seed}.csv',index=False)

if __name__=='__main__':calc(int(sys.argv[1]))
