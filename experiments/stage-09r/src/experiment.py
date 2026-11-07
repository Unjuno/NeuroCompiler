"""Stage9R: reproducible independent PDE / first-principles learning experiment.
All continuous claims are distinguished from finite-grid estimates. CPU only.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,sys,time,platform
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import dstn
from scipy.sparse.linalg import LinearOperator,cg
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(True)

@lru_cache(maxsize=16)
def basis(m:int,n:int):
    if not 1<=m<=n: raise ValueError('1<=m<=n required')
    z=np.arange(1,n+1,dtype=float)/(n+1); x,y=np.meshgrid(z,z,indexing='ij'); x=x.ravel(); y=y.ravel()
    i,j=np.meshgrid(np.arange(1,m+1),np.arange(1,m+1),indexing='ij'); i=i.ravel(); j=j.ravel()
    phi=2*np.sin(np.pi*x[:,None]*i)*np.sin(np.pi*y[:,None]*j)
    return phi,np.pi**2*(i*i+j*j),x,y,1./(n+1)**2

def fields(p,n):
    p=np.asarray(p,dtype=float)
    if p.shape!=(11,) or not np.isfinite(p).all(): raise ValueError('11 finite parameters required')
    if p[2]<=0 or p[6]<=0 or p[8]<0 or abs(p[9])>1 or p[10]<0: raise ValueError('invalid elliptic parameters')
    z=np.arange(1,n+1)/(n+1); x,y=np.meshgrid(z,z,indexing='ij')
    g1=np.exp(-((x-p[0])**2+(y-p[1])**2)/(2*p[2]**2)); g2=np.exp(-((x-p[4])**2+(y-p[5])**2)/(2*p[6]**2))
    f=p[3]*g1+p[7]*g2; k=p[8]*(1+p[9]*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))
    return f,k,np.stack((p[3]*g1,p[7]*g2))

def energy_residual(a,phi,lam,K,F,beta,w):
    u=phi@a
    E=.5*np.dot(lam*a,a)+.5*a@K@a+beta*w*np.sum(u**4)/4-F@a
    R=lam*a+K@a+beta*w*(phi.T@(u**3))-F
    J=np.diag(lam)+K+3*beta*w*((phi.T*u**2)@phi)
    return E,R,J

def solve_galerkin(phi,lam,K,F,beta,w,a0=None,tol=2e-11,maxiter=30):
    a=np.zeros_like(F) if a0 is None else np.array(a0,dtype=float,copy=True); scale=max(np.linalg.norm(F),1.)
    for it in range(maxiter+1):
        E,R,J=energy_residual(a,phi,lam,K,F,beta,w); rel=np.linalg.norm(R)/scale
        if rel<tol: return a,dict(converged=True,iterations=it,relres=float(rel))
        if it==maxiter: break
        d=np.linalg.solve(J,-R); alpha=1.
        for ls in range(25):
            b=a+alpha*d; Eb,Rb,_=energy_residual(b,phi,lam,K,F,beta,w)
            if Eb<=E+1e-4*alpha*(R@d)+1e-14*max(1.,abs(E)) or np.linalg.norm(Rb)<tol*scale: break
            alpha*=.5
        a=b
    return a,dict(converged=False,iterations=maxiter,relres=float(rel))

def solve_reference(f,k,beta,tol=2e-11,maxiter=30):
    f=np.asarray(f,dtype=float); k=np.broadcast_to(np.asarray(k,dtype=float),f.shape); n=f.shape[0]
    if f.shape!=(n,n) or np.min(k)<0 or beta<0: raise ValueError('square field; nonnegative coefficients')
    q=np.arange(1,n+1,dtype=float); lam=np.pi**2*(q[:,None]**2+q[None,:]**2)
    def tr(a): return dstn(a,type=1,norm='ortho')
    def lap(a): return tr(lam*tr(a))
    def energy(a): return .5*np.sum(a*lap(a))+.5*np.sum(k*a*a)+beta*np.sum(a**4)/4-np.sum(f*a)
    u=tr(tr(f)/(lam+k.mean())); den=max(np.linalg.norm(f),1.); total=0
    for it in range(maxiter+1):
        R=lap(u)+k*u+beta*u**3-f; rel=np.linalg.norm(R)/den
        if rel<tol: return u,dict(converged=True,iterations=it,relres=float(rel),cg_iterations=total)
        if it==maxiter: break
        v=k+3*beta*u**2
        A=LinearOperator((n*n,n*n),matvec=lambda d:(lap(d.reshape(n,n))+v*d.reshape(n,n)).ravel(),dtype=float)
        M=LinearOperator((n*n,n*n),matvec=lambda d:tr(tr(d.reshape(n,n))/(lam+v.mean())).ravel(),dtype=float)
        count=[0]
        def cb(_): count[0]+=1
        d,flag=cg(A,-R.ravel(),M=M,rtol=1e-10,atol=0,maxiter=150,callback=cb); total+=count[0]
        if flag!=0: return u,dict(converged=False,iterations=it,relres=float(rel),cg_failure=int(flag))
        d=d.reshape(n,n); E=energy(u); rd=np.sum(R*d); alpha=1.
        for ls in range(25):
            cand=u+alpha*d
            if energy(cand)<=E+1e-4*alpha*rd+1e-13*max(1.,abs(E)): break
            alpha*=.5
        u=cand
    return u,dict(converged=False,iterations=maxiter,relres=float(rel),cg_iterations=total)

def sample_params(n,seed,group='id'):
    rng=np.random.default_rng(seed); p=np.empty((n,11))
    for off in (0,4):
        p[:,off:off+2]=rng.uniform(.25,.75,(n,2)); p[:,off+2]=rng.uniform(.10,.18,n); p[:,off+3]=rng.uniform(10.,50.,n)
        if group=='center':
            side=rng.integers(0,2,(n,2)); low=rng.uniform(.06,.20,(n,2)); p[:,off:off+2]=np.where(side==0,low,1-low)
        if group=='width': p[:,off+2]=rng.uniform(.04,.07,n)
    p[:,8]=rng.uniform(0,20,n); p[:,9]=rng.uniform(0,.8,n); p[:,10]=rng.uniform(0,80,n)
    if group=='operator': p[:,8]=rng.uniform(25,50,n); p[:,10]=rng.uniform(120,240,n)
    for row in p:
        if tuple(row[:4])>tuple(row[4:8]): row[:8]=np.r_[row[4:8].copy(),row[:4].copy()]
    return p

def make_bank(name,p,groups=None):
    ph,lam,_,_,w=basis(6,23); pr,_,_,_,wr=basis(6,63)
    C=[]; CG=[]; norm=[]; floor=[]; Ks=[]; Fs=[]; Fsrc=[]; checks=[]; high=[]; t=time.perf_counter()
    for i,row in enumerate(p):
        f,k,_=fields(row,63); u,info=solve_reference(f,k,row[-1])
        if not info['converged']: raise RuntimeError(('reference',name,i,info))
        coef=dstn(u,type=1,norm='ortho')/64; c=coef[:6,:6].ravel()
        C.append(c); norm.append(np.sum(coef**2)); floor.append(np.sum(coef**2)-np.sum(c*c)); high.append(coef)
        fq,kq,src=fields(row,23); Kq=(ph.T*kq.ravel())@ph*w; Fq=ph.T@fq.ravel()*w
        Ks.append(Kq); Fs.append(Fq); Fsrc.append(src.reshape(2,-1)@ph*w)
        Kr=(pr.T*k.ravel())@pr*wr; Fr=pr.T@f.ravel()*wr
        a,gi=solve_galerkin(pr,lam,Kr,Fr,row[-1],wr)
        if not gi['converged']: raise RuntimeError(('galerkin',name,i,gi))
        CG.append(a); checks.append([info['relres'],gi['relres'],np.max(np.abs(Fq-Fr)),np.max(np.abs(Kq-Kr))])
        if (i+1)%128==0: print(name,i+1,round(time.perf_counter()-t,2),flush=True)
    out=ROOT/'data'/f'{name}.npz'
    np.savez_compressed(out,params=p,coeff=C,galerkin=CG,norm2=norm,floor2=np.maximum(floor,0),K=Ks,F=Fs,Fsrc=Fsrc,numerics=checks,ref_spectral=high,groups=np.array(groups if groups is not None else [name]*len(p)))
    print('BANK DONE',name,len(p),time.perf_counter()-t,hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)

class Compiler(nn.Module):
    def __init__(self,mean,std,scale):
        super().__init__()
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32)); self.register_buffer('std',torch.tensor(std,dtype=torch.float32)); self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.layers=nn.ModuleList([nn.Linear(11,64),nn.Linear(64,64),nn.Linear(64,36)])
        with torch.no_grad(): self.layers[-1].weight.mul_(.1); self.layers[-1].bias.zero_()
    def forward(self,p):
        h=(p-self.mean)/self.std
        for layer in self.layers[:-1]: h=torch.tanh(layer(h))
        return self.layers[-1](h)*self.scale
    def with_tangent(self,p,d):
        h=(p-self.mean)/self.std; dh=d/self.std
        for layer in self.layers[:-1]:
            h=torch.tanh(layer(h)); dh=torch.nn.functional.linear(dh,layer.weight)*(1-h*h)
        return self.layers[-1](h)*self.scale,torch.nn.functional.linear(dh,self.layers[-1].weight)*self.scale

def residual(a,K,F,beta,phi,lam,w):
    u=a@phi.T
    return a*lam+torch.bmm(K,a.unsqueeze(-1)).squeeze(-1)+beta[:,None]*((u**3)@phi)*w-F

def tangent(a,da,K,beta,phi,lam,w):
    u=a@phi.T; du=da@phi.T
    return da*lam+torch.bmm(K,da.unsqueeze(-1)).squeeze(-1)+3*beta[:,None]*((u*u*du)@phi)*w

def tensor_bank(name):
    z=np.load(ROOT/'data'/f'{name}.npz')
    return {k:torch.tensor(z[k],dtype=torch.float32) for k in ['params','coeff','norm2','K','F','Fsrc']}

def train_one(seed,method,steps=800,seconds=None):
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    tr=tensor_bank('train'); va=tensor_bank('val'); p=tr['params']
    model=Compiler(p.mean(0).numpy(),p.std(0).numpy(),tr['coeff'].std(0).numpy()+.001)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    ph,la,_,_,w=basis(6,23); phi=torch.tensor(ph,dtype=torch.float32); lam=torch.tensor(la,dtype=torch.float32)
    rng=np.random.default_rng(seed+70001); best=float('inf'); beststate=None; beststep=-1
    elapsed=0.; step=0; logs=[]; running=0.; ns=0; start=time.perf_counter()
    while True:
        if seconds is None and step>=steps: break
        if seconds is not None and step>=steps and elapsed>=seconds: break
        idx=torch.tensor(rng.integers(0,len(p),64)); choice=torch.tensor(rng.integers(0,2,64)); pb=p[idx]; y=tr['coeff'][idx]; nr=tr['norm2'][idx].clamp_min(1e-10)
        opt.zero_grad(set_to_none=True)
        if method=='Response':
            d=torch.zeros_like(pb); d[torch.arange(64),3+4*choice]=pb[torch.arange(64),3+4*choice]; a,da=model.with_tangent(pb,d)
        else: a=model(pb)
        loss=(((a-y)**2).sum(-1)/nr).mean()
        if method in ['Weak','Response']:
            K=tr['K'][idx]; F=tr['F'][idx]; beta=pb[:,-1]; pre=lam[None,:]+pb[:,8,None]
            R=residual(a,K,F,beta,phi,lam,w); loss=loss+(((R/pre)**2).sum(-1)/nr).mean()
            if method=='Response':
                T=tangent(a,da,K,beta,phi,lam,w)-tr['Fsrc'][idx,choice]
                loss=loss+.25*(((T/pre)**2).sum(-1)/nr).mean()
        if not torch.isfinite(loss): raise RuntimeError(('nonfinite train',seed,method,step))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.); opt.step(); step+=1
        running+=float(loss.detach()); ns+=1; elapsed=time.perf_counter()-start
        done=(seconds is None and step>=steps) or (seconds is not None and step>=steps and elapsed>=seconds)
        if step%100==0 or done:
            with torch.no_grad():
                err=((model(va['params'])-va['coeff'])**2).sum(-1)/va['norm2']; score=float(torch.log(err.clamp_min(1e-12)).mean())
            logs.append(dict(seed=seed,method=method,step=step,seconds=elapsed,train_loss=running/ns,val_log_coeferr=score)); running=0.; ns=0
            if score<best: best=score; beststate=copy.deepcopy(model.state_dict()); beststep=step
        if step>100000: raise RuntimeError('time budget runaway')
    elapsed=time.perf_counter()-start; model.load_state_dict(beststate)
    meta=dict(seed=seed,method=method,steps=step,best_step=beststep,train_seconds=elapsed,best_val_logcoeferr=best,parameters=sum(v.numel() for v in model.parameters()),threads=1)
    torch.save(dict(state=beststate,meta=meta),ROOT/'models'/f'{method}_{seed:02d}.pt')
    (ROOT/'models'/f'{method}_{seed:02d}.json').write_text(json.dumps(meta,indent=2))
    pd.DataFrame(logs).to_csv(ROOT/'analysis'/f'train_{method}_{seed:02d}.csv',index=False)
    return meta

def load_model(seed,method):
    z=torch.load(ROOT/'models'/f'{method}_{seed:02d}.pt',map_location='cpu',weights_only=False); s=z['state']
    m=Compiler(s['mean'].numpy(),s['std'].numpy(),s['scale'].numpy()); m.load_state_dict(s); m.eval(); return m,z['meta']

def verify():
    out={}; ph,la,x,y,w=basis(6,31); out['gram_max']=float(np.max(np.abs(ph.T@ph*w-np.eye(36))))
    rng=np.random.default_rng(4); u=rng.normal(size=len(x)); a=rng.normal(size=36); p=ph.T@u*w
    out['projection_identity_abs']=float(abs(np.sum((u-ph@a)**2)*w-np.sum((u-ph@p)**2)*w-np.sum((p-a)**2)))
    ph,la,x,y,w=basis(3,15); a=rng.normal(size=9)*.01; K=(ph.T*(3+np.sin(2*np.pi*x)*np.sin(2*np.pi*y)))@ph*w; F=ph.T@(20*np.exp(-10*((x-.5)**2+(y-.5)**2)))*w
    E,R,J=energy_residual(a,ph,la,K,F,20.,w); d=rng.normal(size=9); ep=1e-6
    e1,r1,_=energy_residual(a+ep*d,ph,la,K,F,20.,w); e2,r2,_=energy_residual(a-ep*d,ph,la,K,F,20.,w)
    out['energy_fd_abs']=float(abs((e1-e2)/(2*ep)-R@d)); out['hessian_fd_max']=float(np.max(np.abs((r1-r2)/(2*ep)-J@d))); out['hessian_min_eig']=float(np.linalg.eigvalsh(J).min())
    n=31; z=np.arange(1,n+1)/(n+1); xx,yy=np.meshgrid(z,z,indexing='ij'); exact=.3*np.sin(np.pi*xx)*np.sin(2*np.pi*yy); k=5+2*np.sin(2*np.pi*xx)*np.sin(2*np.pi*yy); f=5*np.pi**2*exact+k*exact+40*exact**3
    ur,info=solve_reference(f,k,40.,tol=1e-11); out['manufactured_max']=float(np.max(np.abs(ur-exact))); assert info['converged']
    torch.manual_seed(123); model=Compiler(np.zeros(11),np.ones(11),np.ones(36)).double(); q=torch.randn(3,11,dtype=torch.float64); dq=torch.randn_like(q)
    v,dv=model.with_tangent(q,dq); va,da=torch.autograd.functional.jvp(model,q,dq)
    out['nn_jvp_max']=float(torch.max(torch.abs(dv-da)).detach()); out['nn_forward_max']=float(torch.max(torch.abs(v-va)).detach())
    ph,la,x,y,w=basis(6,23); ph=torch.tensor(ph); la=torch.tensor(la); a=torch.randn(3,36,dtype=torch.float64)*.01; da=torch.randn_like(a); K=torch.eye(36,dtype=torch.float64).repeat(3,1,1)*3; F=torch.randn_like(a); beta=torch.tensor([0.,2.,30.],dtype=torch.float64); dF=torch.randn_like(a)
    T=tangent(a,da,K,beta,ph,la,w)-dF; rp=residual(a+ep*da,K,F+ep*dF,beta,ph,la,w); rm=residual(a-ep*da,K,F-ep*dF,beta,ph,la,w)
    out['physics_jvp_fd_max']=float(torch.max(torch.abs((rp-rm)/(2*ep)-T)))
    assert out['gram_max']<1e-12 and out['projection_identity_abs']<1e-11 and out['energy_fd_abs']<1e-7 and out['hessian_fd_max']<1e-7 and out['manufactured_max']<1e-10 and out['nn_jvp_max']<1e-12 and out['physics_jvp_fd_max']<1e-6
    out['all_pass']=True; (ROOT/'checks'/'numerical_tests.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['verify','bank','testbank','train']); ap.add_argument('--seed',type=int,default=100); ap.add_argument('--steps',type=int,default=800); args=ap.parse_args()
    if args.mode=='verify': verify()
    if args.mode=='bank': make_bank('train',sample_params(1024,80101)); make_bank('val',sample_params(128,80102))
    if args.mode=='testbank':
        ps=[]; labels=[]
        for i,g in enumerate(['id','center','width','operator']): ps.append(sample_params(200,90101+i,g)); labels += [g]*200
        make_bank('test',np.concatenate(ps),labels)
    if args.mode=='train':
        times={}
        for method in ['Data','Weak','Response','DataTime']:
            meta=train_one(args.seed,method,args.steps,seconds=times.get('Response') if method=='DataTime' else None); times[method]=meta['train_seconds']; print(json.dumps(meta),flush=True)
