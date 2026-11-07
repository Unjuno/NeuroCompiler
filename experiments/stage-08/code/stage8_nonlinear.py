"""Stage 8: residual-only nonlinear ODE experiment, independent of the original Stage 7 bank.
Requires numpy scipy pandas torch. All reference solutions held separate from learning banks.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import sys, time, json, copy, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
SRC=Path(__file__).parent/'neurocompiler_stage7'/'code'
sys.path.insert(0,str(SRC))
import stage7_experiment as st
from differential_ir import canonical, evaluate, ir_hash, serialize

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
torch.set_flush_denormal(True)
ROOT=Path(__file__).parent/'stage8_output'
(ROOT/'banks').mkdir(parents=True,exist_ok=True)
(ROOT/'seeds').mkdir(parents=True,exist_ok=True)
(ROOT/'analysis').mkdir(parents=True,exist_ok=True)
H,P,E=32,97,32
SEEDS=list(range(301,311))
K=(0,1,2,5,10,20,50,100)
LR_LIST=[.0003,.001,.003,.01,.03]

class HyperNL(torch.nn.Module):
    def __init__(self,blind=False):
        super().__init__(); self.blind=blind
        self.factor=torch.nn.Sequential(torch.nn.Linear(5,E),torch.nn.Tanh(),torch.nn.Linear(E,E),torch.nn.Tanh())
        self.term=torch.nn.Sequential(torch.nn.Linear(E+1,E),torch.nn.Tanh(),torch.nn.Linear(E,E),torch.nn.Tanh())
        self.root=torch.nn.Sequential(torch.nn.Linear(E+3,64),torch.nn.Tanh(),torch.nn.Linear(64,P))
        base=torch.cat([torch.empty(H).uniform_(-.7,.7),torch.empty(H).uniform_(-.2,.2),torch.randn(H)*.05,torch.zeros(1)])
        self.base=torch.nn.Parameter(base)
        torch.nn.init.normal_(self.root[-1].weight,std=.005);torch.nn.init.zeros_(self.root[-1].bias)
    def forward(self,bank):
        if self.blind:return self.base[None].expand(len(bank['b']),-1)
        m=bank['factor_mask']
        z=(self.factor(bank['desc'])*m[...,None]).sum(2)
        z=self.term(torch.cat([z,m.sum(2)[...,None]/3],-1))
        z=(z*bank['coeff'][...,None]).sum(1)
        eq=torch.stack([torch.log1p(bank['lam']), bank['beta'], bank['b']],1)
        return self.base[None]+self.root(torch.cat([z,eq],1))


def residual_nl(th,x,f,lam,b,beta):
    w,d,a,e=th[:,:H],th[:,H:2*H],th[:,2*H:3*H],th[:,3*H:]
    h=torch.tanh(x[...,None]*w[:,None]+d[:,None]);s=1-h*h
    v=(h*a[:,None]).sum(-1)+e
    vx=(s*(w*a)[:,None]).sum(-1)
    uu=b[:,None]+(x+1)*v
    return v+(x+1)*vx+lam[:,None]*uu+beta[:,None]*uu**3-f


def gradient_nl(th,x,f,lam,b,beta,scale):
    w,d,a,e=th[:,:H],th[:,H:2*H],th[:,2*H:3*H],th[:,3*H:]
    xx=x[...,None];t=(x+1)[...,None]
    h=torch.tanh(xx*w[:,None]+d[:,None]);s=1-h*h
    v=(h*a[:,None]).sum(-1)+e
    vx=(s*(w*a)[:,None]).sum(-1)
    u=b[:,None]+(x+1)*v
    r=v+(x+1)*vx+lam[:,None]*u+beta[:,None]*u**3-f
    A=1+t*(lam[:,None,None]+3*beta[:,None,None]*u[...,None].square())
    dr=(2*r/(x.shape[1]*scale[:,None]))[...,None]
    jw=a[:,None]*s*(A*xx+t*(1-2*w[:,None]*h*xx))
    jd=a[:,None]*s*(A-2*t*w[:,None]*h)
    ja=A*h+t*w[:,None]*s
    return torch.cat([(dr*jw).sum(1),(dr*jd).sum(1),(dr*ja).sum(1),(dr*A).sum(1)],1)


def test_gradient():
    torch.manual_seed(4561)
    th=(torch.randn(4,97,dtype=torch.float64)*.14).requires_grad_()
    x=torch.rand(4,21,dtype=torch.float64)*2-1
    f=torch.randn(4,21,dtype=torch.float64)
    lam=torch.tensor([0.,.7,1.5,3.],dtype=torch.float64)
    beta=torch.tensor([.2,.5,1.2,2.],dtype=torch.float64)
    b=torch.tensor([.3,-.2,.7,-.7],dtype=torch.float64)
    s=torch.rand(4,dtype=torch.float64)+1
    loss=(residual_nl(th,x,f,lam,b,beta).square().mean(1)/s).sum()
    ga=torch.autograd.grad(loss,th)[0]
    diff=float(torch.max(torch.abs(ga-gradient_nl(th.detach(),x,f,lam,b,beta,s))))
    assert diff < 1e-10, diff
    # boundary construction remains exact in float64
    bc=st.forward_solution(th.detach(), -torch.ones((4,1),dtype=torch.float64),b)
    assert float((bc[:,0]-b).abs().max())==0
    return {'max_abs_gradient_disagreement_float64':diff,'boundary_error_float64':0.0}


def ode_solution(poly,lam,beta,b,x,rtol=2e-11,atol=1e-12):
    x=np.asarray(x,dtype=np.float64)
    def rhs(t,y):
        return [float(evaluate(poly,np.array([t]),st.DESC)[0])-lam*y[0]-beta*y[0]**3]
    sol=solve_ivp(rhs,(-1,1),[b],method='DOP853',rtol=rtol,atol=atol,dense_output=True)
    if not sol.success: raise RuntimeError(sol.message)
    return sol.sol(x)[0]


def generate(name,n,seed,group=None):
    rs=np.random.default_rng(seed)
    polys=[];rows=[];lam=[];beta=[];b=[];xs=[];fs=[];xq=[];fq=[];solutions=[]
    Q=129
    xv=np.linspace(-1,1,Q)
    for i in range(n):
        g='train' if group is None else group[i%len(group)]
        if g=='train':
            pool=range(10);kind=str(rs.choice(st.TRAIN_KINDS));ll=float(rs.choice([0.,.5,1.,2.])); bb=float(rs.choice([.25,.6,1.,1.5]))
        else:
            kind=st.TEST_KINDS[(i//len(group))%len(st.TEST_KINDS)]
            pool=range(10,14) if g=='unseen_leaf' else range(10)
            ll=float(rs.choice([.25,.75,1.5])) if g!='coefficient_ood' else float(rs.choice([3.,5.]))
            bb=float(rs.choice([.4,1.25])) if g!='coefficient_ood' else float(rs.choice([2.,3.]))
        for trial in range(15000):
            poly=canonical(st.forcing(kind,rs,pool));power=float(np.mean(evaluate(poly,xv,st.DESC)**2))
            if power>.003:break
        else:raise ValueError('no nondegenerate expression')
        boundary=float(rs.uniform(-.6,.6))
        xx=rs.uniform(-1,1,size=128);xqq=rs.uniform(-1,1,size=128) if g=='train' else xv
        polys.append(poly);lam.append(ll);beta.append(bb);b.append(boundary)
        xs.append(xx);fs.append(evaluate(poly,xx,st.DESC))
        xq.append(xqq);fq.append(evaluate(poly,xqq,st.DESC))
        if name not in ('train','validation'):
            solutions.append(ode_solution(poly,ll,bb,boundary,xqq))
        rows.append({'task':i,'group':g,'kind':kind,'lambda':ll,'beta':bb,'boundary':boundary,
                     'forcing_power':power,'ir_hash':ir_hash(poly),'ir':serialize(poly)})
    d,m,c=st.pack_ir(polys)
    data=dict(desc=d,factor_mask=m,coeff=c,lam=np.array(lam,np.float32),beta=np.array(beta,np.float32),
              b=np.array(b,np.float32),xs=np.array(xs,np.float32),fs=np.array(fs,np.float32),
              xq=np.array(xq,np.float32),fq=np.array(fq,np.float32))
    data['scale']=(1+np.array([r['forcing_power'] for r in rows])+np.square(data['lam']*data['b']+data['beta']*data['b']**3)).astype(np.float32)
    np.savez_compressed(ROOT/'banks'/f'{name}.npz',**data)
    if solutions:
        np.savez_compressed(ROOT/'banks'/f'{name}_EVAL_ONLY.npz',solution=np.array(solutions,np.float64))
    (ROOT/'banks'/f'{name}.json').write_text(json.dumps({'name':name,'seed':seed,'task_count':n,'tasks':rows},indent=1))
    print('Bank',name,n,'groups',sorted({r['group'] for r in rows}),flush=True)


def make_banks():
    generate('train',1152,823001)
    generate('validation',96,823002)
    generate('development',72,823003,['structure','unseen_leaf','coefficient_ood'])
    generate('test',240,823004,['structure','unseen_leaf','coefficient_ood'])
    train=json.loads((ROOT/'banks'/'train.json').read_text())['tasks']
    vals=json.loads((ROOT/'banks'/'validation.json').read_text())['tasks']
    tests=json.loads((ROOT/'banks'/'test.json').read_text())['tasks']
    train_ir=set(x['ir_hash'] for x in train)
    final_ir=set(x['ir_hash'] for x in tests)
    common=len(train_ir & final_ir)
    assert common==0, common
    detail={'nonlinear_gradient':test_gradient(), 'train_test_ir_overlap':common,
            'train_tasks':len(train),'validation_tasks':len(vals),'test_tasks':len(tests),
            'test_seed':823004,'no_reference_solution_in_train_or_validation':True}
    (ROOT/'analysis'/'checks.json').write_text(json.dumps(detail,indent=2))


def load(name,include_reference=False):
    z=np.load(ROOT/'banks'/f'{name}.npz');out={k:torch.from_numpy(z[k]) for k in z.files}
    out['meta']=json.loads((ROOT/'banks'/f'{name}.json').read_text())['tasks']
    if include_reference and (ROOT/'banks'/f'{name}_EVAL_ONLY.npz').exists():
        out['solution']=torch.tensor(np.load(ROOT/'banks'/f'{name}_EVAL_ONLY.npz')['solution'])
    return out


def sub(b,idx):return {k:v[idx] for k,v in b.items() if k!='meta'}

def metrics(th,b):
    r=residual_nl(th,b['xq'],b['fq'],b['lam'],b['b'],b['beta'])
    ans={'residual':r.square().mean(1),'norm_residual':r.square().mean(1)/b['scale']}
    if 'solution' in b:
        err=st.forward_solution(th,b['xq'],b['b'])-b['solution']
        ans['solution_mse']=err.square().mean(1)
    return ans


def trajectory(th,b,lr,seed,last=100,record=(0,1,2,5,10,20,50,100),policy='backtrack'):
    """All tasks share minibatch indices. Support-only Armijo-style monotone backtracking.
    No reference solution/query labels are inspected. Zero update if ten halvings fail.
    """
    rng=torch.Generator().manual_seed(seed)
    th=th.detach().clone()
    halves=torch.zeros(len(th),dtype=torch.float32)
    stalls=torch.zeros(len(th),dtype=torch.float32)
    def snapshot():
        v={k:z.detach().cpu().numpy() for k,z in metrics(th,b).items()}
        v['line_search_halvings']=halves.detach().cpu().numpy()
        v['line_search_stalls']=stalls.detach().cpu().numpy()
        return v
    saved={0:snapshot()}
    for k in range(1,last+1):
        j=torch.randint(b['xs'].shape[1],(len(th),32),generator=rng)
        x=b['xs'].gather(1,j);f=b['fs'].gather(1,j)
        with torch.no_grad():
            gg=gradient_nl(th,x,f,b['lam'],b['b'],b['beta'],b['scale'])
            gn=gg.norm(dim=1,keepdim=True).clamp_min(1e-9)
            gg=gg*(30./gn).clamp(max=1.)
            if policy=='backtrack':
                base=residual_nl(th,x,f,b['lam'],b['b'],b['beta']).square().mean(1)/b['scale']
                factor=torch.ones(len(th))
                accepted=torch.zeros(len(th),dtype=torch.bool)
                result=th.clone()
                for attempt in range(11):
                    new=th-lr*factor[:,None]*gg
                    curr=residual_nl(new,x,f,b['lam'],b['b'],b['beta']).square().mean(1)/b['scale']
                    ok=(~accepted)&torch.isfinite(curr)&(curr<=base+1e-12)
                    result=torch.where(ok[:,None],new,result)
                    accepted|=ok
                    if accepted.all():break
                    halving=(~accepted)
                    halves+=halving.float()
                    factor=torch.where(halving,factor*.5,factor)
                stalls+=(~accepted).float()
                th=result
            else:
                new=th-lr*gg
                ok=torch.isfinite(new).all(1)
                stalls+=(~ok).float()
                th=torch.where(ok[:,None],new,th)
        if k in record:saved[k]=snapshot()
    return saved


def validation_lr(th,b,seed):
    rows=[]
    for lr in LR_LIST:
        cv=trajectory(th,b,lr,seed,10)
        score=float(np.mean([np.log(cv[k]['norm_residual']+1e-12).mean() for k in (1,2,5,10)]))
        rows.append({'lr':lr,'log_residual_score':score})
    valid=[r for r in rows if np.isfinite(r['log_residual_score'])]
    return min(valid,key=lambda q:q['log_residual_score'])['lr'] if valid else min(LR_LIST), rows


def train(seed,mode,direct_steps=1200,meta_steps=300,horizon=3,inner_lr=.003,anchor=.25):
    torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+999987)
    b=load('train');model=HyperNL(blind=False)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    n=direct_steps if mode=='direct' else meta_steps
    snapshots={0:copy.deepcopy(model.state_dict())}
    t0=time.perf_counter()
    train_explosions=0
    for it in range(1,n+1):
        ix=torch.randint(len(b['b']),(16,),generator=rng)
        batch=sub(b,ix);th=model(batch);adapted=th.detach().clone()
        if mode!='direct':
            for _ in range(horizon):
                jj=torch.randint(128,(16,32),generator=rng)
                with torch.no_grad():
                    grad=gradient_nl(adapted,batch['xs'].gather(1,jj),batch['fs'].gather(1,jj),batch['lam'],batch['b'],batch['beta'],batch['scale'])
                    # Safe gradient: clip per task for nonlinear residuals
                    gn=grad.norm(dim=1,keepdim=True).clamp_min(1e-8)
                    grad=grad*(30./gn).clamp(max=1.)
                    adapted=adapted-inner_lr*grad
        jj=torch.randint(128,(16,64),generator=rng)
        with torch.no_grad():
            grad=gradient_nl(adapted,batch['xq'].gather(1,jj),batch['fq'].gather(1,jj),batch['lam'],batch['b'],batch['beta'],batch['scale'])/16
            if mode=='regmeta':
                g0=gradient_nl(th.detach(),batch['xq'].gather(1,jj),batch['fq'].gather(1,jj),batch['lam'],batch['b'],batch['beta'],batch['scale'])/16
                grad=(grad+anchor*g0)/(1+anchor)
        if not torch.isfinite(grad).all():
            train_explosions+=1
            continue
        opt.zero_grad(set_to_none=True);th.backward(grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if it%50==0 or it==n:
            snapshots[it]=copy.deepcopy(model.state_dict())
    return model,snapshots,time.perf_counter()-t0,train_explosions


def evaluate_seed(seed,direct_steps=1200,meta_steps=300,horizon=3,inner_lr=.003):
    outdir=ROOT/'seeds'/f'{seed:03d}';outdir.mkdir(parents=True,exist_ok=True)
    done=outdir/'metrics.csv'
    if done.exists():
        print('already done',seed,flush=True);return
    models={};snapshots={};training=[]
    val=load('validation');test=load('test',include_reference=True)
    for mode in ('direct','meta','regmeta'):
        model,ck,secs,explosions=train(seed,mode,direct_steps,meta_steps,horizon,inner_lr)
        model.eval();models[mode]=model;snapshots[mode]=ck
        training.append({'method':mode,'training_seconds':secs,'outer_steps':direct_steps if mode=='direct' else meta_steps,
                         'horizon':0 if mode=='direct' else horizon, 'gradient_evaluations_per_task':direct_steps if mode=='direct' else meta_steps*(horizon+1+(mode=='regmeta')),
                         'train_explosions':explosions})
        torch.save(model.state_dict(),outdir/f'{mode}.pt')
        print('seed',seed,mode,'train_s',round(secs,2),'errors',explosions,flush=True)
    with torch.no_grad():
        vals=[]
        for step,state in snapshots['direct'].items():
            candidate=HyperNL();candidate.load_state_dict(state)
            vals.append((float(metrics(candidate(val),val)['norm_residual'].mean()),step))
        _,best=min(vals)
        direct=HyperNL();direct.load_state_dict(snapshots['direct'][best]);direct.eval()
        random_seed=seed+300_000;torch.manual_seed(random_seed)
        random=HyperNL(blind=True).eval()
        available={'DirectBest':direct,'Meta':models['meta'],'RegMeta':models['regmeta'],'Random':random}
        selection=[];result=[]
        for name,mm in available.items():
            thv=mm(val);lr,candidates=validation_lr(thv,val,921800+seed)
            tt=time.perf_counter();thr=mm(sub(test,slice(None)));forward_sec=time.perf_counter()-tt
            tt=time.perf_counter();cv=trajectory(thr,test,lr,950000+seed,last=100)
            adapt_sec=time.perf_counter()-tt
            selection.append({'method':name,'best_validation_checkpoint':best if name=='DirectBest' else -1,
                              'best_lr':lr,'forward_seconds_total':forward_sec,'adapt100_seconds_total':adapt_sec,
                              'validation_scores':candidates})
            for kk,entry in cv.items():
                for ti,row in enumerate(test['meta']):
                    result.append({'seed':seed,'method':name,'step':kk,'task':ti,'group':row['group'],'kind':row['kind'],
                                   'lambda':row['lambda'],'beta':row['beta'],
                                   **{metric:float(q[ti]) for metric,q in entry.items()}})
    pd.DataFrame(result).to_csv(done,index=False)
    pd.DataFrame(training).to_csv(outdir/'training.csv',index=False)
    (outdir/'selection.json').write_text(json.dumps(selection,indent=1))
    print('seed',seed,'written rows',len(result),flush=True)


def cheb_matrices(x,degree):
    x=np.asarray(x,dtype=np.float64)
    xx=np.polynomial.chebyshev.chebvander(x,degree-1)
    tt=np.empty_like(xx)
    for i in range(degree):
        c=np.zeros(i+1);c[-1]=1
        dd=np.polynomial.chebyshev.chebder(c)
        tt[:,i]=np.polynomial.chebyshev.chebval(x,dd) if len(dd)>0 else 0.
    t=x+1
    return xx*t[:,None], xx+t[:,None]*tt


def cheb_solve(x_coll,f_coll,lam,beta,b,x_eval,degree=16):
    basis,dbasis=cheb_matrices(x_coll,degree)
    def fun(w):
        u=b+basis@w
        return dbasis@w+lam*u+beta*u**3-f_coll
    def jac(w):
        u=b+basis@w
        return dbasis+(lam+3*beta*u*u)[:,None]*basis
    opt=least_squares(fun,np.zeros(degree),jac=jac,method='lm',max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    bx,_=cheb_matrices(x_eval,degree)
    return b+bx@opt.x,{'success':bool(opt.success),'nfev':int(opt.nfev),'cost':float(opt.cost)}


def baseline_bank(nval=48,ntest=240,degrees=(8,12,20,32)):
    # No reference solution enters the selection objective.
    val=load('development');test=load('test',True)
    vb=json.loads((ROOT/'banks'/'development.json').read_text())['tasks']
    tb=json.loads((ROOT/'banks'/'test.json').read_text())['tasks']
    def one(bank,rows,idx,degree):
        row=rows[idx];poly={tuple(tuple(i) for i in a):__import__('fractions').Fraction(b,c) for a,b,c in row['ir']}
        xx=bank['xs'][idx].numpy().astype(np.float64)
        fx=bank['fs'][idx].numpy().astype(np.float64)
        x_eval=bank['xq'][idx].numpy().astype(np.float64)
        ll,bb,bi=row['lambda'],row['beta'],row['boundary']
        t0=time.perf_counter()
        y,diag=cheb_solve(xx,fx,ll,bb,bi,x_eval,degree)
        elapsed=time.perf_counter()-t0
        r=np.gradient(y,xx) if False else None
        # collocation residual (using exact basis derivative, from least squares cost)
        score=2*diag['cost']/len(xx)
        return {'mse':float(np.mean((y-bank['solution'][idx].numpy())**2)) if 'solution' in bank else np.nan,
                'residual':score,'wall_s':elapsed,'degree':degree,'success':diag['success'],'nfev':diag['nfev']}
    # choose best degree by development residual. This is a new, exploratory baseline control.
    # Need correct reference-free development selection.
    v=load('development')
    scores={}
    for d in degrees:
        scores[d]=np.median([one(v,vb,i,d)['residual'] for i in range(len(vb))])
    best=min(scores,key=scores.get)
    outcomes=[]
    for i,row in enumerate(tb):
        item=one(test,tb,i,best)
        outcomes.append({'task':i,'group':row['group'],'kind':row['kind'],**item})
    pd.DataFrame(outcomes).to_csv(ROOT/'analysis'/'cheb_classical.csv',index=False)
    (ROOT/'analysis'/'cheb_selection.json').write_text(json.dumps({'development_scores':scores,'degree':best,'selection':'median development support-only collocation residual','test_not_used_for_selection':True},indent=2))
    print('Chebyshev selection',best,'test median mse',np.median([x['mse'] for x in outcomes]),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['banks','run','classical','gradient'])
    p.add_argument('--seed',type=int,default=301);p.add_argument('--direct-steps',type=int,default=1200)
    p.add_argument('--meta-steps',type=int,default=300);p.add_argument('--horizon',type=int,default=3)
    p.add_argument('--inner-lr',type=float,default=.003)
    a=p.parse_args()
    if a.action=='banks':make_banks()
    if a.action=='gradient':print(test_gradient())
    if a.action=='run':evaluate_seed(a.seed,a.direct_steps,a.meta_steps,a.horizon,a.inner_lr)
    if a.action=='classical':baseline_bank()
if __name__=='__main__':main()
