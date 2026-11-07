"""ODE residual-only Knowledge-to-Weights experiment.
Restored from the recorded Stage-7 work. Reference labels remain float64 and are
used for evaluation only, never for training, checkpoint selection or LR selection.
"""
from __future__ import annotations
import argparse,copy,json,os,platform,sys,time
from pathlib import Path
from fractions import Fraction
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(v,'1')
import numpy as np,pandas as pd,torch
from differential_ir import *
torch.set_num_threads(1)
try:torch.set_num_interop_threads(1)
except RuntimeError:pass
ROOT=Path(__file__).resolve().parents[1]
H=32;P=97;E=32;TERMS=8;FACTORS=3
DESC=[(m,s) for s in (.25,.40) for m in (-.7,-.35,0.,.35,.7)]+[(m,.325) for m in (-.525,-.175,.175,.525)]
TRAIN_KINDS=['g','sum','product','Dg','Dsum','D2g'];TEST_KINDS=['Dproduct','Dmix','D2product','Dproduct_plus']
STEPS=[0,1,2,5,10,20,50,100];LRS=[.0003,.001,.003,.01,.03]

def forcing(kind,r,pool):
    a,b,c=[leaf(int(i),Fraction(int(r.choice([-1,1]))*int(r.integers(10,31)),20)) for i in r.choice(pool,size=3,replace=False)]
    if kind=='g':return a
    if kind=='sum':return add(a,b)
    if kind=='product':return mul(a,b)
    if kind=='Dg':return scale(diff(a),Fraction(1,5))
    if kind=='Dsum':return scale(diff(add(a,b)),Fraction(1,5))
    if kind=='D2g':return scale(diff(diff(a)),Fraction(1,25))
    if kind=='Dproduct':return scale(diff(mul(a,b)),Fraction(1,5))
    if kind=='Dmix':return scale(diff(mul(add(a,b),c)),Fraction(1,5))
    if kind=='D2product':return scale(diff(diff(mul(a,b))),Fraction(1,25))
    if kind=='Dproduct_plus':return add(scale(diff(mul(a,b)),Fraction(1,5)),c)
    raise ValueError(kind)

def pack_ir(polys):
    n=len(polys);d=np.zeros((n,TERMS,FACTORS,5),np.float32);fm=np.zeros((n,TERMS,FACTORS),np.float32);cs=np.zeros((n,TERMS),np.float32)
    for j,p in enumerate(polys):
        if len(p)>TERMS:raise ValueError('tensor term budget')
        for t,(key,c) in enumerate(p.items()):
            cs[j,t]=float(c)
            if len(key)>FACTORS:raise ValueError('tensor factor budget')
            for f,(i,order) in enumerate(key):
                mu,sigma=DESC[i];d[j,t,f]=[mu,(sigma-.325)/.15,mu*mu,order/2,1.];fm[j,t,f]=1
    return d,fm,cs

def reference(poly,x,lam,boundary,order=96):
    nodes,weights=np.polynomial.legendre.leggauss(order);x=np.asarray(x,dtype=np.float64);half=(x+1)/2
    t=-1+half[:,None]*(nodes[None,:]+1);f=evaluate(poly,t,DESC)
    return boundary*np.exp(-lam*(x+1))+(weights[None,:]*np.exp(-lam*(x[:,None]-t))*f).sum(1)*half

def build_bank(name,n,seed,groups,q=257):
    r=np.random.default_rng(seed);polys=[];rows=[];lams=[];bs=[];xs=[];fs=[];xq=[];fq=[];ref=[];rejected=0
    for j in range(n):
        group=groups[j%len(groups)]
        if group=='train':kind,pool=str(r.choice(TRAIN_KINDS)),range(10);lam=float(r.choice([0.,.5,1.,2.]))
        else:
            kind=TEST_KINDS[(j//len(groups))%len(TEST_KINDS)];pool=range(10,14) if group=='unseen_leaf' else range(10)
            lam=float(r.choice([4.,8.])) if group=='lambda_ood' else float(r.choice([.25,.75,1.5]))
        for attempt in range(10000):
            poly=canonical(forcing(kind,r,pool));xv=np.linspace(-1,1,q);fv=evaluate(poly,xv,DESC);power=float(np.mean(fv*fv))
            if power>=.002:break
            rejected+=1
        else:raise RuntimeError('degenerate bank')
        b=float(r.uniform(-.75,.75));xx=r.uniform(-1,1,128);xqq=r.uniform(-1,1,128) if name=='train' else xv
        polys.append(poly);xs.append(xx);fs.append(evaluate(poly,xx,DESC));xq.append(xqq);fq.append(evaluate(poly,xqq,DESC));lams.append(lam);bs.append(b)
        if name!='train':ref.append(reference(poly,xqq,lam,b))
        rows.append({'task':j,'group':group,'kind':kind,'lambda':lam,'boundary':b,'forcing_power':power,'ir_hash':ir_hash(poly),'ir':serialize(poly)})
    d,fm,c=pack_ir(polys)
    arr=dict(desc=d,factor_mask=fm,coeff=c,lam=np.array(lams,np.float32),b=np.array(bs,np.float32),xs=np.array(xs,np.float32),fs=np.array(fs,np.float32),xq=np.array(xq,np.float32),fq=np.array(fq,np.float32))
    arr['scale']=(1+np.array([t['forcing_power'] for t in rows])+np.square(arr['lam']*arr['b'])).astype(np.float32)
    if ref:arr['solution']=np.array(ref,np.float64)
    np.savez_compressed(ROOT/'banks'/f'{name}.npz',**arr)
    (ROOT/'banks'/f'{name}.json').write_text(json.dumps({'seed':seed,'rejections':rejected,'tasks':rows}))
    print('bank',name,n,'rejections',rejected,flush=True)

def make_banks():
    build_bank('train',2048,700100,['train']);build_bank('validation',192,700200,['train']);build_bank('development',120,700300,['structure','unseen_leaf','lambda_ood']);build_bank('test',480,700400,['structure','unseen_leaf','lambda_ood'])

def load_bank(name):
    z=np.load(ROOT/'banks'/f'{name}.npz');b={k:torch.tensor(z[k],dtype=torch.float64 if k=='solution' else torch.float32) for k in z.files}
    b['meta']=json.loads((ROOT/'banks'/f'{name}.json').read_text())['tasks'];return b

def select(bank,ix):return {k:v[ix] for k,v in bank.items() if k!='meta'}

class Hyper(torch.nn.Module):
    def __init__(self,blind=False):
        super().__init__();self.blind=blind
        self.factor=torch.nn.Sequential(torch.nn.Linear(5,E),torch.nn.Tanh(),torch.nn.Linear(E,E),torch.nn.Tanh())
        self.term=torch.nn.Sequential(torch.nn.Linear(E+1,E),torch.nn.Tanh(),torch.nn.Linear(E,E),torch.nn.Tanh())
        self.root=torch.nn.Sequential(torch.nn.Linear(E+2,64),torch.nn.Tanh(),torch.nn.Linear(64,P))
        base=torch.cat([torch.empty(H).uniform_(-.7,.7),torch.empty(H).uniform_(-.2,.2),torch.randn(H)*.05,torch.zeros(1)])
        self.base=torch.nn.Parameter(base);torch.nn.init.normal_(self.root[-1].weight,std=.005);torch.nn.init.zeros_(self.root[-1].bias)
    def forward(self,bank):
        if self.blind:return self.base[None].expand(len(bank['b']),-1)
        m=bank['factor_mask'];z=(self.factor(bank['desc'])*m[...,None]).sum(2)
        z=self.term(torch.cat([z,m.sum(2)[...,None]/3],-1));z=(z*bank['coeff'][...,None]).sum(1)
        eq=torch.stack([torch.log1p(bank['lam']),bank['b']],1)
        return self.base[None]+self.root(torch.cat([z,eq],1))

def forward_solution(th,x,b):
    w,d,a,e=th[:,:H],th[:,H:2*H],th[:,2*H:3*H],th[:,3*H:]
    h=torch.tanh(x[...,None]*w[:,None]+d[:,None]);v=(h*a[:,None]).sum(-1)+e
    return b[:,None]+(x+1)*v

def residual(th,x,f,lam,b):
    w,d,a,e=th[:,:H],th[:,H:2*H],th[:,2*H:3*H],th[:,3*H:]
    h=torch.tanh(x[...,None]*w[:,None]+d[:,None]);s=1-h*h;v=(h*a[:,None]).sum(-1)+e;vx=(s*(w*a)[:,None]).sum(-1)
    return (1+lam[:,None]*(x+1))*v+(x+1)*vx+(lam*b)[:,None]-f

def physics_grad(th,x,f,lam,b,norm):
    w,d,a,e=th[:,:H],th[:,H:2*H],th[:,2*H:3*H],th[:,3*H:]
    xx=x[...,None];t=(x+1)[...,None];h=torch.tanh(xx*w[:,None]+d[:,None]);s=1-h*h;A=1+lam[:,None,None]*t
    v=(h*a[:,None]).sum(-1)+e;vx=(s*(a*w)[:,None]).sum(-1);r=A.squeeze(-1)*v+(x+1)*vx+(lam*b)[:,None]-f
    dr=(2*r/(x.shape[1]*norm[:,None]))[...,None]
    jw=a[:,None]*s*(A*xx+t*(1-2*w[:,None]*h*xx));jd=a[:,None]*s*(A-2*t*w[:,None]*h);ja=A*h+t*w[:,None]*s
    return torch.cat([(dr*jw).sum(1),(dr*jd).sum(1),(dr*ja).sum(1),(dr*A).sum(1)],1)

def loss_values(th,bank):
    r=residual(th,bank['xq'],bank['fq'],bank['lam'],bank['b']);out={'residual':r.square().mean(1),'norm_residual':r.square().mean(1)/bank['scale']}
    if 'solution' in bank:
        err=forward_solution(th,bank['xq'],bank['b'])-bank['solution'];out['solution_mse']=err.square().mean(1);out['solution_nmse']=err.square().mean(1)/(bank['solution'].square().mean(1)+1e-8)
    return out

def curve(th,bank,lr,last=100,seed=89000):
    rng=torch.Generator().manual_seed(seed);th=th.detach().clone();out={0:{k:v.detach().numpy() for k,v in loss_values(th,bank).items()}}
    for k in range(1,last+1):
        ix=torch.randint(bank['xs'].shape[1],(len(th),32),generator=rng);x,f=bank['xs'].gather(1,ix),bank['fs'].gather(1,ix)
        with torch.no_grad():th-=lr*physics_grad(th,x,f,bank['lam'],bank['b'],bank['scale'])
        if k in STEPS:out[k]={j:v.detach().numpy() for j,v in loss_values(th,bank).items()}
    return out

def choose_lr(th,bank):
    scores=[]
    for lr in LRS:
        cv=curve(th,bank,lr,last=10);score=float(np.mean([np.log(cv[k]['norm_residual']+1e-12).mean() for k in (1,2,5,10)]))
        scores.append((score if np.isfinite(score) else float('inf'),lr))
    return min(scores)[1],scores

def train_model(seed,mode,args):
    torch.manual_seed(seed);r=torch.Generator().manual_seed(900000+seed);bank,val=load_bank('train'),load_bank('validation');model=Hyper(blind=mode=='shared');opt=torch.optim.Adam(model.parameters(),lr=.001)
    nsteps=args.direct_steps if mode=='direct' else args.meta_steps;ck={0:copy.deepcopy(model.state_dict())};logs=[];t0=time.perf_counter()
    for step in range(1,nsteps+1):
        ix=torch.randint(len(bank['b']),(16,),generator=r);batch=select(bank,ix);th=model(batch);adapted=th.detach();horizon=0 if mode=='direct' else args.horizon
        for _ in range(horizon):
            j=torch.randint(128,(16,32),generator=r)
            with torch.no_grad():adapted=adapted-args.inner_lr*physics_grad(adapted,batch['xs'].gather(1,j),batch['fs'].gather(1,j),batch['lam'],batch['b'],batch['scale'])
        j=torch.randint(128,(16,64),generator=r)
        with torch.no_grad():
            grad=physics_grad(adapted,batch['xq'].gather(1,j),batch['fq'].gather(1,j),batch['lam'],batch['b'],batch['scale'])/16
            if mode=='regmeta':
                g0=physics_grad(th.detach(),batch['xq'].gather(1,j),batch['fq'].gather(1,j),batch['lam'],batch['b'],batch['scale'])/16;grad=(grad+args.anchor*g0)/(1+args.anchor)
        if not torch.isfinite(grad).all():raise FloatingPointError((seed,mode,step))
        opt.zero_grad(set_to_none=True);th.backward(grad);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%20==0 or step==nsteps:ck[step]=copy.deepcopy(model.state_dict())
        if step%200==0 or step==nsteps:
            with torch.no_grad():score=float(loss_values(model(val),val)['norm_residual'].mean())
            logs.append({'step':step,'validation_mean_norm_residual':score})
    return model,ck,time.perf_counter()-t0,logs

def run_seed(args):
    dest=ROOT/'results'/args.phase/f'seed{args.seed:03d}';dest.mkdir(parents=True,exist_ok=True);val,test=load_bank('validation'),load_bank('development' if args.phase=='development' else 'test')
    out=[];timers=[];selections={};models={};cps={}
    for mode in ('direct','meta','regmeta','shared'):
        model,ck,sec,logs=train_model(args.seed,mode,args);models[mode]=model;cps[mode]=ck
        timers.append({'mode':mode,'train_seconds':sec,'steps':args.direct_steps if mode=='direct' else args.meta_steps,'horizon':0 if mode=='direct' else args.horizon,'anchor':args.anchor if mode=='regmeta' else 0.,'target_gradient_evaluations':args.direct_steps if mode=='direct' else args.meta_steps*(args.horizon+1+int(mode=='regmeta')),'batch_tasks':16,'support_points':32,'query_points':64})
        torch.save(model.state_dict(),dest/f'{mode}.pt');(dest/f'{mode}_training.json').write_text(json.dumps(logs,indent=2));print(args.seed,mode,sec,flush=True)
    with torch.no_grad():
        meta_zero=float(loss_values(models['meta'](val),val)['norm_residual'].mean());candidate=Hyper();matches=[];abs_scores=[]
        for k,state in cps['direct'].items():
            candidate.load_state_dict(state);err=float(loss_values(candidate(val),val)['norm_residual'].mean());matches.append((abs(np.log((err+1e-12)/(meta_zero+1e-12))),k,err));abs_scores.append((err,k))
        gap,ck,direct_zero=min(matches);_,bestck=min(abs_scores)
        matched=Hyper();matched.load_state_dict(cps['direct'][ck]);bestdirect=Hyper();bestdirect.load_state_dict(cps['direct'][bestck]);torch.manual_seed(args.seed);rand=Hyper(blind=True)
        methods={'DirectFinal':models['direct'],'DirectBest':bestdirect,'DirectMatched':matched,'Meta':models['meta'],'RegMeta':models['regmeta'],'SharedMeta':models['shared'],'Random':rand}
        def append(name,cv):
            for k,metrics in cv.items():
                for i,t in enumerate(test['meta']):out.append({'seed':args.seed,'method':name,'step':k,'task':i,'group':t['group'],'kind':t['kind'],'lambda':t['lambda'],**{key:float(v[i]) for key,v in metrics.items()}})
        for name,m in methods.items():
            lr,scores=choose_lr(m(val),val);selections[name]={'lr':lr,'validation_lr_scores':scores};append(name,curve(m(test),test,lr))
        perm=torch.roll(torch.arange(len(test['b'])),17);append('ShuffledDirect',curve(models['direct'](select(test,perm)),test,selections['DirectFinal']['lr']))
    torch.save(bestdirect.state_dict(),dest/'bestdirect.pt');torch.save(matched.state_dict(),dest/'matched.pt')
    pd.DataFrame(out).to_csv(dest/'task_results.csv',index=False);pd.DataFrame(timers).to_csv(dest/'timing.csv',index=False)
    (dest/'selection.json').write_text(json.dumps({'meta_validation_residual':meta_zero,'direct_validation_residual':direct_zero,'match_log_gap':gap,'matched_checkpoint':ck,'best_checkpoint':bestck,'lrs':selections},indent=2))

def selfcheck():
    torch.manual_seed(7341);th=(torch.randn(7,P,dtype=torch.float64)*.17).requires_grad_();x=torch.rand(7,19,dtype=torch.float64)*2-1;f=torch.randn(7,19,dtype=torch.float64);lam=torch.rand(7,dtype=torch.float64)*2;b=torch.randn(7,dtype=torch.float64);norm=torch.rand(7,dtype=torch.float64)+1
    loss=(residual(th,x,f,lam,b).square().mean(1)/norm).sum();auto=torch.autograd.grad(loss,th)[0];gerr=float((auto-physics_grad(th.detach(),x,f,lam,b,norm)).abs().max())
    xx=x.clone().requires_grad_();yy=forward_solution(th.detach(),xx,b);yp=torch.autograd.grad(yy.sum(),xx)[0];rerr=float((yp+lam[:,None]*yy-f-residual(th.detach(),xx,f,lam,b)).detach().abs().max());bc=float((forward_solution(th.detach(),-torch.ones(7,1,dtype=torch.float64),b).squeeze()-b).abs().max())
    from scipy.integrate import solve_ivp
    refs=[];quads=[];rng=np.random.default_rng(714)
    for j in range(12):
        p=canonical(forcing(TEST_KINDS[j%4],rng,range(10)));q=np.linspace(-1,1,257);ll=float(j%5);ref=reference(p,q,ll,.25,96);low=reference(p,q,ll,.25,64)
        ode=solve_ivp(lambda t,u:evaluate(p,np.array([t]),DESC)-ll*u,(-1,1),[.25],t_eval=q,method='DOP853',rtol=1e-11,atol=1e-12)
        refs.append(float(np.max(np.abs(ode.y[0]-ref))));quads.append(float(np.max(np.abs(low-ref))))
    ans={'manual_gradient_max_abs_float64':gerr,'residual_vs_autograd_max_abs_float64':rerr,'hard_boundary_max_abs_float64':bc,'reference_quadrature96_vs_DOP853_max_abs':max(refs),'quadrature64_vs96_max_abs':max(quads),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),'cuda':torch.cuda.is_available()}
    assert gerr<1e-10 and rerr<1e-10 and bc==0 and max(refs)<1e-8 and max(quads)<1e-10
    (ROOT/'checks'/'numerical_checks.json').write_text(json.dumps(ans,indent=2));print(ans)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['banks','checks','run']);p.add_argument('--seed',type=int,default=101);p.add_argument('--phase',choices=['development','confirmatory'],default='confirmatory');p.add_argument('--direct-steps',type=int,default=3600);p.add_argument('--meta-steps',type=int,default=600);p.add_argument('--horizon',type=int,default=5);p.add_argument('--inner-lr',type=float,default=.01);p.add_argument('--anchor',type=float,default=.25);a=p.parse_args()
    if a.action=='banks':make_banks()
    elif a.action=='checks':selfcheck()
    else:run_seed(a)
if __name__=='__main__':main()
