"""Final-bank evaluation; never used for checkpoint or hyperparameter choice."""
from experiment import *

def corrections(c,K,F,p,nsteps=1):
    """Damped Newton in the SAME 36-mode space, physics-only, quadrature23."""
    phi,lam,_,_,w=basis(6,23)
    ph=torch.tensor(phi); la=torch.tensor(lam); a=torch.tensor(c); kt=torch.tensor(K); ft=torch.tensor(F); be=torch.tensor(p[:,-1])
    def E(v):
        uv=v@ph.T
        return .5*(v*v*la).sum(-1)+.5*(v*torch.bmm(kt,v.unsqueeze(-1)).squeeze(-1)).sum(-1)+be*(uv**4).sum(-1)*w/4-(ft*v).sum(-1)
    def R(v): return residual(v,kt,ft,be,ph,la,w)
    for step in range(nsteps):
        u=a@ph.T; rr=R(a)
        # Do batches to bound memory and preserve one CPU thread.
        ds=[]
        for lo in range(0,len(a),64):
            hi=lo+64; weighted=ph.T[None,:,:]*(u[lo:hi,None,:]**2)
            jac=torch.diag(la)[None,:,:]+kt[lo:hi]+3*be[lo:hi,None,None]*(weighted@ph)*w
            ds.append(torch.linalg.solve(jac,-rr[lo:hi,:,None]).squeeze(-1))
        d=torch.cat(ds); alpha=torch.ones(len(a),dtype=torch.float64); ea=E(a); rd=(rr*d).sum(-1)
        for ls in range(24):
            trial=a+alpha[:,None]*d; et=E(trial)
            good=et<=ea+1e-4*alpha*rd+1e-14*torch.maximum(torch.ones_like(ea),torch.abs(ea))
            if bool(good.all()): break
            alpha=torch.where(good,alpha,alpha*.5)
        a=trial
    return a.numpy()

def rows_for(c,z,method,seed):
    coeferr=((c-z['coeff'])**2).sum(-1); total=coeferr+z['floor2']; rel=np.sqrt(total/z['norm2'])
    return pd.DataFrame(dict(seed=seed,method=method,task_id=np.arange(len(c)),group=z['groups'],relative_l2=rel,relative_coefficient_error=np.sqrt(coeferr/z['norm2']),projection_floor=np.sqrt(z['floor2']/z['norm2']),coefficient_error2=coeferr,total_error2=total,projection_fraction=z['floor2']/np.maximum(total,1e-300)))

def run():
    z=dict(np.load(ROOT/'data'/'test.npz')); p=z['params']; n=len(p); ph,lam,_,_,w=basis(6,63)
    Ks=[]; Fs=[]; Ts=[]; base=[]; indicator=[]
    q=np.arange(1,64); lf=np.pi**2*(q[:,None]**2+q[None,:]**2)
    ph100,lam100,_,_,w100=basis(10,63); g100=[]
    for i,row in enumerate(p):
        f,k,src=fields(row,63); K=(ph.T*k.ravel())@ph*w; F=ph.T@f.ravel()*w; Ks.append(K); Fs.append(F)
        _,r,J=energy_residual(z['galerkin'][i],ph,lam,K,F,row[-1],w)
        Ts.append(np.linalg.solve(J,ph.T@src[0].ravel()*w))
        a=z['galerkin'][i]; u=(ph@a).reshape(63,63); lhs=(ph@(lam*a)).reshape(63,63)+k*u+row[-1]*u**3-f
        rcoef=dstn(lhs,type=1,norm='ortho')/64; hn=np.sqrt(np.sum(rcoef*rcoef/lf)); ub=hn/np.sqrt(2*np.pi**2)
        err=np.sqrt(z['floor2'][i]+np.sum((a-z['coeff'][i])**2))
        indicator.append(dict(task_id=i,group=z['groups'][i],low_residual_norm=float(np.linalg.norm(r)),full_hminus1=float(hn),discrete_l2_bound=float(ub),l2_error=err,bound_ratio=ub/max(err,1e-300)))
        K100=(ph100.T*k.ravel())@ph100*w100; F100=ph100.T@f.ravel()*w100
        a100,info=solve_galerkin(ph100,lam100,K100,F100,row[-1],w100)
        if not info['converged']: raise RuntimeError(('Gal100',i,info))
        cg=z['ref_spectral'][i,:10,:10].ravel(); floor100=max(0,z['norm2'][i]-np.sum(cg*cg))
        g100.append(dict(seed=-1,method='Galerkin100',task_id=i,group=z['groups'][i],relative_l2=np.sqrt((np.sum((a100-cg)**2)+floor100)/z['norm2'][i]),projection_floor=np.sqrt(floor100/z['norm2'][i])))
    Ks=np.array(Ks); Fs=np.array(Fs); Ts=np.array(Ts)
    np.savez_compressed(ROOT/'data'/'test_fine_operators.npz',K=Ks,F=Fs,tangent_reference=Ts)
    pd.DataFrame(indicator).to_csv(ROOT/'analysis'/'residual_indicators.csv',index=False)
    bases=[rows_for(z['coeff'],z,'Projection36',-1),rows_for(z['galerkin'],z,'Galerkin36',-1),pd.DataFrame(g100)]
    diag=Fs/(lam[None,:]+p[:,8,None]); bases.append(rows_for(diag,z,'Diagonal',-1))
    for steps in (1,2):
        a=corrections(diag,z['K'],z['F'],p,nsteps=steps); bases.append(rows_for(a,z,f'Diagonal+Newton{steps}',-1))
    baseline=pd.concat(bases,ignore_index=True); baseline.to_csv(ROOT/'analysis'/'baseline_tasks.csv',index=False)
    results=[]; tangent_rows=[]; correction_rows=[]; audit=[]
    for seed in range(20):
        for method in ['Data','Weak','Response','DataTime']:
            m,meta=load_model(seed,method)
            pt=torch.tensor(p,dtype=torch.float32); dt=torch.zeros_like(pt); dt[:,3]=pt[:,3]
            with torch.no_grad(): a,da=m.with_tangent(pt,dt)
            a=a.numpy().astype(float); da=da.numpy().astype(float)
            if not np.isfinite(a).all(): raise RuntimeError(('nonfinite final',seed,method))
            results.append(rows_for(a,z,method,seed))
            trel=np.linalg.norm(da-Ts,axis=1)/np.linalg.norm(Ts,axis=1)
            tangent_rows.append(pd.DataFrame(dict(seed=seed,method=method,task_id=np.arange(n),group=z['groups'],tangent_relative_error=trel)))
            np.savez_compressed(ROOT/'models'/f'predictions_{method}_{seed:02d}.npz',coefficients=a,tangents=da)
            if method in ['DataTime','Response']:
                ca=corrections(a,z['K'],z['F'],p); correction_rows.append(rows_for(ca,z,method+'+Newton1',seed))
            # Independent direct-grid check of the projection identity.
            for idx in [0,200,400,600]:
                exact=dstn(z['ref_spectral'][idx]*64,type=1,norm='ortho').ravel()
                direct=np.sum((exact-ph@a[idx])**2)*w
                decomposed=z['floor2'][idx]+np.sum((z['coeff'][idx]-a[idx])**2)
                audit.append(abs(direct-decomposed))
        print('evaluated seed',seed,flush=True)
    res=pd.concat(results,ignore_index=True); res.to_csv(ROOT/'analysis'/'task_results.csv',index=False)
    pd.concat(correction_rows,ignore_index=True).to_csv(ROOT/'analysis'/'correction_tasks.csv',index=False)
    pd.concat(tangent_rows,ignore_index=True).to_csv(ROOT/'analysis'/'tangent_tasks.csv',index=False)
    allm=pd.concat([res,pd.concat(correction_rows)],ignore_index=True)
    per=allm.groupby(['seed','method','group']).agg(median_relative_l2=('relative_l2','median'),mean_log_relative_l2=('relative_l2',lambda x:np.log(x).mean()),median_projection_fraction=('projection_fraction','median'),mean_coeferr2=('coefficient_error2','mean')).reset_index()
    per.to_csv(ROOT/'analysis'/'seed_metrics.csv',index=False)
    summary=per.groupby(['method','group']).agg(relative_l2_median=('median_relative_l2','median'),seed_median_min=('median_relative_l2','min'),seed_median_max=('median_relative_l2','max'),projection_fraction_median=('median_projection_fraction','median'),mean_log_rel=('mean_log_relative_l2','mean')).reset_index()
    bsum=baseline.groupby(['method','group']).agg(relative_l2_median=('relative_l2','median')).reset_index()
    pd.concat([summary,bsum],ignore_index=True).to_csv(ROOT/'analysis'/'summary.csv',index=False)
    rng=np.random.default_rng(991100); contrasts=[]
    for g in ['id','center','width','operator']:
        piv=per[per.group==g].pivot(index='seed',columns='method',values='mean_log_relative_l2')
        for a,b in [('Weak','Data'),('Response','Data'),('Response','Weak'),('Response','DataTime'),('Weak','DataTime'),('DataTime','Data'),('Response+Newton1','Response'),('DataTime+Newton1','DataTime')]:
            d=(piv[a]-piv[b]).to_numpy(); bs=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
            primary=g=='center' and b=='Data' and a in ['Weak','Response']; alpha=.025 if primary else .05
            lo,hi=np.quantile(bs,[alpha/2,1-alpha/2])
            contrasts.append(dict(group=g,numerator=a,denominator=b,n_seeds=len(d),mean_log_ratio=d.mean(),ratio=np.exp(d.mean()),ci_level=1-alpha,ci_low=lo,ci_high=hi,ratio_low=np.exp(lo),ratio_high=np.exp(hi),wins=int((d<0).sum()),primary=primary,status='PASS' if hi<0 else ('WORSE' if lo>0 else 'UNCERTAIN')))
    pd.DataFrame(contrasts).to_csv(ROOT/'analysis'/'contrasts.csv',index=False)
    audit_data=dict(seeds=20,methods=4,test_tasks=n,rows=len(res),nonfinite=int((~np.isfinite(res.relative_l2)).sum()),projection_identity_max_abs=max(audit),reference_residual_max=float(z['numerics'][:,0].max()),galerkin_residual_max=float(z['numerics'][:,1].max()),min_discrete_bound_ratio=min(x['bound_ratio'] for x in indicator))
    (ROOT/'checks'/'evaluation_audit.json').write_text(json.dumps(audit_data,indent=2)); print(json.dumps(audit_data,indent=2))
    print(summary.to_string(index=False)); print(pd.DataFrame(contrasts).query('primary').to_string(index=False))
if __name__=='__main__': run()
