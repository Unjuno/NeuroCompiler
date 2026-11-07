from experiment import *
if __name__=='__main__':
    z=dict(np.load(ROOT/'data'/'test.npz')); rec=[]; deriv=[]
    for group in ['id','center','width','operator']:
        ids=np.flatnonzero(z['groups']==group)[::16][:12]
        for j,idx in enumerate(ids):
            p=z['params'][idx]; f,k,_=fields(p,127); u,info=solve_reference(f,k,p[-1]); assert info['converged']
            c=dstn(u,type=1,norm='ortho')/128; old=z['ref_spectral'][idx]
            diff=np.sum((c[:63,:63]-old)**2)+np.sum(c*c)-np.sum(c[:63,:63]**2)
            row=dict(task_id=int(idx),group=group,relative_grid_change=np.sqrt(max(0,diff)/np.sum(c*c)),projection_coeff_change=np.linalg.norm(c[:6,:6].ravel()-z['coeff'][idx])/np.linalg.norm(c),relative_127_255=np.nan)
            if j<2:
                f2,k2,_=fields(p,255); u2,i2=solve_reference(f2,k2,p[-1]); assert i2['converged']; c2=dstn(u2,type=1,norm='ortho')/256
                d2=np.sum((c2[:127,:127]-c)**2)+np.sum(c2*c2)-np.sum(c2[:127,:127]**2)
                row['relative_127_255']=np.sqrt(max(0,d2)/np.sum(c2*c2))
            rec.append(row)
            if j==0:
                ep=1e-4; pp=p.copy(); pm=p.copy(); pp[3]*=1+ep; pm[3]*=1-ep
                fp,kp,_=fields(pp,63); fm,km,_=fields(pm,63)
                up,ip=solve_reference(fp,kp,pp[-1]); um,im=solve_reference(fm,km,pm[-1]); assert ip['converged'] and im['converged']
                d=(dstn(up,type=1,norm='ortho')-dstn(um,type=1,norm='ortho'))/(128*ep)
                ph,la,_,_,w=basis(6,63); f0,k0,src=fields(p,63); K=(ph.T*k0.ravel())@ph*w; F=ph.T@f0.ravel()*w
                cg=z['galerkin'][idx]; _,_,J=energy_residual(cg,ph,la,K,F,p[-1],w); dt=np.linalg.solve(J,ph.T@src[0].ravel()*w)
                # The two are not identical: derivative of Galerkin differs from projection of reference derivative.
                deriv.append(dict(task_id=int(idx),group=group,galerkin_vs_projected_reference_tangent_rel=np.linalg.norm(dt-d[:6,:6].ravel())/np.linalg.norm(d[:6,:6])))
        print('refinement',group,flush=True)
    pd.DataFrame(rec).to_csv(ROOT/'checks'/'reference_refinement.csv',index=False)
    pd.DataFrame(deriv).to_csv(ROOT/'checks'/'reference_tangent_check.csv',index=False)
    # Audit disjoint parameter banks, not merely filenames.
    train=np.load(ROOT/'data'/'train.npz')['params']; val=np.load(ROOT/'data'/'val.npz')['params']; te=z['params']
    sets=[set(map(tuple,x)) for x in [train,val,te]]
    audit=dict(train_val_overlap=len(sets[0]&sets[1]),train_test_overlap=len(sets[0]&sets[2]),val_test_overlap=len(sets[1]&sets[2]),test_unique=len(sets[2]),refined_tasks=len(rec),max_relative_grid_change=max(x['relative_grid_change'] for x in rec),max_projection_coeff_change=max(x['projection_coeff_change'] for x in rec))
    (ROOT/'checks'/'bank_audit.json').write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))
