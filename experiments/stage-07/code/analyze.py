"""Paired training-seed analysis; fixed task bank, explicit failure accounting."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parents[1];O=R/'analysis';O.mkdir(exist_ok=True)
EPS=1e-18

def boot(v,name,group,category):
    v=np.asarray(v,float);assert len(v)==10 and np.isfinite(v).all()
    rng=np.random.default_rng(70881);bs=v[rng.integers(0,10,(20000,10))].mean(1);lo,hi=np.quantile(bs,[.025,.975]);lo99,hi99=np.quantile(bs,[.005,.995])
    return {'contrast':name,'group':group,'category':category,'mean_log_ratio':v.mean(),'ci95_low':lo,'ci95_high':hi,'ci99_low':lo99,'ci99_high':hi99,'geometric_ratio':float(np.exp(v.mean())),'negative_seeds':int((v<0).sum()),'n_seeds':10,'seed_SE_type_A':v.std(ddof=1)/np.sqrt(10),'k_t95':float(t.ppf(.975,9)),'decision':'NEGATIVE_SUPPORTED' if hi<0 else 'POSITIVE_SUPPORTED' if lo>0 else 'UNCERTAIN'}

def main():
    dirs=[R/'results'/'confirmatory'/f'seed{s}' for s in range(101,111)]
    df=pd.concat([pd.read_csv(p/'task_results.csv') for p in dirs],ignore_index=True)
    hd=pd.concat([pd.read_csv(p/'head_results.csv') for p in dirs],ignore_index=True)
    cc=pd.concat([pd.read_csv(p/'basis_controls.csv') for p in dirs],ignore_index=True)
    sa=pd.concat([pd.read_csv(p/'safe_adaptation.csv') for p in dirs],ignore_index=True)
    sd=pd.concat([pd.read_csv(p/'safe_adaptation_diag.csv') for p in dirs],ignore_index=True)
    assert len(df)==307200 and len(hd)==76800 and len(cc)==28800 and len(sa)==115200
    assert not df.duplicated(['seed','method','step','task']).any()
    assert not hd.duplicated(['seed','method','points','deployment','task']).any()
    fail=~np.isfinite(df[['solution_mse','norm_residual']]).all(axis=1)
    assert (df.loc[fail,'group']=='lambda_ood').all()
    assert all(np.isfinite(x[['solution_mse','norm_residual']]).all().all() for x in (hd,cc,sa))
    failures=df.assign(nonfinite=fail).groupby(['seed','group','method','step']).nonfinite.agg(['sum','mean']).reset_index()
    failures.to_csv(O/'nonfinite_failures.csv',index=False)
    rates=failures.query("group=='lambda_ood' and step==100").groupby('method').agg(rate=('mean','mean'),rate_min=('mean','min'),rate_max=('mean','max'),n_fail=('sum','sum')).reset_index();rates['n_cases']=1600;rates.to_csv(O/'lambda_ood_failure_summary.csv',index=False)
    # Never let pandas silently drop numerical failures from descriptive medians.
    display=df.copy();display.loc[fail,['solution_mse','norm_residual','solution_nmse']]=np.inf
    med=display.groupby(['seed','group','method','step'])[['solution_mse','norm_residual','solution_nmse']].median().reset_index();med.to_csv(O/'seed_medians.csv',index=False)
    med.groupby(['group','method','step']).agg(solution_mse_median=('solution_mse','median'),solution_mse_p10=('solution_mse',lambda v:v.quantile(.1)),solution_mse_p90=('solution_mse',lambda v:v.quantile(.9)),residual_median=('norm_residual','median'),solution_nmse_median=('solution_nmse','median')).reset_index().to_csv(O/'absolute_summary.csv',index=False)
    hm=hd.groupby(['seed','group','method','points','deployment'])[['solution_mse','norm_residual','rank','max_abs_head','naive_fp32_solution_mse']].median().reset_index();hm.to_csv(O/'head_seed_medians.csv',index=False)
    hm.groupby(['group','method','points','deployment']).median(numeric_only=True).drop(columns='seed').reset_index().to_csv(O/'head_absolute_summary.csv',index=False)
    cm=cc.groupby(['seed','group','method'])[['solution_mse','norm_residual']].median().reset_index();cm.to_csv(O/'basis_seed_medians.csv',index=False)
    cm.groupby(['group','method']).median(numeric_only=True).drop(columns='seed').reset_index().to_csv(O/'basis_absolute_summary.csv',index=False)
    sm=sa.groupby(['seed','group','method','step'])[['solution_mse','norm_residual']].median().reset_index();sm.to_csv(O/'safe_seed_medians.csv',index=False)
    sm.groupby(['group','method','step']).solution_mse.agg(['median','min','max']).reset_index().to_csv(O/'safe_absolute_summary.csv',index=False)
    contrasts=[];raw=[]
    def pair(a,b,name,group,cat='secondary'):
        ab=pd.concat([a.rename('a'),b.rename('b')],axis=1)
        if not np.isfinite(ab).all().all():
            contrasts.append({'contrast':name,'group':group,'category':cat,'decision':'UNDEFINED_nonfinite','n_failed_pairs':int((~np.isfinite(ab).all(axis=1)).sum())});return
        delta=(np.log(ab.a+EPS)-np.log(ab.b+EPS)).groupby(level='seed').mean();contrasts.append(boot(delta,name,group,cat))
        raw.extend({'seed':int(s),'contrast':name,'group':group,'delta':float(z)} for s,z in delta.items())
    for group in ('structure','unseen_leaf','lambda_ood'):
        d=display[display.group==group].set_index(['seed','task']);h=hd[(hd.group==group)&(hd.points==128)&(hd.deployment=='FP32aware')].set_index(['seed','task']);h64=hd[(hd.group==group)&(hd.points==128)&(hd.deployment=='FP64')].set_index(['seed','task']);c=cc[cc.group==group].set_index(['seed','task'])
        for a,b,k,name in [('Meta','DirectBest',10,'Meta_vs_DirectBest_MSE10'),('RegMeta','DirectBest',10,'RegMeta_vs_DirectBest_MSE10'),('RegMeta','Meta',0,'RegMeta_vs_Meta_MSE0'),('RegMeta','Meta',10,'RegMeta_vs_Meta_MSE10'),('DirectBest','Random',0,'DirectBest_vs_Random_MSE0'),('DirectFinal','ShuffledDirect',0,'Direct_vs_Shuffled_MSE0')]:
            pair(d[(d.method==a)&(d.step==k)].solution_mse,d[(d.method==b)&(d.step==k)].solution_mse,name,group,'primary' if group=='structure' and name=='Meta_vs_DirectBest_MSE10' else 'secondary')
        for m in ('DirectBest','Meta','RegMeta','Random'):
            pair(h[h.method==m].solution_mse,d[(d.method==m)&(d.step==10)].solution_mse,m+'_Head128_FP32aware_vs_SGD10',group,'head_branch')
            z=h64[h64.method==m];pair(z.naive_fp32_solution_mse,z.solution_mse,m+'_NaiveFP32_vs_FP64',group,'precision')
        pair(h[h.method=='RegMeta'].solution_mse,h[h.method=='Random'].solution_mse,'RegMetaHead_vs_UntunedRandomHead',group,'head_branch')
        for m in ('DirectBest','Meta','RegMeta'):
            pair(h[h.method==m].solution_mse,c[c.method==m+'_ForcingShuffled_Head32'].solution_mse,m+'_HeadCorrect_vs_ForcingShuffle',group,'followup')
        for a,b in [('RandomTuned_Head32','RegMeta'),('RandomTuned_Head32','Meta'),('Chebyshev_FP32','RegMeta')]:
            pair(c[c.method==a].solution_mse,h[h.method==b].solution_mse,a+'_vs_'+b+'Head',group,'followup')
    mt=[]
    for p in dirs:
        v=json.loads((p/'selection.json').read_text());mt.append({'seed':int(p.name[4:]),**{k:v[k] for k in ('meta_validation_residual','direct_validation_residual','match_log_gap','matched_checkpoint','best_checkpoint')}})
    match=pd.DataFrame(mt);match['within10pct']=match.match_log_gap<=np.log(1.1);match.to_csv(O/'matching_audit.csv',index=False)
    contrasts.append({'contrast':'loss_matched_initialization_mechanism','group':'all','category':'mechanistic','decision':'INVALID_MATCHING','matching_pass_seeds':int(match.within10pct.sum()),'n_seeds':10})
    pd.DataFrame(contrasts).to_csv(O/'contrasts.csv',index=False);pd.DataFrame(raw).to_csv(O/'seed_contrasts.csv',index=False)
    ti=pd.concat([pd.read_csv(p/'timing.csv').assign(seed=int(p.name[4:])) for p in dirs]);ti.groupby('mode').agg(seconds_median=('train_seconds','median'),seconds_min=('train_seconds','min'),seconds_max=('train_seconds','max'),steps=('steps','median'),target_gradient_evals=('target_gradient_evaluations','median')).to_csv(O/'training_timing_summary.csv')
    ht=pd.concat([pd.read_csv(p/'head_timing.csv') for p in dirs]);ht.groupby(['method','points','deployment']).agg(solve_seconds_median=('solve_seconds','median'),solve_seconds_min=('solve_seconds','min'),solve_seconds_max=('solve_seconds','max'),n_tasks=('n_tasks','median')).to_csv(O/'head_timing_summary.csv')
    audit={'n_training_seeds':10,'n_unique_final_tasks':480,'n_neural_rows':len(df),'n_head_rows':len(hd),'n_basis_rows':len(cc),'n_safe_rows':len(sa),'nonfinite_neural_rows':int(fail.sum()),'all_nonfinite_in_lambda_ood':True,'head_and_basis_nonfinite_rows':0,'safe_nonfinite_rows':0,'max_accepted_support_loss_rise':float(sd.max_support_loss_rise_all_tasks.max()),'safe_fallback_steps':int(sd.fallback_steps.sum()),'matching_pass_seeds':int(match.within10pct.sum()),'bootstrap_unit':'training seed, conditional on fixed test bank','solution_reference_dtype':'float64'}
    (O/'audit.json').write_text(json.dumps(audit,indent=2))
    actual=pd.DataFrame(contrasts).query("contrast=='Meta_vs_DirectBest_MSE10' and group=='structure'").iloc[0]
    reproduction={'prior_recorded_primary':-.320181,'rerun_primary':float(actual.mean_log_ratio),'absolute_difference_to_recorded_6decimal':float(abs(actual.mean_log_ratio+.320181)),'prior_nonfinite_count':8313,'rerun_nonfinite_count':int(fail.sum()),'not_an_independent_replicate':True}
    (O/'reproduction_check.json').write_text(json.dumps(reproduction,indent=2))
    print(pd.DataFrame(contrasts).query("group=='structure'")[['contrast','mean_log_ratio','ci95_low','ci95_high','negative_seeds','decision']].to_string(index=False));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
