"""Stage 8 analysis: seed-level paired bootstrap, descriptive summaries, independent reference checks.
Input roots are frozen Stage8a and Stage8b runs; never treats tasks as independent seeds.
"""
from pathlib import Path
import json, time
from fractions import Fraction
import numpy as np,pandas as pd
from scipy.integrate import solve_ivp
import stage8_nonlinear as s
from differential_ir import evaluate
W=Path(__file__).parent
rng=np.random.default_rng(929111)
REPS=20000
EPS=1e-12

def stage_data(folder):
    root=W/folder
    rs=[]
    for seed in range(301,311):
        p=root/'seeds'/f'{seed:03d}'/'metrics.csv'
        assert p.exists(),p
        rs.append(pd.read_csv(p))
    d=pd.concat(rs,ignore_index=True)
    assert len(d)==10*240*4*8
    assert not d.solution_mse.isna().any()
    assert np.isfinite(d.solution_mse).all()
    return d

def bootstrap(v):
    v=np.asarray(v,float)
    if len(v)!=10: raise ValueError(f'Expected 10 seeds got {len(v)}')
    ids=rng.integers(0,len(v),size=(REPS,len(v)))
    z=v[ids].mean(1)
    return tuple(np.quantile(z,[.025,.975]))

def analyze_stage(folder):
    d=stage_data(folder)
    out=W/folder/'analysis';out.mkdir(exist_ok=True)
    rows=[];absolute=[]
    for g in ['structure','unseen_leaf','coefficient_ood']:
        for step in [0,1,2,5,10,20,50,100]:
            q=d[(d.group==g)&(d.step==step)]
            p=q.pivot(index=['seed','task'],columns='method',values='solution_mse')
            for method in ('Meta','RegMeta','Random'):
                seed_diff=np.log((p[method]+EPS)/(p['DirectBest']+EPS)).groupby(level='seed').mean()
                ci=bootstrap(seed_diff.values)
                rows.append(dict(group=g,step=step,method=method,contrast=f'{method}/DirectBest',mean_log_ratio=float(seed_diff.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),negative_seeds=int((seed_diff<0).sum()),n_seeds=10,n_task_per_seed=80,decision=('META_BETTER' if ci[1]<0 else 'DIRECT_BETTER' if ci[0]>0 else 'UNCERTAIN')))
            for method in ('DirectBest','Meta','RegMeta','Random'):
                seed_median=q[q.method==method].groupby('seed').solution_mse.median()
                absrow=dict(group=g,step=step,method=method,median_of_seed_medians=float(seed_median.median()),p10_seed_median=float(seed_median.quantile(.1)),p90_seed_median=float(seed_median.quantile(.9)),hit_at_mse_001=float((q[q.method==method].solution_mse<.01).mean()),nonfinite=int((~np.isfinite(q[q.method==method].solution_mse)).sum()))
                absolute.append(absrow)
    pd.DataFrame(rows).to_csv(out/'contrasts.csv',index=False)
    pd.DataFrame(absolute).to_csv(out/'absolute_mse.csv',index=False)
    tm=[]
    for seed in range(301,311):
        folderp=W/folder/'seeds'/f'{seed:03d}'
        train=pd.read_csv(folderp/'training.csv')
        select=json.loads((folderp/'selection.json').read_text())
        for rec in train.to_dict('records'): tm.append({'seed':seed,**rec})
        for row in select:tm.append({'seed':seed,'method':row['method'],'inference_all240_seconds':row['forward_seconds_total'],'adapt100_all240_seconds':row['adapt100_seconds_total'],'lr':row['best_lr'],'phase':'inference'})
    pd.DataFrame(tm).to_csv(out/'timings.csv',index=False)
    return rows,absolute


a,absa=analyze_stage('stage8a_output')
b,absb=analyze_stage('stage8_output')

# Stage8c is exploratory: posthoc, repeated constant does not increase independent N.
croot=W/'stage8_output'/'stage8c_exploratory'
cd=pd.concat([pd.read_csv(croot/f'correction_seed{sid}.csv') for sid in range(301,311)],ignore_index=True)
comp=[];rel=[]
for group in ('structure','unseen_leaf','coefficient_ood'):
 for k in (0,1,2,3):
  q=cd[(cd.group==group)&(cd.newton_round==k)]
  for method in ['Constant','DirectFinal','Meta','RegMeta']:
   qq=q[q.method==method]
   seedm=qq.groupby('seed').solution_mse.median()
   comp.append(dict(group=group,correction_round=k,method=method,median_seed_median=float(seedm.median()),
                    success_mse_below_1e_5=float((qq.solution_mse<1e-5).mean()),success_mse_below_1e_7=float((qq.solution_mse<1e-7).mean()),
                    correction_batch_seconds_median=float(qq.groupby('seed').correction_batch_seconds.median().median())))
  p=q.pivot(index=['seed','task'],columns='method',values='solution_mse')
  for method in ('DirectFinal','Meta','RegMeta'):
   sf=np.log((p[method]+EPS)/(p['Constant']+EPS)).groupby('seed').mean()
   lo,hi=bootstrap(sf.values)
   rel.append(dict(group=group,round=k,comparison=f'{method}/Constant',mean_log_ratio=float(sf.mean()),ci95_low=lo,ci95_high=hi,negative_seeds=int((sf<0).sum())))
pd.DataFrame(comp).to_csv(croot/'correction_summary.csv',index=False)
pd.DataFrame(rel).to_csv(croot/'correction_contrasts.csv',index=False)

# Fresh Stage8b reference checked against a different integration algorithm and tighter DOP853.
bank=s.load('test',include_reference=True)
raw=bank['meta'];q=bank['xq'].numpy();original=bank['solution'].numpy()
selected=list(range(0,240,20))
max_dop=0;max_radau=0;per=[]
for j in selected:
 row=raw[j];p={tuple(tuple(x) for x in k):Fraction(num,den) for k,num,den in row['ir']}
 def rhs(t,y):return [float(evaluate(p,np.array([t]),s.st.DESC)[0])-row['lambda']*y[0]-row['beta']*y[0]**3]
 ode=solve_ivp(rhs,(-1,1),[row['boundary']],method='Radau',rtol=5e-11,atol=1e-12,dense_output=True)
 dop=s.ode_solution(p,row['lambda'],row['beta'],row['boundary'],q[j],rtol=2e-12,atol=2e-13)
 rad=ode.sol(q[j])[0]
 gaprad=np.max(np.abs(original[j]-rad));gapdop=np.max(np.abs(original[j]-dop))
 max_radau=max(max_radau,gaprad);max_dop=max(max_dop,gapdop)
 per.append({'task':j,'radau_abs_max':float(gaprad),'tighter_dop_abs_max':float(gapdop)})
assert max_radau<1e-7 and max_dop<1e-8
pd.DataFrame(per).to_csv(W/'stage8_output'/'analysis'/'reference_crosscheck.csv',index=False)

# Fair deployment precision: calculate support-only Chebyshev with FP32-quantized output.
classical=pd.read_csv(W/'stage8_output'/'analysis'/'cheb_classical.csv')
summ=classical.groupby('group').agg(median_mse=('mse','median'),p90_mse=('mse',lambda z:z.quantile(.9)),
           max_mse=('mse','max'),median_wall_s=('wall_s','median'),p90_wall_s=('wall_s',lambda z:z.quantile(.9)),success_rate=('success','mean')).reset_index()
summ.to_csv(W/'stage8_output'/'analysis'/'cheb_summary.csv',index=False)

# Print easy-to-verify results
print('Stage8a primary',next(r for r in a if r['group']=='structure' and r['step']==10 and r['method']=='Meta'))
print('Stage8b primary',next(r for r in b if r['group']=='structure' and r['step']==10 and r['method']=='Meta'))
print('Stage8b Cheb summary\n',summ.to_string(index=False))
print('Reference maximum cross-checks',max_radau,max_dop)
print('Stage8c correction by stage, group structure\n',pd.DataFrame(comp).query('group == "structure"').to_string(index=False))
