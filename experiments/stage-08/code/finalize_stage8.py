from __future__ import annotations
from pathlib import Path
import json,hashlib,shutil,zipfile,platform
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
W=Path(__file__).parent
OUT=Path('/mnt/data/neurocompiler_stage8_final');OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)

stagea=W/'stage8a_output';stageb=W/'stage8_output'
a=pd.read_csv(stagea/'analysis'/'contrasts.csv');b=pd.read_csv(stageb/'analysis'/'contrasts.csv')
abs_a=pd.read_csv(stagea/'analysis'/'absolute_mse.csv');abs_b=pd.read_csv(stageb/'analysis'/'absolute_mse.csv')
classical=pd.read_csv(stageb/'analysis'/'cheb_classical.csv')
factorial=pd.read_csv(stageb/'analysis'/'cheb_sampling_precision_2x2.csv')
fresh=pd.read_csv(stageb/'stage8d_confirm'/'correction_fresh_results.csv')
chebfresh=pd.read_csv(stageb/'stage8d_confirm'/'cheb_lobatto_fresh_results.csv')
bench=pd.read_csv(stageb/'analysis'/'latency_exploratory.csv')

# Figure 1 primary MSE curves, group structure (median of seed medians).
fig,ax=plt.subplots(figsize=(7.4,4.8))
for method in ['DirectBest','Meta','RegMeta','Random']:
    z=abs_b[(abs_b.group=='structure')&(abs_b.method==method)]
    ax.plot(z.step,z.median_of_seed_medians,marker='o',label=method)
ax.set_yscale('log');ax.set_xlabel('Support-only gradient updates')
ax.set_ylabel('Solution MSE (dimensionless)')
ax.set_title('Stage 8b: nonlinear ODE, unseen expression structures')
ax.legend();ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(OUT/'figures'/'stage8b_learning_curve.png',dpi=170);plt.close(fig)

# Figure 2 the primary contrast moves with budget.
q1=a.query("group == 'structure' and step == 10 and method == 'Meta'").iloc[0]
q2=b.query("group == 'structure' and step == 10 and method == 'Meta'").iloc[0]
fig,ax=plt.subplots(figsize=(7.4,4.4))
ys=[q1.mean_log_ratio,q2.mean_log_ratio];lo=[q1.ci95_low,q2.ci95_low];hi=[q1.ci95_high,q2.ci95_high]
for j in range(2):
    ax.errorbar([j],[ys[j]],yerr=[[ys[j]-lo[j]],[hi[j]-ys[j]]],fmt='o',capsize=6)
ax.axhline(0,linewidth=1,linestyle='--')
ax.set_xticks([0,1],['Stage 8a: Meta 300','Stage 8b: Meta 900'])
ax.set_ylabel('Seed-mean log(Meta MSE / Direct MSE)')
ax.set_title('Meta underperforms at lower budget; the gap closes at higher budget')
ax.grid(alpha=.2,axis='y');fig.tight_layout()
fig.savefig(OUT/'figures'/'meta_budget_contrasts.png',dpi=170);plt.close(fig)

# Figure 3 placement/precision factorial: maximum error, no fixed visual color.
fac=factorial.groupby(['arrangement','rhs_dtype']).solution_mse.agg(['median','max']).reset_index()
labels=[f"{'Uniform' if r.arrangement=='uniform_random' else 'Lobatto'}\n{r.rhs_dtype}" for _,r in fac.iterrows()]
fig,ax=plt.subplots(figsize=(7.4,4.6))
ax.bar(labels,fac['max'])
ax.set_yscale('log');ax.set_ylabel('Worst-case test MSE (240 tasks)')
ax.set_title('Sampling locations dominate FP32 vs FP64 RHS precision')
ax.grid(alpha=.2,axis='y');fig.tight_layout()
fig.savefig(OUT/'figures'/'chebyshev_sampling_precision.png',dpi=170);plt.close(fig)

# Figure 4 posthoc-vs-fresh correction pass rates.
forpaper=[]
for g in ['structure','unseen_leaf','coefficient_ood']:
 z=fresh[fresh.group==g]
 for method,roundno in [('Constant',2),('Constant',3),('DirectFinal',1),('Meta',1)]:
  q=z[(z.method==method)&(z['round']==roundno)]
  forpaper.append(dict(group=g,method=f'{method} + {roundno} correction',pass_at_1e5=float((q.mse<1e-5).mean()),median_mse=float(q.mse.median())))
cat=pd.DataFrame(forpaper)
cat.to_csv(OUT/'fresh_correction_coverage.csv',index=False)
fig,ax=plt.subplots(figsize=(9.2,5))
methods=cat['method'].unique().tolist()
x=np.arange(3)
for j,m in enumerate(methods):
 z=cat[cat.method==m]
 ax.bar(x+(.7*j/len(methods)-.35),z.pass_at_1e5*100,width=.7/len(methods),label=m)
ax.set_xticks(x,['Structure OOD','Unseen primitive','Coefficient OOD'])
ax.set_ylabel('Tasks meeting MSE < 1e-5 (%)')
ax.set_ylim(0,110)
ax.set_title('Fresh task bank (Stage 8d): fixed correction counts')
ax.legend(loc='lower left',fontsize=8);fig.tight_layout()
fig.savefig(OUT/'figures'/'fresh_correction_coverage.png',dpi=170);plt.close(fig)

for src,dst in [(stagea/'analysis'/'contrasts.csv','stage8a_contrasts.csv'),(stageb/'analysis'/'contrasts.csv','stage8b_contrasts.csv'),
                (stageb/'analysis'/'cheb_sampling_precision_2x2.csv','chebyshev_factorial.csv'),
                (stageb/'analysis'/'latency_exploratory.csv','latency_exploratory.csv'),
                (stageb/'analysis'/'reference_crosscheck.csv','reference_crosscheck.csv'),
                (stageb/'stage8d_confirm'/'cheb_lobatto_fresh_results.csv','fresh_chebyshev_results.csv'),
                (stageb/'stage8d_confirm'/'correction_fresh_results.csv','fresh_correction_results.csv')]:
    shutil.copy2(src,OUT/dst)

# Clear formula variable dictionary and numerical consistency information for machine readers.
train=[]
for sid in range(301,311):
 for stage,root in [('a',stagea),('b',stageb)]:
  zz=pd.read_csv(root/'seeds'/str(sid)/'training.csv')
  for r in zz.to_dict('records'):train.append({'stage':stage,'seed':sid,**r})
training=pd.DataFrame(train);training.to_csv(OUT/'training_times.csv',index=False)
refcheck=pd.read_csv(OUT/'reference_crosscheck.csv')

def mainrow(x):return {'mean_log_ratio':float(x.mean_log_ratio),'ci95':[float(x.ci95_low),float(x.ci95_high)],'negative_seed_count':int(x.negative_seeds)}
summary={
 'study':'NeuroCompiler Stage8: nonlinear ODE benchmarks and failure-driven branches',
 'experimental_groups':{'Stage8a':'10 seeds; 300-step Meta; heldout bank 823005',
                        'Stage8b':'10 seeds; 900-step Meta; separate heldout bank 823006',
                        'Stage8d':'fresh nonparametric solver routing test bank 823007, trained weights reused from 8b'},
 'Stage8a_primary':mainrow(q1),'Stage8b_primary':mainrow(q2),
 'gradient_float64_diff':float(json.load(open(stagea/'analysis'/'checks.json'))['nonlinear_gradient']['max_abs_gradient_disagreement_float64']),
 'reference_max_dop853_vs_radau':float(refcheck.radau_abs_max.max()),
 'Chebyshev_random_support_median_mse':float(classical.mse.median()),
 'Chebyshev_uniform_worst_mse':float(factorial.query('arrangement=="uniform_random" and rhs_dtype=="float64"').solution_mse.max()),
 'Chebyshev_Lobatto_worst_mse':float(factorial.query('arrangement=="chebyshev_lobatto" and rhs_dtype=="float64"').solution_mse.max()),
 'Stage8d_const3_success_tau_1e5':float((fresh[(fresh.method=="Constant")&(fresh["round"]==3)].mse<1e-5).mean()),
 'Stage8d_Cheb_lobatto_max_mse':float(chebfresh.mse_float64.max()),
 'latency_benchmarks':bench.to_dict('records'),
 'independent_learning_seeds':10,'tasks_per_group':80,
 'caveats':['classical and neural solvers differ in FP64/FP32 operating precision',
            'wall timings exclude symbolic front end and bank construction; neural timings are batched, classical is per-task serial',
            'Stage8d router experiment is a fresh task bank but correction counts were selected following exploratory Stage8c',
            'sampled residual checks do not provide a mathematically certified error bound',
            'Stage8b Meta has 3x target gradient evaluations despite similar training wall time',
            'multiple secondary contrasts do not have multiplicity-adjusted simultaneous confidence intervals']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in summary.items() if k in ('Stage8a_primary','Stage8b_primary','Stage8d_const3_success_tau_1e5','Stage8d_Cheb_lobatto_max_mse')},indent=2))
print('Stage8b training median seconds\n',training.query('stage == "b"').groupby('method').training_seconds.agg(['median','min','max']).to_string())
