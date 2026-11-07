"""Independent fresh-task replication; same already-frozen 80 models, no retraining."""
from experiment import *
from evaluate import rows_for,corrections
if __name__=='__main__':
    ps=[]; groups=[]
    for i,g in enumerate(['id','center','width','operator']): ps.append(sample_params(100,110101+i,g)); groups += [g]*100
    make_bank('fresh',np.concatenate(ps),groups)
    z=dict(np.load(ROOT/'data'/'fresh.npz')); p=z['params']; rec=[]
    for seed in range(20):
        for method in ['Data','Weak','Response','DataTime']:
            m,_=load_model(seed,method)
            with torch.no_grad(): a=m(torch.tensor(p,dtype=torch.float32)).numpy().astype(float)
            rec.append(rows_for(a,z,method,seed))
    out=pd.concat(rec,ignore_index=True); out.to_csv(ROOT/'analysis'/'fresh_task_results.csv',index=False)
    sm=out.groupby(['group','seed','method']).relative_l2.apply(lambda x:np.log(x).mean()).reset_index(name='meanlog'); contrasts=[]; rng=np.random.default_rng(882301)
    for g in ['id','center','width','operator']:
        pv=sm[sm.group==g].pivot(index='seed',columns='method',values='meanlog')
        for a,b in [('Weak','Data'),('Response','Data'),('Response','DataTime')]:
            d=(pv[a]-pv[b]).to_numpy(); bs=d[rng.integers(0,20,(10000,20))].mean(1); lo,hi=np.quantile(bs,[.025,.975])
            contrasts.append(dict(group=g,numerator=a,denominator=b,mean_logratio=d.mean(),ratio=np.exp(d.mean()),ci_low=lo,ci_high=hi,wins=int((d<0).sum())))
    pd.DataFrame(contrasts).to_csv(ROOT/'analysis'/'fresh_contrasts.csv',index=False)
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(out.groupby(['method','group']).relative_l2.median().unstack())
