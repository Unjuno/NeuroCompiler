"""2x2 ablation for numerical failure: node placement vs forcing precision.
All else same Chebyshev degree32, same FP64 LM residual solver. Exploratory post hoc.
"""
from fractions import Fraction
import numpy as np,pandas as pd,stage8_nonlinear as s
from differential_ir import evaluate
from pathlib import Path
bank=s.load('test',True)
rows=bank['meta'];N=len(rows)
xlobatto=-np.cos(np.pi*np.arange(128)/127)
y=bank['solution'].numpy();xq=bank['xq'].numpy().astype(float)[0]
res=[]
for i,row in enumerate(rows):
    poly={tuple(tuple(atom) for atom in key):Fraction(num,den) for key,num,den in row['ir']}
    for arrangement in ('uniform_random','chebyshev_lobatto'):
        x=bank['xs'][i].numpy().astype(float) if arrangement=='uniform_random' else xlobatto
        for rhs_dtype in ('float32','float64'):
            f=evaluate(poly,x,s.st.DESC)
            if rhs_dtype=='float32':f=f.astype(np.float32).astype(float)
            z,diag=s.cheb_solve(x,f,row['lambda'],row['beta'],row['boundary'],xq,32)
            res.append({'task':i,'group':row['group'],'arrangement':arrangement,'rhs_dtype':rhs_dtype,
                        'solution_mse':float(np.mean((z-y[i])**2)),'success':bool(diag['success']),
                        'fit_residual_per_point':float(2*diag['cost']/len(x))})
    if i%80==79:print('progress',i+1,flush=True)
df=pd.DataFrame(res);df.to_csv(s.ROOT/'analysis'/'cheb_sampling_precision_2x2.csv',index=False)
print(df.groupby(['arrangement','rhs_dtype']).agg(median=('solution_mse','median'),p90=('solution_mse',lambda v:v.quantile(.90)),worst=('solution_mse','max'),above1e_7=('solution_mse',lambda v:sum(v>1e-7))).to_string())
print('worst case task54')
print(df[df.task==54].to_string(index=False))
