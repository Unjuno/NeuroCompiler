import numpy as np,torch
import stage8_nonlinear as s
v=s.load('validation');d=s.load('development',True)
for mode,steps in [('direct',1200),('meta',900)]:
 model,ck,sec,fail=s.train(301,mode,meta_steps=steps)
 if mode=='direct':
  losses=[]
  for k,state in ck.items():
   m=s.HyperNL();m.load_state_dict(state)
   with torch.no_grad():losses.append((float(s.metrics(m(v),v)['norm_residual'].mean()),k))
  model.load_state_dict(ck[min(losses)[1]])
 with torch.no_grad():
  tv=model(v);td=model(s.sub(d,slice(None)))
  for lr in (.01,.03,.1,.2,.4,.8,1.6):
   cv=s.trajectory(tv,v,lr,921800+301,10)
   vscore=np.mean([np.log(cv[k]['norm_residual']+1e-12).mean() for k in [1,2,5,10]])
   dv=s.trajectory(td,d,lr,950000+301,10)
   print(mode,steps,lr,'validation',round(float(vscore),4),'devMSE10median',round(float(np.median(dv[10]['solution_mse'])),6),'devMSE10meanlog',round(float(np.mean(np.log(dv[10]['solution_mse']+1e-12))),5),'halvings',round(float(np.median(dv[10]['line_search_halvings'])),1),flush=True)
