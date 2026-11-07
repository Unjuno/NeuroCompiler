from __future__ import annotations
import os,sys
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[v]='1'
import stage8_nonlinear as s
s.LR_LIST=[.001,.003,.01,.03,.1,.2,.4,.8]
if __name__=='__main__':
 s.evaluate_seed(int(sys.argv[1]),direct_steps=1200,meta_steps=900,horizon=3,inner_lr=.003)
