"""Prototype symbolic-IR solver router for the tested scalar monotone cubic ODE family.
No probabilistic guarantee, no neural training required. For a *fixed sampled grid*.
mode=fast -> constant initialization + 3 exponential-midpoint Newton residual corrections
mode=accurate -> degree32 Chebyshev-Lobatto nonlinear least squares
Implementation assumption: 1D dimensionless interval [-1,1], lambda>=0, beta>=0.
Not a general PDE solver or formal numerical error certificate.
"""
from __future__ import annotations
import numpy as np
import stage8_nonlinear as exp
from differential_ir import evaluate
from stage8c_correction import newton_correction

def compile_solve(polynomial, lam, beta, boundary, x_eval, *, mode='accurate', descriptors=exp.st.DESC):
    if lam<0 or beta<0:raise ValueError('This prototype only covers dissipative lambda,beta>=0')
    if mode not in ('fast','accurate'):raise ValueError('mode must be fast or accurate')
    x_eval=np.asarray(x_eval,dtype=np.float64)
    if mode=='fast':
        if len(x_eval)<2 or not np.isclose(x_eval[0],-1) or not np.all(np.diff(x_eval)>0):
            raise ValueError('fast mode requires ascending grid beginning at x=-1')
        u=np.full((1,len(x_eval)),boundary)
        f=evaluate(polynomial,x_eval,descriptors)[None,:]
        r=(lam*boundary+beta*boundary**3)-f
        prediction=newton_correction(u,r,np.array([lam]),np.array([beta]),x_eval,rounds=3)[-1][0]
        return prediction,{'backend':'exponential_midpoint_3_newton','iterations':3,'formal_accuracy_certificate':False}
    xnodes=-np.cos(np.pi*np.arange(128)/127)
    f=evaluate(polynomial,xnodes,descriptors)
    prediction,diag=exp.cheb_solve(xnodes,f,lam,beta,boundary,x_eval,32)
    return prediction,{'backend':'cheb_lobatto_nonlinear_lstsq_32','diagnostics':diag,'formal_accuracy_certificate':False}

if __name__=='__main__':
    from fractions import Fraction
    import json
    bank=exp.load('stage8d_test',True);item=bank['meta'][0]
    p={tuple(tuple(a) for a in atoms):Fraction(num,den) for atoms,num,den in item['ir']}
    xx=bank['xq'].numpy()[0].astype(float)
    for mode in ('fast','accurate'):
        u,info=compile_solve(p,item['lambda'],item['beta'],item['boundary'],xx,mode=mode)
        ref=bank['solution'].numpy()[0]
        print(mode,'MSE',float(np.mean((u-ref)**2)),info)
