#!/usr/bin/env python3
"""Spot-check prominent cross-stage claims against actual saved aggregate CSVs.

Checks arithmetic/source transcription only; does not rerun training or prove causality.
"""
from pathlib import Path
import csv,math,sys
ROOT=Path(__file__).resolve().parents[1]
checks=[]

def read(rel):
 with (ROOT/rel).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def check(label,rows,selector,field,expected,tol=0.00006):
 values=[r for r in rows if all(r.get(k)==v for k,v in selector.items())]
 if len(values)!=1:raise ValueError(f'{label}: expected exactly one row, got {len(values)}')
 value=float(values[0][field]);passed=math.isfinite(value) and abs(value-expected)<=tol
 checks.append((label,passed,value,expected))

s1=read('experiments/stage-01e/results/stage1e_contrasts.csv')
check('Stage1e h10 held-out AUC',s1,{'a':'MetaInit_h10','b':'DirectFit_matched_h10','metric':'rel_log_auc'},'mean_diff',-.6889)
s2=read('experiments/stage-02/results/stage2_contrasts.csv')
check('Stage2a 8 primitive meta effect',s2,{'experiment':'Stage2a_cross4','contrast':'MetaInit_h10 - DirectFit_matched_h10'},'mean',-.308567)
s3=read('experiments/stage-03/results/stage3a_contrasts.csv')
check('Stage3a generic DeepSets negative',s3,{'contrast':'SetMeta - SetMatchedDirect, structural masks avg'},'mean',-.0563847357)
s9=read('experiments/stage-09r/analysis/summary.csv')
check('Stage9R Data center OOD relative L2',s9,{'method':'Data','group':'center'},'relative_l2_median',1.04464)
check('Stage9R Response center OOD relative L2',s9,{'method':'Response','group':'center'},'relative_l2_median',1.016598)
s9c=read('experiments/stage-09r/analysis/contrasts.csv')
check('Stage9R Response/Data relative improvement',s9c,{'group':'center','numerator':'Response','denominator':'Data'},'ratio',.979188)
for label,passed,value,expected in checks:
 print(('PASS' if passed else 'FAIL')+f' {label}: stored={value:.8g}, reported={expected:.8g}')
print('Reconciled',sum(x[1] for x in checks),'/',len(checks),'source CSV claims.')
sys.exit(0 if all(x[1] for x in checks) else 1)
