"""Replay training and follow-ups in a new destination; never overwrite a run."""
from __future__ import annotations
import argparse,os,shutil,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    src=Path(__file__).resolve().parents[1];out=a.output.resolve()
    if out.exists():raise SystemExit(f'Refusing to overwrite existing directory: {out}')
    out.mkdir(parents=True)
    shutil.copytree(src/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(src/'banks',out/'banks')
    for name in ('checks','results','analysis','figures'):(out/name).mkdir()
    shutil.copy2(src/'protocol.json',out/'protocol.json')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for script,args in [('test_differential_ir.py',[]),('stage7_experiment.py',['checks']),('run_training.py',[]),('run_branches.py',[]),('analyze.py',[])]:
        subprocess.run([sys.executable,str(out/'code'/script)]+args,cwd=out,env=env,check=True)
    print(f'Completed fresh training and evaluation: {out}')
if __name__=='__main__':main()
