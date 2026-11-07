from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,subprocess,sys,os,time
ROOT=Path(__file__).resolve().parents[1]
def one(seed):
    with (ROOT/'results'/f'train_{seed}.log').open('w') as f:
        p=subprocess.run([sys.executable,str(ROOT/'code'/'stage7_experiment.py'),'run','--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    return {'seed':seed,'returncode':p.returncode}
if __name__=='__main__':
    results=[];t=time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(one,s) for s in range(101,111)]):
            results.append(f.result());(ROOT/'results'/'training_status.json').write_text(json.dumps({'results':results,'elapsed':time.perf_counter()-t},indent=2));print(results[-1],flush=True)
    if any(r['returncode'] for r in results):sys.exit(1)
