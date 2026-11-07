from experiment import *
import scipy,subprocess
if __name__=='__main__':
    env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,visible_cpus=os.cpu_count(),cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),cuda=torch.cuda.is_available(),threads=torch.get_num_threads(),cpu_info=subprocess.check_output(['lscpu'],text=True),torch_config=torch.__config__.show())
    (ROOT/'checks'/'environment.json').write_text(json.dumps(env,indent=2))
    # Lock before generating or reading the final evaluation bank.
    (ROOT/'checks'/'protocol_lock.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256((ROOT/'docs'/'protocol.md').read_bytes()).hexdigest(),implementation_sha256=hashlib.sha256((ROOT/'src'/'experiment.py').read_bytes()).hexdigest(),steps=800,seeds=list(range(20)),pilot=100,changed_after_pilot=False),indent=2))
    ps=[]; labels=[]
    for i,g in enumerate(['id','center','width','operator']): ps.append(sample_params(200,90101+i,g)); labels += [g]*200
    make_bank('test',np.concatenate(ps),labels)
    metas=[]
    for seed in range(20):
        times={}
        for method in ['Data','Weak','Response','DataTime']:
            meta=train_one(seed,method,800,seconds=times.get('Response') if method=='DataTime' else None)
            times[method]=meta['train_seconds']; metas.append(meta); print(json.dumps(meta),flush=True)
        pd.DataFrame(metas).to_csv(ROOT/'analysis'/'training.csv',index=False)
    (ROOT/'checks'/'training_complete.json').write_text(json.dumps(dict(runs=len(metas),seeds=20)))
