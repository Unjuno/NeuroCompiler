"""Replay selected arms without replacing the actual saved experiment."""
import experiment as e
import json,torch,shutil
from pathlib import Path
root=e.ROOT; replay=root/'checks'/'replay_temp'; replay.mkdir(exist_ok=True)
for d in ['models','analysis']: (replay/d).mkdir(exist_ok=True)
(replay/'data').symlink_to(root/'data',target_is_directory=True)
results=[]
for method in ['Data','Response','DataTime']:
    old=torch.load(root/'models'/f'{method}_00.pt',weights_only=False)
    e.ROOT=replay
    meta=e.train_one(0,method,steps=old['meta']['steps'],seconds=None)
    new=torch.load(replay/'models'/f'{method}_00.pt',weights_only=False)
    diff=max(float(torch.max(torch.abs(new['state'][k]-v))) for k,v in old['state'].items())
    results.append(dict(method=method,seed=0,recorded_steps=old['meta']['steps'],max_state_difference=diff,same_best_step=meta['best_step']==old['meta']['best_step']))
    assert diff==0 and results[-1]['same_best_step']
e.ROOT=root
(root/'checks'/'replay.json').write_text(json.dumps(results,indent=2))
shutil.rmtree(replay)
print(json.dumps(results,indent=2))
