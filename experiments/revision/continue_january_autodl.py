"""Evaluate new full-panel fits with frozen January parameters and replay checks."""
from pathlib import Path,PureWindowsPath
import json,sys,time,traceback,fcntl
ROOT=Path('/root/autodl-tmp/poi_continuation_20261114')
sys.path.insert(0,str(ROOT/'analysis_source_v2/experiments/revision'))
import torch
from evaluate_january_strict_v2 import extended_panel,replay_model,save_and_evaluate
from summarize_queue import merge,write_json
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
lock=(ROOT/'january.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior={e['path'] for e in json.loads((ROOT/'transfer_manifest.json').read_text())['files'] if e['path'].endswith('/selected_runs.json')}
raw,dates,attrs=extended_panel(Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'),ROOT/'private_validation_202101/extended_panel.npz')
out=ROOT/'new_january_results';out.mkdir(exist_ok=True)
(out/'run_records').mkdir(exist_ok=True)
(out/'replay_diagnostics').mkdir(exist_ok=True)
failures=[];planned=0
for selection in sorted(ROOT.glob('v*/queue/*/selected_runs.json')):
    if str(selection.relative_to(ROOT)) in prior:continue
    selected=json.loads(selection.read_text());r=selected['recipe']
    if r['stage']=='density':continue
    assert sorted(x['seed'] for x in selected['runs'])==[42,43,44,45,46]
    for run in selected['runs']:
        planned+=1
        name=r['id'].replace(' ','_')+f"_seed{run['seed']}"
        if (out/'run_records'/f'{name}.json').exists():continue
        directory=selection.parent/PureWindowsPath(run['directory']).name
        fit=json.loads((directory/'fit.json').read_text())
        meta={k:r[k] for k in ('id','stage','model','T','H','adjacency','variant','loss','graph','k')}
        meta.update(seed=run['seed'],selected_lr=selected['selected_lr'],validation_MAE=fit['validation_MAE'],best_epoch=fit['best_epoch'])
        try:
            origins,pred,extra=replay_model(directory,raw,dates,attrs,'cuda:0',ROOT/'private_graphs')
            save_and_evaluate(out,origins,pred,meta,raw,dates,attrs,extra)
        except Exception as error:
            failure=dict(id=r['id'],seed=run['seed'],error=str(error),traceback=traceback.format_exc())
            failures.append(failure)
            path=out/'replay_diagnostics'/f'{name}_{time.time_ns()}.json'
            write_json(path,failure)
            print(json.dumps(dict(failed=name,error=str(error))),flush=True)
        torch.cuda.empty_cache()
    if list((out/'run_records').glob('*.json')):merge(out/'run_records',out)
assert planned==215,planned
completed=len(list((out/'run_records').glob('*.json')))
write_json(out/'execution_status.json',dict(planned=planned,completed=completed,failures=failures,
    status='complete' if completed==planned else 'requires_replay_review',finished_at=time.time()))
