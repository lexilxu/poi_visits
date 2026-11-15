"""Evaluate newly completed recipes on AutoDL with the frozen evaluator."""
from pathlib import Path, PureWindowsPath
import json,sys,time,os,fcntl
ROOT=Path('/root/autodl-tmp/poi_continuation_20261114')
sys.path.insert(0,str(ROOT/'analysis_source_v2/experiments/revision'))
from summarize_queue import evaluate_one,merge,write_json
from audit_protocol import load_panel
lock=(ROOT/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
initial=json.loads((ROOT/'transfer_manifest.json').read_text())
prior={e['path'] for e in initial['files'] if e['path'].endswith('/selected_runs.json')}
out=ROOT/'new_december_results_v2';out.mkdir(exist_ok=True)
cache=out/'run_records';cache.mkdir(exist_ok=True)
raw,dates,attrs=load_panel(Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'))
started=time.monotonic()
while time.monotonic()-started<48*3600:
    changed=False
    for selection in sorted(ROOT.glob('v*/queue/*/selected_runs.json')):
        if str(selection.relative_to(ROOT)) in prior:continue
        selected=json.loads(selection.read_text());r=selected['recipe']
        assert sorted(x['seed'] for x in selected['runs'])==[42,43,44,45,46]
        for run in selected['runs']:
            name=r['id'].replace(' ','_')+f"_seed{run['seed']}.json"
            if (cache/name).exists():continue
            folder=selection.parent/PureWindowsPath(run['directory']).name
            if not (folder/'predictions.npz').exists():continue
            fit=json.loads((folder/'fit.json').read_text())
            assert fit['status']=='complete' and fit['lr']==selected['selected_lr']
            meta={k:r[k] for k in ('id','stage','model','T','H','adjacency','variant','loss','graph','k')}
            meta.update(seed=run['seed'],selected_lr=selected['selected_lr'],validation_MAE=fit['validation_MAE'],best_epoch=fit['best_epoch'])
            result=evaluate_one(folder/'predictions.npz',meta,raw,dates,attrs,r['stage'] in ('core','loss','adaptive'))
            write_json(cache/name,result);changed=True
            row=next(x for x in result['metrics'] if x['cohort']=='All' and x['calibration']=='raw')
            print(json.dumps(dict(recipe=r['id'],seed=run['seed'],MAE=row['MAE'])),flush=True)
    if changed:merge(cache,out)
    if (ROOT/'training_complete.json').exists():break
    time.sleep(60)
records=list(cache.glob('*.json'))
assert len(records)==275, f'Expected 275 new selected runs, found {len(records)}'
write_json(out/'complete.json',dict(selected_runs=len(records),recipes=55,completed_at=time.time()))
