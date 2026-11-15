from pathlib import Path
import json,sys,traceback,os
import torch
root=Path('/root/autodl-tmp/poi_continuation_20261114')
sys.path.insert(0,str(root/'analysis_source_v2/experiments/revision'))
from evaluate_january_strict_v3 import extended_panel,replay_model,save_and_evaluate
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
if os.environ.get('DETERMINISTIC_REPLAY')=='1':
    torch.use_deterministic_algorithms(True)
recipe,seed=sys.argv[1],int(sys.argv[2])
device=sys.argv[3] if len(sys.argv)>3 else 'cuda:0'
selection=next(root.glob(f'v*/queue/{recipe}/selected_runs.json'))
s=json.loads(selection.read_text());r=s['recipe']
folder=selection.parent/f"seed{seed}_lr{s['selected_lr']:g}"
fit=json.loads((folder/'fit.json').read_text())
torch.manual_seed(seed)
raw,dates,attrs=extended_panel(Path('/root/autodl-tmp/poi_revision_20261114a/approved_panel.zip'),root/'private_validation_202101/extended_panel.npz')
out=root/'replay_investigation_v3';out.mkdir(exist_ok=True)
diagnostic=out/f'{recipe}_seed{seed}_{device.replace(":","_")}_{"deterministic" if os.environ.get("DETERMINISTIC_REPLAY")=="1" else "standard"}.json'
try:
    origins,pred,extra=replay_model(folder,raw,dates,attrs,device,root/'private_graphs',diagnostic)
    meta={k:r[k] for k in ('id','stage','model','T','H','adjacency','variant','loss','graph','k')}
    meta.update(seed=seed,selected_lr=s['selected_lr'],validation_MAE=fit['validation_MAE'],best_epoch=fit['best_epoch'])
    result=root/'new_january_results'
    assert not (result/'run_records'/f'{recipe.replace(" ","_")}_seed{seed}.json').exists()
    save_and_evaluate(result,origins,pred,meta,raw,dates,attrs,extra)
    print(json.dumps(dict(recipe=recipe,seed=seed,strict_passed=extra['strict_replay_passed'])),flush=True)
except Exception as error:
    print(json.dumps(dict(recipe=recipe,seed=seed,error=str(error))),flush=True)
    sys.exit(1)
