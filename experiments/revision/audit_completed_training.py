from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
r=Path('/root/autodl-tmp/poi_continuation_20261114')
recipes=[];fits=0
for v in ('v1','v2'):
    plan=json.loads((r/v/'continuation_plan.json').read_text())
    for recipe in plan['recipes']:
        folder=r/v/'queue'/recipe['id']
        selection=json.loads((folder/'selected_runs.json').read_text())
        tuning=[json.loads((folder/f'seed42_lr{lr:g}/fit.json').read_text()) for lr in (.0003,.001,.003)]
        chosen=min(tuning,key=lambda x:(x['validation_MAE'],x['lr']))
        assert selection['selected_lr']==chosen['lr']
        assert sorted(x['seed'] for x in selection['runs'])==[42,43,44,45,46]
        for seed,lr in [(42,lr) for lr in (.0003,.001,.003)]+[(s,chosen['lr']) for s in (43,44,45,46)]:
            d=folder/f'seed{seed}_lr{lr:g}';f=json.loads((d/'fit.json').read_text())
            assert f['status']=='complete' and f['seed']==seed and f['lr']==lr
            assert f['T']==recipe['T'] and f['H']==7 and f['train_end']==50 and f['validation_end']==64
            for name in ('checkpoint.pt','predictions.npz','epoch_log.json'):
                assert (d/name).is_file() and (d/name).stat().st_size>0,(d,name)
            fits+=1
        recipes.append(recipe['id'])
assert fits==497 and len(recipes)==71
report=dict(as_of_utc=datetime.now(timezone.utc).isoformat(),fits=fits,recipes=len(recipes),
    selected_runs=355,checks=['fit completion','artifact presence','validation-only LR selection','seeds 42-46','H7 split boundaries'],
    training_completed_at=json.loads((r/'training_complete.json').read_text())['completed_at'])
out=r/'training_completion_audit.json'
with out.open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
