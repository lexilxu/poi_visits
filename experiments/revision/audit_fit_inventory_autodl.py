"""Audit the complete frozen queues and export a data-free fitting inventory."""
from pathlib import Path,PureWindowsPath
from hashlib import sha256
from collections import Counter
import csv
import json
import time
import zipfile
root=Path('/root/autodl-tmp/poi_continuation_20261114')
out=root/'final_fit_audit';out.mkdir(exist_ok=True)
def digest(path):
    h=sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
rows=[];sources=[];recipes=0
for name in ('v1','v2'):
    run=root/name;queue=run/'queue'
    execution=json.loads((run/'execution.json').read_text())
    archive_digest=digest(run/'training_source.zip')
    assert archive_digest==execution['source_archive_sha256']
    with zipfile.ZipFile(run/'training_source.zip') as archive:
        source_hashes={name:sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    sources.append(dict(queue=name,source_archive_sha256=archive_digest))
    manifest=json.loads((run/'continuation_plan.json').read_text())
    for recipe in manifest['recipes']:
        recipes+=1;folder=queue/recipe['id']
        selected=json.loads((folder/'selected_runs.json').read_text())
        assert sorted(x['seed'] for x in selected['runs'])==[42,43,44,45,46]
        tuning=[json.loads(p.read_text()) for p in folder.glob('seed42_*/fit.json')]
        assert sorted(f['lr'] for f in tuning)==[.0003,.001,.003]
        winner=min(tuning,key=lambda f:(f['validation_MAE'],f['lr']))
        assert selected['selected_lr']==winner['lr']
        selected_names={PureWindowsPath(x['directory']).name for x in selected['runs']}
        for p in sorted(folder.glob('*/fit.json')):
            fit=json.loads(p.read_text());d=p.parent
            assert fit['status']=='complete' and fit['H']==7
            assert fit['source_sha256']==source_hashes,str(p)
            assert fit['train_end']==50 and fit['validation_end']==64
            assert fit['origins']['validation']==8 and fit['origins']['test']==22
            for required in ('epoch_log.json','checkpoint.pt','predictions.npz'):assert (d/required).is_file()
            rows.append(dict(queue=name,recipe=recipe['id'],stage=recipe['stage'],model=fit['model'],
                adjacency=fit['adjacency'],T=fit['T'],H=fit['H'],variant=fit['variant'],loss=fit['training_loss'],
                seed=fit['seed'],learning_rate=fit['lr'],selected=d.name in selected_names,
                validation_MAE=fit['validation_MAE'],best_epoch=fit['best_epoch'],completed_epochs=len(fit['epochs']),
                parameters=fit['parameters'],seconds=fit['seconds'],gpu=fit['environment'].get('gpu'),
                torch=fit['environment'].get('torch'),fit_sha256=digest(p),
                checkpoint_sha256=digest(d/'checkpoint.pt'),predictions_sha256=digest(d/'predictions.npz'),
                source_archive_sha256=archive_digest))
assert recipes==71 and len(rows)==497 and sum(r['selected'] for r in rows)==355
with (out/'fitting_inventory.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=dict(status='passed',fits=497,recipes=71,selected_seeds=355,
            fits_by_gpu=dict(Counter(r['gpu'] for r in rows)),sources=sources,
            inventory_sha256=digest(out/'fitting_inventory.csv'),completed_at=time.time())
(out/'fit_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
