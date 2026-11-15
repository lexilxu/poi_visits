"""Paired controls and artifact audit on AutoDL, reusing all original fits."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import sys
import numpy as np
import pandas as pd


def main(a):
    sys.path.insert(0,str(a.source/'experiments/revision'))
    from summarize_queue import selected_jobs,write_json
    from audit_protocol import load_panel,low_history
    from paired_seed_analysis import aggregate,intervals
    from evaluate_saved import aligned
    from controlled_checks import grid_groups
    raw,dates,attrs=load_panel(a.bundle);low,_=low_history(raw[:50])
    groups={}
    for path,meta,_ in selected_jobs([a.old/'v1/queue',a.old/'v2/queue',a.queue]):
        groups.setdefault(meta['id'],dict(meta=meta,paths=[]))['paths'].append(path)
    pairs=[]
    for key,value in groups.items():
        m=value['meta']
        if m['stage']=='matched':
            pairs += [(key,f"{name}_T{m['T']}_Identity_full_{m['loss']}") for name in ('GraphGRU','Shared-LSTM')]
        if m['stage']=='parent_resolved':
            if m['variant']=='full':pairs.append((key,'GNN-static_T14_Identity_full_MAE'))
            else:pairs.append((key,'GNN-parent_T14_Identity_full_MAE'))
    a.out.mkdir(parents=True,exist_ok=True)
    fit_rows=[]
    for key in sorted({key for pair in pairs for key in pair}):
        assert len(groups[key]['paths'])==5,key
        for path in groups[key]['paths']:
            fit=json.loads((path.parent/'fit.json').read_text());assert fit['status']=='complete'
            assert path.exists() and (path.parent/'checkpoint.pt').exists()
            fit_rows.append(dict(id=key,seed=fit['seed'],model=fit['model'],T=fit['T'],loss=fit['training_loss'],
                                 parameters=fit['parameters'],seconds=fit['seconds'],epochs=len(fit['epochs']),
                                 best_epoch=fit['best_epoch'],lr=fit['lr'],validation_MAE=fit['validation_MAE']))
    pd.DataFrame(fit_rows).to_csv(a.out/'selected_fit_resources.csv',index=False)
    for period in ('december','january'):
        folder=a.out/period;cache=folder/'comparisons';cache.mkdir(parents=True,exist_ok=True)
        loaded={}
        def data(key):
            if key not in loaded:
                paths=groups[key]['paths']
                if period=='january':
                    base=(a.evaluation/'january' if groups[key]['meta']['stage'] in ('matched','parent_resolved')
                          else a.old/'consolidated_final_20261114/january')
                    paths=[base/'private_predictions'/(key.replace(' ','_')+f'_seed{s}.npz') for s in (42,43,44,45,46)]
                loaded[key]=aggregate(paths)
            return loaded[key]
        for model,reference in pairs:
            dest=cache/(model+'__vs__'+reference+'.json')
            if dest.exists():continue
            left,loss_left,hash_left=data(model);right,loss_right,hash_right=data(reference);aligned(left,right)
            test=left['dataset_split']=='test';valid=left['valid'][test];difference=loss_left-loss_right
            cohorts={'All':np.ones(len(attrs),bool),'Low-history':low}
            if groups[model]['meta']['stage']=='parent_resolved':
                cohorts['Known parent']=attrs.parent_placekey.fillna('').astype(str).ne('').to_numpy()
            rows=[]
            for cohort,nodes in cohorts.items():
                mask=valid&nodes[None,None,:];delta=np.where(mask,difference,0.)
                for km in (1.,3.,5.):
                    rows.append(dict(period=period,model=model,reference=reference,cohort=cohort,cell_km=km,
                                     **intervals(delta,mask,grid_groups(attrs.latitude,attrs.longitude,km))))
            write_json(dest,dict(rows=rows,model_predictions=hash_left,reference_predictions=hash_right,
                                estimand='Difference of five-seed mean absolute losses'))
            print(json.dumps(dict(period=period,model=model,reference=reference,status='complete')),flush=True)
        pd.DataFrame([r for p in cache.glob('*.json') for r in json.loads(p.read_text())['rows']]).to_csv(folder/'paired_intervals.csv',index=False)
    fits=list(a.queue.rglob('fit.json'));assert len(fits)==28
    selected=list(a.queue.rglob('selected_runs.json'));assert len(selected)==4
    for path in selected:
        r=json.loads(path.read_text());assert sorted(x['seed'] for x in r['runs'])==[42,43,44,45,46]
        candidates=[json.loads(p.read_text()) for p in path.parent.glob('seed42_*/fit.json')]
        assert len(candidates)==3 and r['selected_lr']==min(candidates,key=lambda x:(x['validation_MAE'],x['lr']))['lr']
    write_json(a.out/'complete.json',dict(fits=28,configurations=4,selected_runs=20,pairs_per_period=len(pairs),
                                        fit_hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in fits}))


if __name__=='__main__':
    p=ArgumentParser()
    for f in ('source','old','queue','evaluation','bundle','out'):p.add_argument('--'+f,type=Path,required=True)
    main(p.parse_args())
