"""Paired uncertainty for seed-averaged losses, never prediction ensembles."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import time
import numpy as np
import pandas as pd
from exp.common.audit import load_panel, low_history
from exp.common.evaluate import load_events, aligned
from exp.common.summarize import selected_jobs, write_json
from exp.common.controls import grid_groups


def aggregate(paths):
    reference=None;loss=None;seeds=[];hashes=[]
    for path in paths:
        data=load_events(path)
        if reference is None:reference=data
        else:aligned(data,reference)
        test=data['dataset_split']=='test'
        current=np.where(data['valid'][test],np.abs(data['predicted_visits'][test]-data['observed_visits'][test]),0.).astype(float)
        if loss is None:loss=current
        else:loss+=current
        seeds.append(int(data['seed']));hashes.append(sha256(path.read_bytes()).hexdigest())
    if sorted(seeds) not in ([-1],[42,43,44,45,46]):raise ValueError('Seed set is incomplete')
    return reference,loss/len(paths),hashes


def intervals(difference,valid,groups,repetitions=2000):
    n=valid.shape[0]; ng=int(groups.max())+1
    sums=np.stack([np.bincount(groups,weights=difference[i].sum(0),minlength=ng) for i in range(n)])
    counts=np.stack([np.bincount(groups,weights=valid[i].sum(0),minlength=ng) for i in range(n)])
    selected=counts.sum(0)>0;sums=sums[:,selected];counts=counts[:,selected];ng=int(selected.sum())
    if not ng:raise ValueError('Empty comparison cohort')
    rng=np.random.default_rng(421);draws=np.empty((repetitions,3))
    for i in range(repetitions):
        origins=((rng.integers(n,size=int(np.ceil(n/7)))[:,None]+np.arange(7))%n).ravel()[:n]
        weights=np.bincount(rng.integers(ng,size=ng),minlength=ng)
        draws[i]=[sums[origins].sum()/counts[origins].sum(),
                  (sums.sum(0)@weights)/(counts.sum(0)@weights),
                  (sums[origins].sum(0)@weights)/(counts[origins].sum(0)@weights)]
    row=dict(delta_MAE=float(sums.sum()/counts.sum()),origins=n,spatial_groups=ng)
    for k,name in enumerate(('origin_7day','spatial_group','joint')):
        lo,hi=np.quantile(draws[:,k],[.025,.975]);row[name+'_low']=float(lo);row[name+'_high']=float(hi)
    return row


def collected(args):
    groups={}
    for path,meta,_ in selected_jobs(args.queues):
        if args.january:
            path=args.january/'private_predictions'/(meta['id'].replace(' ','_')+f"_seed{meta['seed']}.npz")
            if not path.exists():continue
        groups.setdefault(meta['id'],dict(meta=meta,paths=[]))['paths'].append(path)
    groups={key:value for key,value in groups.items() if len(value['paths'])==5}
    for t in (14,28):
        for name in ('NLinear ridge','Calendar HA','Recent mean','TSB'):
            path=(args.january/'private_predictions'/f'{name.replace(" ","_")}_T{t}_seed-1.npz'
                  if args.january else args.baselines/f'{name.replace(" ","_")}_T{t}_H7.npz')
            if path.exists():groups[f'{name}_T{t}']=dict(meta=dict(id=f'{name}_T{t}',T=t,stage='baseline'),paths=[path])
    return groups


def pairs(groups):
    result=[]
    for key,value in groups.items():
        meta=value['meta'];t=meta['T'];stage=meta['stage']
        if stage in ('core','loss','adaptive','baseline'):
            refs=[f'NLinear ridge_T{t}',f'Calendar HA_T{t}']
            if stage=='loss':refs.append(key.replace('_MSE','_MAE'))
            if stage=='core' and meta['adjacency'] in ('NAICS-2','NAICS-4'):
                refs.append(f"{meta['model']}_T{t}_Identity_full_MAE")
        elif stage=='graph':
            refs=[f"{meta['model']}_T{t}_Identity_full_MAE"]
            if meta['graph']!='Random':refs.append(key.rsplit('_',2)[0]+f"_Random_k{meta['k']}")
        elif stage=='ablation':refs=[f'GNN-static_T{t}_Identity_full_MAE']
        elif stage=='spatial':refs=[f"{meta['model']}_T{t}_Identity_full_MAE"]
        elif stage=='metadata':
            refs=[f"GNN-static_T{t}_Identity_{meta['variant']}_MAE"]
            if meta['variant']=='without_parent':
                refs.append(f'GNN-static_T{t}_Identity_full_MAE_early_metadata')
        else:continue
        for ref in refs:
            if ref in groups and key!=ref:result.append((key,ref))
    return result


def main(args):
    raw,dates,attrs=load_panel(args.bundle)
    low,_=low_history(raw[:50]);parent=attrs.parent_placekey.fillna('').astype(str).ne('').to_numpy()
    args.out.mkdir(parents=True,exist_ok=True);cache=args.out/'comparisons';cache.mkdir(exist_ok=True)
    start=time.monotonic()
    while True:
        groups=collected(args);loaded={};changed=False
        for model,reference in pairs(groups):
            destination=cache/(model.replace(' ','_')+'__vs__'+reference.replace(' ','_')+'.json')
            if destination.exists():continue
            for key in (model,reference):
                if key not in loaded:loaded[key]=aggregate(groups[key]['paths'])
            a,loss_a,hash_a=loaded[model];b,loss_b,hash_b=loaded[reference];aligned(a,b)
            np.testing.assert_array_equal(a['poi_id'],attrs.placekey.to_numpy(dtype=str))
            test=a['dataset_split']=='test';valid=a['valid'][test];difference=loss_a-loss_b
            cohorts={'All':np.ones(len(attrs),bool),'Low-history':low}
            if groups[model]['meta']['stage'] in ('ablation','metadata'):cohorts['Known parent']=parent
            rows=[]
            for cohort,nodes in cohorts.items():
                mask=valid & nodes[None,None,:];delta=np.where(mask,difference,0.)
                for km in (1.,3.,5.):
                    rows.append(dict(model=model,reference=reference,cohort=cohort,cell_km=km,
                                     **intervals(delta,mask,grid_groups(attrs.latitude,attrs.longitude,km))))
            write_json(destination,dict(rows=rows,model_predictions=hash_a,reference_predictions=hash_b,
              estimand='Difference of five-seed mean absolute losses, not MAE of averaged predictions',
              uncertainty='Paired seven-origin blocks and 1/3/5-km occupied-cell sensitivity, conditional on the fixed fitted seeds and Seattle panel'))
            changed=True;print(json.dumps(dict(model=model,reference=reference,status='complete')),flush=True)
        if changed:
            rows=[row for path in cache.glob('*.json') for row in json.loads(path.read_text())['rows']]
            pd.DataFrame(rows).to_csv(args.out/'paired_intervals.csv',index=False)
        if not args.watch or time.monotonic()-start>args.max_hours*3600:break
        time.sleep(30)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--queues',nargs='+',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--baselines',type=Path,required=True);p.add_argument('--january',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--max-hours',type=float,default=12)
    main(p.parse_args())
