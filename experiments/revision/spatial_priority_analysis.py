"""Paired origin-block uncertainty for observed-visit spatial prioritization.

Coverage is calculated for each fitted seed separately, then averaged across
seeds. The comparator is the fixed training-period historical average. These
are observed visits captured by selected grid cells, not transport demand.
"""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from audit_protocol import load_panel
from evaluate_saved import load_events,aligned
from controlled_checks import grid_groups,spatial_coverage
from summarize_queue import selected_jobs,write_json


def paired_interval(difference,repetitions=2000):
    # Average leads within each origin before resampling overlapping forecasts.
    values=difference.mean(1);n=len(values)
    rng=np.random.default_rng(421)
    starts=rng.integers(n,size=(repetitions,int(np.ceil(n/7))))
    origins=((starts[:,:,None]+np.arange(7))%n).reshape(repetitions,-1)[:,:n]
    lo,hi=np.quantile(values[origins].mean(1),[.025,.975])
    return float(lo),float(hi)


def run(args):
    _,_,attrs=load_panel(args.bundle)
    groups=grid_groups(attrs.latitude,attrs.longitude)
    models={}
    for path,meta,_ in selected_jobs(args.queues):
        if meta['stage'] not in ('core','adaptive'):continue
        if args.january:path=args.january/'private_predictions'/(meta['id'].replace(' ','_')+f"_seed{meta['seed']}.npz")
        if not path.exists():raise FileNotFoundError(path)
        models.setdefault(meta['id'],dict(meta=meta,paths=[]))['paths'].append(path)
    required=[f'{model}_T{t}_{adj}_full_MAE' for t in (14,28) for model,adj in
              [('GraphGRU','Identity'),('Fixed-MTGNN','Identity'),('Shared-LSTM','Identity'),
               ('GNN-static','Identity'),('Adaptive-MTGNN','Adaptive')]]
    if any(key not in models or len(models[key]['paths'])!=5 for key in required):
        raise ValueError('The five-seed core and adaptive models must be complete')
    for t in (14,28):
        for model in ('Calendar HA','Weekday HA','Seasonal naive','NLinear ridge','TSB'):
            path=(args.january/'private_predictions'/f'{model.replace(" ","_")}_T{t}_seed-1.npz' if args.january
                  else args.baselines/f'{model.replace(" ","_")}_T{t}_H7.npz')
            models[f'{model}_T{t}']=dict(meta=dict(model=model,T=t,adjacency='None'),paths=[path])
    computed={};sources={};rows=[]
    for key,entry in models.items():
        reference=None;coverage=[];seeds=[];hashes=[]
        for path in entry['paths']:
            data=load_events(path)
            if reference is None:reference=data
            else:aligned(data,reference)
            np.testing.assert_array_equal(data['poi_id'],attrs.placekey.to_numpy(dtype=str))
            test=data['dataset_split']=='test'
            if not data['valid'][test].all():raise ValueError('Missing scored visits need an explicit coverage policy')
            y=data['observed_visits'][test];p=data['predicted_visits'][test]
            values=[]
            for fraction in (.05,.1,.2):
                achieved,ideal,k,n=spatial_coverage(y,p,groups,fraction)
                values.append(achieved)
                rows.append(dict(id=key,**{c:entry['meta'][c] for c in ('model','T','adjacency')},
                    seed=int(data['seed']),budget_fraction=fraction,selected_cells=k,total_cells=n,
                    coverage=float(achieved.mean()),oracle_coverage=float(ideal.mean())))
            coverage.append(np.stack(values));seeds.append(int(data['seed']));hashes.append(sha256(path.read_bytes()).hexdigest())
        if sorted(seeds) not in ([-1],[42,43,44,45,46]):raise ValueError('Incomplete seed set')
        computed[key]=(reference,np.mean(coverage,axis=0));sources[key]=hashes
    comparisons=[]
    for key,(data,values) in computed.items():
        meta=models[key]['meta'];baseline=f"Calendar HA_T{meta['T']}"
        reference,baseline_values=computed[baseline];aligned(data,reference)
        for i,fraction in enumerate((.05,.1,.2)):
            difference=values[i]-baseline_values[i]
            lo,hi=paired_interval(difference)
            comparisons.append(dict(id=key,reference=baseline,model=meta['model'],T=meta['T'],
                adjacency=meta['adjacency'],budget_fraction=fraction,coverage=float(values[i].mean()),
                gain_over_HA=float(difference.mean()),origin_7day_low=lo,origin_7day_high=hi))
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out/'coverage_by_seed.csv',index=False)
    pd.DataFrame(comparisons).to_csv(args.out/'paired_coverage.csv',index=False)
    write_json(args.out/'protocol.json',dict(status='complete',period='January' if args.january else 'December',
        grid='1 km occupied cells, fixed equirectangular coordinates at 47.6 degrees latitude',
        estimand='Mean of per-seed observed-visit coverage, not coverage of ensemble predictions',
        interval='2,000 paired circular seven-origin block bootstrap draws, conditional on the fixed city, cohort, and fitted seeds',
        interpretation='Observed visits captured by a fixed spatial inspection budget; no transport ridership or operational benefit is estimated',
        sources=sources,analysis_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status='complete',models=len(computed),period='January' if args.january else 'December')))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--queues',nargs='+',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True)
    p.add_argument('--january',type=Path);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
