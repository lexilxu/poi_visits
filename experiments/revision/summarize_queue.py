"""Validate saved runs and summarize all five prespecified seeds without selecting on test data."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path, PureWindowsPath
import json
import time
import numpy as np
import pandas as pd
from audit_protocol import load_panel,low_history,EXPECTED_BUNDLE_SHA256
from training_protocol import Protocol
from evaluate_saved import load_events,aligned
from run_baseline_checks import metrics
from controlled_checks import affine_calibration,matched_far_threshold,grid_groups,spatial_coverage

SEEDS=[42,43,44,45,46]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)


def validate(data,raw,dates,attrs):
    t=int(data['T']);h=int(data['H'])
    origins,_,target,split,valid=Protocol(t,h).windows(raw)
    selected=split!='train'
    expected=dict(forecast_origin=np.asarray(dates[origins[selected]],dtype=str),
                  horizon_step=np.arange(1,h+1),poi_id=attrs.placekey.to_numpy(dtype=str),
                  dataset_split=split[selected],valid=valid[selected],observed_visits=raw[target[selected]])
    aligned(data,expected)
    if int(data['train_end'])!=50 or str(data['bundle_sha256'])!=EXPECTED_BUNDLE_SHA256:
        raise ValueError('Unregistered training boundary or dataset')


def evaluate_one(path,meta,raw,dates,attrs,detailed):
    data=load_events(path)
    if str(data.get('density','')):
        from experiment_inputs import density_indices
        nodes,definition=density_indices(raw,str(data['density']))
        raw=raw[:,nodes];attrs=attrs.iloc[nodes].reset_index(drop=True)
        validate(data,raw,dates,attrs)
        test=data['dataset_split']=='test';y=data['observed_visits'][test]
        p=data['predicted_visits'][test];mask=data['valid'][test]
        return dict(meta=meta,metrics=[dict(**meta,cohort='All',calibration='raw',
                    **metrics(y,p,mask,np.ones(len(attrs),bool)))],
                    horizons=[dict(**meta,cohort='All',lead=h+1,
                       MAE=float(np.abs(y[:,h]-p[:,h])[mask[:,h]].mean())) for h in range(int(data['H']))],
                    categories=[],matched_far=[],activity_coverage=[],calibration=None,
                    density=definition,source_prediction_sha256=sha256(path.read_bytes()).hexdigest())
    validate(data,raw,dates,attrs)
    result=evaluate_arrays(data,meta,raw,attrs,detailed)
    result['source_prediction_sha256']=sha256(path.read_bytes()).hexdigest()
    return result


def evaluate_arrays(data,meta,raw,attrs,detailed):
    """Evaluate already aligned events; callers must validate dates and POIs first."""
    y,p=data['observed_visits'],data['predicted_visits']
    test=data['dataset_split']=='test';val=data['dataset_split']=='validation';mask=data['valid']
    low,definition=low_history(raw[:50])
    known_parent=attrs.parent_placekey.notna().to_numpy() & attrs.parent_placekey.fillna('').astype(str).ne('').to_numpy()
    means=np.nanmean(raw[:50],0)
    cohorts={'All':np.ones(len(attrs),bool),'Low-history':low,'Known parent':known_parent,
             'No recorded parent':~known_parent}
    for quantile in (.5,.75,.9):
        cohorts[f'Training mean above Q{int(quantile*100)}']=means>=np.quantile(means,quantile)
    results=[];horizons=[];categories=[];far=[];coverage=[]
    for name,cohort in cohorts.items():
        results.append(dict(**meta,cohort=name,calibration='raw',**metrics(y[test],p[test],mask[test],cohort)))
    for lead in range(int(data['H'])):
        for name,cohort in [('All',cohorts['All']),('Low-history',low)]:
            keep=mask[test][:,lead] & cohort[None,:]
            horizons.append(dict(**meta,cohort=name,lead=lead+1,
                                 MAE=float(np.abs(y[test][:,lead]-p[test][:,lead])[keep].mean())))
    # Categories are the recorded two-digit NAICS codes, defined before outcomes.
    labels=attrs.naics_code.astype('int64').astype(str).str[:2].to_numpy()
    for label in np.unique(labels):
        for name,cohort in [('All',labels==label),('Known parent',(labels==label)&known_parent)]:
            keep=mask[test] & cohort[None,None,:]
            if keep.any():
                categories.append(dict(**meta,NAICS2=label,cohort=name,POIs=int(cohort.sum()),
                                       MAE=float(np.abs(p[test]-y[test])[keep].mean())))
    calibration=None
    if detailed:
        selected=mask & val[:,None,None] & low[None,None,:]
        calibration=affine_calibration(p[selected],y[selected])
        calibrated=np.maximum(0,calibration['a']*p+calibration['c'])
        for name,cohort in [('All',cohorts['All']),('Low-history',low)]:
            results.append(dict(**meta,cohort=name,calibration='calibrated',
                                **metrics(y[test],calibrated[test],mask[test],cohort)))
        for name,pred in [('raw',p),('calibrated',calibrated)]:
            zeros=selected & (y==0)
            keep=mask[test] & low[None,None,:];positive=y[test][keep]>0
            for level in (.05,.1,.2,.3):
                threshold=matched_far_threshold(pred[zeros],level)
                active=pred[test][keep]>=threshold
                far.append(dict(**meta,calibration=name,target_validation_FAR=level,threshold=threshold,
                                validation_FAR=float((pred[zeros]>=threshold).mean()),
                                test_FAR=float(active[~positive].mean()),test_recall=float(active[positive].mean())))
        groups=grid_groups(attrs.latitude,attrs.longitude)
        for fraction in (.05,.1,.2):
            achieved,ideal,k,n=spatial_coverage(y[test],p[test],groups,fraction)
            coverage.append(dict(**meta,budget_fraction=fraction,selected_cells=k,total_cells=n,
                                 coverage=float(np.nanmean(achieved)),oracle_coverage=float(np.nanmean(ideal))))
    return dict(meta=meta,metrics=results,horizons=horizons,categories=categories,matched_far=far,
                activity_coverage=coverage,calibration=calibration,
                low_history_pois=int(low.sum()),low_history_definition=definition)


def selected_jobs(queues):
    jobs=[]
    for queue in queues:
        for path in sorted(queue.glob('*/selected_runs.json')):
            selected=json.loads(path.read_text(encoding='utf8'));r=selected['recipe']
            if sorted(run['seed'] for run in selected['runs'])!=SEEDS:
                raise ValueError(f'Incomplete or duplicate seeds at {path}')
            for run in selected['runs']:
                directory=path.parent/PureWindowsPath(run['directory']).name
                fit=json.loads((directory/'fit.json').read_text(encoding='utf8'))
                if fit['status']!='complete' or fit['seed']!=run['seed'] or fit['lr']!=selected['selected_lr']:
                    raise ValueError(f'Run selection and fitted model disagree: {directory}')
                meta={k:r[k] for k in ('id','stage','model','T','H','adjacency','variant','loss','graph','k')}
                meta.update(seed=run['seed'],selected_lr=selected['selected_lr'],
                            validation_MAE=fit['validation_MAE'],best_epoch=fit['best_epoch'])
                jobs.append((directory/'predictions.npz',meta,r['stage'] in ('core','loss','adaptive')))
    return jobs


def merge(cache,out):
    reports=[json.loads(path.read_text(encoding='utf8')) for path in sorted(cache.glob('*.json'))]
    for key in ('metrics','horizons','categories','matched_far','activity_coverage'):
        rows=[row for report in reports for row in report[key]]
        if not rows:continue
        table=pd.DataFrame(rows)
        table.to_csv(out/f'{key}_by_seed.csv',index=False)
        group=['id','stage','model','T','H','adjacency','variant','loss','graph','k']
        group += {'metrics':['cohort','calibration'],'horizons':['cohort','lead'],
                  'categories':['cohort','NAICS2'],'matched_far':['calibration','target_validation_FAR'],
                  'activity_coverage':['budget_fraction']}[key]
        values=[c for c in table.select_dtypes('number').columns if c not in group+['seed','selected_lr','best_epoch']]
        grouped=table.groupby(group,dropna=False,sort=False)
        counts=grouped['seed'].agg(lambda s:tuple(sorted(s))).rename('seeds')
        # Deterministic baselines have one run; neural rows require every seed.
        keep=counts.apply(lambda s:s==tuple(SEEDS) or s==(-1,))
        summary=grouped[values].agg(['mean','std']).loc[keep]
        summary.columns=['_'.join(c) for c in summary.columns]
        summary=summary.join(counts).reset_index()
        summary.to_csv(out/f'{key}_summary.csv',index=False)
    write_json(out/'evaluation_status.json',dict(evaluated_runs=len(reports),
               neural_recipes=sorted({r['meta']['id'] for r in reports if r['meta']['seed']>=0}),
               seeds=SEEDS,cohort='First 50 equally weighted calendar days',
               selection='Validation-only learning-rate selection; all five fixed seeds retained',
               uncertainty='SD across fitted seeds; no IID event confidence intervals',
               bundle_sha256=EXPECTED_BUNDLE_SHA256))


def main(args):
    raw,dates,attrs=load_panel(args.bundle)
    args.out.mkdir(parents=True,exist_ok=True);cache=args.out/'run_records';cache.mkdir(exist_ok=True)
    start=time.monotonic()
    while True:
        jobs=selected_jobs(args.queues)
        if args.baselines:
            for path in sorted(args.baselines.glob('*_H7.npz')):
                with np.load(path,allow_pickle=False) as z:
                    model=str(z['model']);t=int(z['T'])
                meta=dict(id=f'{model}_T{t}',stage='baseline',model=model,T=t,H=7,
                          adjacency='None',variant='full',loss='deterministic',graph=None,k=None,
                          seed=-1,selected_lr=None,validation_MAE=None,best_epoch=None)
                jobs.append((path,meta,True))
        changed=False
        for path,meta,detailed in jobs:
            filename=meta['id'].replace(' ','_')+f"_seed{meta['seed']}.json"
            destination=cache/filename
            if destination.exists():continue
            result=evaluate_one(path,meta,raw,dates,attrs,detailed)
            write_json(destination,result);changed=True
            row=next(r for r in result['metrics'] if r['cohort']=='All' and r['calibration']=='raw')
            print(json.dumps(dict(id=meta['id'],seed=meta['seed'],test_MAE=row['MAE'])),flush=True)
        if changed or not (args.out/'evaluation_status.json').exists():merge(cache,args.out)
        if not args.watch:break
        states=[json.loads((q/'queue_status.json').read_text()) if (q/'queue_status.json').exists() else {} for q in args.queues]
        if all(s.get('status') in ('complete','stopped') for s in states):break
        if time.monotonic()-start>args.max_hours*3600:break
        time.sleep(10)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--queues',nargs='+',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baselines',type=Path);p.add_argument('--watch',action='store_true')
    p.add_argument('--max-hours',type=float,default=12)
    main(p.parse_args())
