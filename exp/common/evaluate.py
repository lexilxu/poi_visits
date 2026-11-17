"""Evaluate saved predictions without fitting or selecting on test outcomes.

Inputs use the common event-array schema. Aggregate outputs can be shared;
event CSV exports contain licensed POI identifiers and must remain private.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import gzip
import json
import numpy as np
import pandas as pd
from exp.common.audit import load_panel, low_history, window_arrays
from exp.common.protocol import Protocol
from exp.exp1.baselines import metrics
from exp.common.controls import affine_calibration, matched_far_threshold, grid_groups, spatial_coverage


FIELDS = ('model','adjacency','seed','forecast_origin','horizon_step','poi_id',
          'observed_visits','predicted_visits','dataset_split')


def load_events(path):
    with np.load(path, allow_pickle=False) as z:
        missing = set(FIELDS)-set(z.files)
        if missing:
            raise ValueError(f'{path.name}: missing {sorted(missing)}')
        data = {key:z[key] for key in z.files}
    shape = (len(data['forecast_origin']),len(data['horizon_step']),len(data['poi_id']))
    if data['predicted_visits'].shape != shape or data['observed_visits'].shape != shape:
        raise ValueError('Event axes do not match prediction/observation arrays')
    if len(np.unique(data['poi_id'])) != shape[2] or len(np.unique(data['forecast_origin'])) != shape[0]:
        raise ValueError('Duplicate event-axis identifiers')
    if not np.array_equal(data['horizon_step'],np.arange(1,shape[1]+1)):
        raise ValueError('Forecast leads must be consecutive, starting at one')
    prediction = data['predicted_visits']
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Predictions must be finite and nonnegative')
    data['valid'] = data.get('valid',np.isfinite(data['observed_visits']))
    if np.any(data['valid'] & ~np.isfinite(data['observed_visits'])):
        raise ValueError('Observed mask includes a missing outcome')
    if data['valid'].shape != shape or data['dataset_split'].shape != (shape[0],):
        raise ValueError('Invalid mask or split shape')
    return data


def aligned(a,b):
    for key in ('forecast_origin','horizon_step','poi_id','dataset_split','valid'):
        if not np.array_equal(a[key],b[key]):
            raise ValueError(f'Paired comparison misaligned on {key}')
    if not np.allclose(a['observed_visits'],b['observed_visits'],equal_nan=True,rtol=0,atol=0):
        raise ValueError('Paired comparison has different realized outcomes')


def paired_intervals(y,p,q,valid,groups,seed=421,repetitions=2000):
    """Origin blocks and spatial groups preserve paired models and event clusters.

    Spatial resampling is a sensitivity analysis on this panel, not evidence
    about an independent population or city. All leads travel with their origin.
    """
    n,_,_ = y.shape
    ng = int(groups.max())+1
    difference = np.where(valid,np.abs(p-y)-np.abs(q-y),0.)
    sums = np.column_stack([difference[:,:,groups==g].sum((1,2)) for g in range(ng)])
    counts = np.column_stack([valid[:,:,groups==g].sum((1,2)) for g in range(ng)])
    if counts.sum() == 0:
        raise ValueError('No scored paired events')
    result = {'delta_MAE':float(sums.sum()/counts.sum()),'origins':n,'spatial_groups':ng}
    rng = np.random.default_rng(seed)
    draws = {name:[] for name in ('origin_7day','spatial_group','joint')}
    for _ in range(repetitions):
        origins = ((rng.integers(n,size=int(np.ceil(n/7)))[:,None]+np.arange(7))%n).ravel()[:n]
        weights = np.bincount(rng.integers(ng,size=ng),minlength=ng)
        totals = [(sums[origins].sum(),counts[origins].sum()),
                  (sums.sum(0)@weights,counts.sum(0)@weights),
                  (sums[origins].sum(0)@weights,counts[origins].sum(0)@weights)]
        for name,(num,den) in zip(draws,totals):
            if den:
                draws[name].append(float(num/den))
    for name,values in draws.items():
        result[name+'_CI95'] = (np.quantile(values,[.025,.975]).tolist()
                                    if values and (name=='spatial_group' or n>=14) else None)
    return result


def export_events(data,path,limit=None):
    """Stream the full rectangular event table; missing observations stay blank."""
    opener = gzip.open if path.suffix == '.gz' else open
    path.parent.mkdir(parents=True,exist_ok=True)
    count = 0
    with opener(path,'wt',newline='',encoding='utf8') as f:
        writer = csv.writer(f); writer.writerow((*FIELDS,'observed'))
        for i,origin in enumerate(data['forecast_origin']):
            for h,lead in enumerate(data['horizon_step']):
                for j,poi in enumerate(data['poi_id']):
                    if limit is not None and count>=limit:
                        return count
                    observed = bool(data['valid'][i,h,j])
                    writer.writerow((str(data['model']),str(data['adjacency']),int(data['seed']),
                                     str(origin),int(lead),str(poi),
                                     float(data['observed_visits'][i,h,j]) if observed else '',
                                     float(data['predicted_visits'][i,h,j]),str(data['dataset_split'][i]),observed))
                    count += 1
    return count


def evaluate(paths,reference_path,bundle,out,t,train_end):
    raw,dates,attrs = load_panel(bundle)
    reference = load_events(reference_path)
    h = len(reference['horizon_step'])
    windows = Protocol(T=t,H=h,train_end=train_end).windows(raw) if train_end==50 else window_arrays(raw,t,h)
    if train_end not in (50,57):
        raise ValueError('Only registered train_end=50/57 protocols are supported')
    origins,history,target,split,_ = windows
    keep = split != 'train'
    if not np.array_equal(reference['forecast_origin'],dates[origins[keep]]):
        raise ValueError('Saved forecast origins differ from the selected protocol')
    if not np.allclose(reference['observed_visits'],raw[target[keep]],equal_nan=True,rtol=0,atol=0):
        raise ValueError('Saved observations differ from the approved panel')
    cohort,_ = low_history(raw[:train_end] if train_end==50 else raw[history[split=='train']].reshape(-1,len(attrs)))
    cohort_name = 'Calendar low-history' if train_end==50 else 'Window-weighted low-history'
    mask = reference['valid']
    test = reference['dataset_split']=='test'
    val = reference['dataset_split']=='validation'
    if not val.any() or not test.any():
        raise ValueError('Both validation and test predictions are required')
    if not np.array_equal(reference['poi_id'],attrs.placekey.to_numpy(dtype=str)):
        raise ValueError('Prediction POI order differs from approved panel')
    rows,far_rows,intervals,calibrations,coverage = [],[],[],[],[]
    for path in paths:
        data = load_events(path); aligned(data,reference)
        if 'train_end' in data and int(data['train_end']) != train_end:
            raise ValueError('Prediction training boundary does not match evaluation protocol')
        meta = {key:data[key].item() for key in ('model','adjacency','seed')}
        y,p = data['observed_visits'],data['predicted_visits']
        selected = mask & val[:,None,None] & cohort[None,None,:]
        calibration = affine_calibration(p[selected],y[selected])
        calibrations.append({**meta,**calibration})
        for variant,pred in [('raw',p),('calibrated',np.maximum(0,calibration['a']*p+calibration['c']))]:
            for name,selection in [('All',np.ones(len(attrs),bool)),(cohort_name,cohort)]:
                rows.append({**meta,'variant':variant,'cohort':name,**metrics(y[test],pred[test],mask[test],selection)})
            for level in (.05,.1,.2,.3):
                threshold = matched_far_threshold(pred[selected & (y==0)],level)
                keep = mask[test] & cohort[None,None,:]
                positive = y[test][keep]>0; detected = pred[test][keep]>=threshold
                far_rows.append({**meta,'variant':variant,'target_validation_FAR':level,'threshold':threshold,
                                 'test_FAR':float(detected[~positive].mean()),'test_recall':float(detected[positive].mean())})
        # Compare raw predictions with the same raw reference; no test-based selection.
        for cell in (1.,3.,5.):
            groups = grid_groups(attrs.latitude,attrs.longitude,cell)
            intervals.append({**meta,'cell_km':cell,**paired_intervals(y[test],p[test],reference['predicted_visits'][test],mask[test],groups)})
        for fraction in (.05,.1,.2):
            achieved,ideal,k,n = spatial_coverage(y[test],p[test],grid_groups(attrs.latitude,attrs.longitude),fraction)
            coverage.append({**meta,'selected_fraction':fraction,'selected_cells':k,'total_cells':n,
                             'coverage':float(np.nanmean(achieved)),'oracle_coverage':float(np.nanmean(ideal))})
        print(json.dumps({**meta,'evaluation':'complete'}),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    for name,values in [('saved_metrics',rows),('saved_matched_far',far_rows),('saved_activity_coverage',coverage)]:
        pd.DataFrame(values).to_csv(out/f'{name}.csv',index=False)
    (out/'saved_evaluation.json').write_text(json.dumps({
        'T':t,'H':h,'train_end':train_end,'reference':reference_path.name,
        'calibration_cohort':cohort_name+', validation only',
        'calibration':calibrations,'paired_comparisons':intervals,
        'uncertainty_scope':'Paired seven-origin blocks and occupied-grid sensitivity within this Seattle panel.',
        'activity_scope':'Coverage of observed visits; not transit demand or measured operational savings.'
    },indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--predictions',nargs='+',type=Path,required=True)
    p.add_argument('--reference',type=Path); p.add_argument('--bundle',type=Path)
    p.add_argument('--out',type=Path); p.add_argument('--T',type=int,default=14)
    p.add_argument('--train-end',type=int,choices=(50,57),default=50)
    p.add_argument('--export-events',type=Path); p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.export_events:
        if len(args.predictions)!=1:
            p.error('Event export takes exactly one prediction file')
        print(export_events(load_events(args.predictions[0]),args.export_events,args.limit))
    else:
        if None in (args.reference,args.bundle,args.out):
            p.error('--reference, --bundle, and --out are required for evaluation')
        evaluate(args.predictions,args.reference,args.bundle,args.out,args.T,args.train_end)
