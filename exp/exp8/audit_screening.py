"""Recompute screening with training-defined candidate sets and audit saved errors."""
from argparse import ArgumentParser
from pathlib import Path
import sys
import json
from hashlib import sha256
import numpy as np
import pandas as pd
from exp.exp8.evaluate import ridge_predict, metrics, select_budget, LEADS


def main(a):
    sys.path.insert(0,str(a.source/'exp/common'))
    from exp.common.paired import intervals
    from exp.common.controls import grid_groups
    a.out.mkdir(parents=True,exist_ok=False)
    with np.load(a.inputs/'parking_panel.npz') as z:p={k:z[k] for k in z.files}
    dates=p['dates'];ratio=p['occupancy']/p['capacity'];valid=np.isfinite(ratio)
    train=(dates>='2020-10-01')&(dates<='2020-11-19');tm=valid&train[:,None,None]
    counts=tm.sum(0);candidates=counts>0
    sums=np.where(tm,ratio,0.).sum(0);means=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    weekday=pd.to_datetime(dates).dayofweek.to_numpy();calendar=np.zeros(ratio.shape)
    for day in range(7):
        mask=tm&(weekday[:,None,None]==day)
        calendar[weekday==day]=(np.where(mask,ratio,0.).sum(0)+5*means)/(mask.sum(0)+5)
    features=[calendar];lags=[]
    for lag in (7,14):
        past=np.full(ratio.shape,np.nan);past[lag:]=ratio[:-lag]
        missing=~np.isfinite(past);filled=np.where(missing,means[None],past)
        lags.append(filled);features.extend([filled,missing.astype(float)])
    for hour in (1,2):features.append(np.broadcast_to((np.arange(3)==hour)[None,:,None],ratio.shape))
    for day in range(1,7):features.append(np.broadcast_to((weekday==day)[:,None,None],ratio.shape))
    base=np.stack(features,axis=-1)
    saved=json.loads((a.previous/'parking_only_fit.json').read_text())
    fit=tuple(np.array(saved[k]) for k in ('mean','scale','weights'))
    old=pd.read_csv(a.previous/'metrics_by_seed.csv');records=[];interval_rows=[];audit=[]
    for period in ('december','january'):
        selection_groups={};target=None;baseline_preds=None
        for file in sorted((a.previous/'private_predictions').glob(period+'_*.npz')):
            suffix=file.stem[len(period)+1:];key,seed_radius=suffix.rsplit('_seed',1)
            seed_str,radius_str=seed_radius.split('_r');seed=int(seed_str);radius=int(radius_str)
            with np.load(file) as z:
                pred=z['prediction'];target=(z['target_dates'].astype('datetime64[D]')-np.datetime64(dates[0])).astype(int)
                np.testing.assert_array_equal(z['block_id'],p['block_id'])
            y=ratio[target];cap=p['capacity'][target];occ=p['occupancy'][target]
            for j,lead in enumerate(LEADS):
                result=metrics(pred[:,j],y[:,j],cap[:,j],occ[:,j],p['block_id'],candidates)
                previous=old[(old.period==period)&(old.model==key)&(old.seed==seed)&(old.radius_m==radius)&(old.lead==lead)].iloc[0]
                assert abs(result['ratio_MAE']-previous.ratio_MAE)<1e-7
                assert abs(result['count_MAE']-previous.count_MAE)<1e-7
                records.append(dict(period=period,model=key,seed=seed,radius_m=radius,lead=lead,**result))
            audit.append(dict(path=file.name,sha256=sha256(file.read_bytes()).hexdigest(),MAE_recomputed=True))
            if radius==500:
                selection_groups.setdefault(key,[]).append(np.stack([select_budget(pred[:,-1],np.broadcast_to(candidates,y[:,-1].shape),p['block_id'],f) for f in (.1,.2)]))
        baseline_preds={'Parking calendar':calendar[target],'Parking lag7':lags[0][target],'Parking ridge':ridge_predict(fit,base[target])}
        for key,pred in baseline_preds.items():
            for j,lead in enumerate(LEADS):
                result=metrics(pred[:,j],y[:,j],cap[:,j],occ[:,j],p['block_id'],candidates)
                previous=old[(old.period==period)&(old.model==key)&(old.lead==lead)].iloc[0]
                assert abs(result['ratio_MAE']-previous.ratio_MAE)<1e-7
                records.append(dict(period=period,model=key,seed=-1,radius_m=0,lead=lead,**result))
        selection_groups['Parking ridge']=[np.stack([select_budget(baseline_preds['Parking ridge'][:,-1],np.broadcast_to(candidates,y[:,-1].shape),p['block_id'],f) for f in (.1,.2)])]
        selections={key:np.mean(value,axis=0) for key,value in selection_groups.items()}
        contrasts=[(key,'Parking ridge') for key in selections if key!='Parking ridge']
        contrasts += [(f'{name}_T14_NAICS-2_full_MAE',f'{name}_T14_Identity_full_MAE') for name in ('GraphGRU','Fixed-MTGNN')]
        mask=np.isfinite(y[:,-1]);high=(y[:,-1]>=.85)&mask;paid=np.where(mask,occ[:,-1],0.)
        for model,reference in contrasts:
            for km in (1.,3.,5.):
                groups=grid_groups(p['latitude'],p['longitude'],km)
                for b,f in enumerate((.1,.2)):
                    difference=selections[model][b]-selections[reference][b]
                    for metric,weights in [('high_pressure_recall',high),('paid_occupancy_share',paid)]:
                        row=intervals(difference*weights,weights,groups);row['difference']=row.pop('delta_MAE')
                        interval_rows.append(dict(period=period,model=model,reference=reference,cell_km=km,budget_fraction=f,metric=metric,**row))
    frame=pd.DataFrame(records);frame.to_csv(a.out/'metrics_by_seed.csv',index=False)
    keys=['period','model','radius_m','lead'];summary=frame.groupby(keys)[[c for c in frame if c not in keys+['seed']]].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(a.out/'metrics_summary.csv',index=False)
    pd.DataFrame(interval_rows).to_csv(a.out/'screening_intervals.csv',index=False)
    (a.out/'audit.json').write_text(json.dumps(dict(status='complete',predictions_checked=audit,records=len(records),
        correction='Training-defined blockface/clock-time candidates ranked before applying outcome availability masks; fitted models and magnitude errors retained',
        candidates_per_hour=candidates.sum(1).tolist(),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(status='complete',records=len(records),predictions_checked=len(audit),candidates_per_hour=candidates.sum(1).tolist())),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for f in ('source','inputs','previous','out'):p.add_argument('--'+f,type=Path,required=True)
    main(p.parse_args())
