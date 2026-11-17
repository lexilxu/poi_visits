"""Frozen two-stage paid-parking forecast experiment; execute on AutoDL4."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse

PENALTIES=(.0001,.01,1.,100.)
LEADS=(1,3,7)


def ridge_fit(x,y,penalty):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    regularizer=np.eye(design.shape[1])*penalty;regularizer[0,0]=0
    weights=np.linalg.solve(design.T@design/len(x)+regularizer,design.T@y/len(x))
    return mean,scale,weights


def ridge_predict(model,x):
    mean,scale,w=model
    return np.maximum(0,w[0]+((x-mean)/scale)@w[1:])


def spatial_operator(attrs,lat,lon,radius):
    rad=np.pi/180;rows=[];cols=[];counts=[]
    a=attrs.latitude.to_numpy()*rad;b=attrs.longitude.to_numpy()*rad
    for i,(la,lo) in enumerate(zip(lat*rad,lon*rad)):
        d=6371008.8*2*np.arcsin(np.sqrt(np.clip(np.sin((a-la)/2)**2+np.cos(a)*np.cos(la)*np.sin((b-lo)/2)**2,0,1)))
        ids=np.flatnonzero(d<=radius);counts.append(len(ids));rows.extend([i]*len(ids));cols.extend(ids.tolist())
    return sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(lat),len(attrs))),counts


def select_budget(pred,valid,block_ids,fraction):
    selected=np.zeros(valid.shape,bool)
    for i in range(len(valid)):
        for hour in range(3):
            ids=np.flatnonzero(valid[i,hour])
            if not len(ids):continue
            order=np.lexsort((block_ids[ids],-pred[i,hour,ids]))
            chosen=ids[order[:int(np.ceil(fraction*len(ids)))]];selected[i,hour,chosen]=True
    return selected


def metrics(pred,y,capacity,occupancy,block_ids,candidates):
    valid=np.isfinite(y);error=np.abs(pred-y)
    result=dict(ratio_MAE=float(error[valid].mean()),count_MAE=float((error*capacity)[valid].mean()),events=int(valid.sum()))
    high=(y>=.85)&valid
    for fraction in (.1,.2):
        selected=select_budget(pred,np.broadcast_to(candidates,valid.shape),block_ids,fraction)
        planned=int(selected.sum());selected=selected&valid
        hits=int((selected&high).sum());positives=int(high.sum());n=int(selected.sum())
        tag=f'budget{int(fraction*100)}'
        result.update({tag+'_recall':hits/positives if positives else None,
                       tag+'_precision':hits/n if n else None,
                       tag+'_paid_occupancy_share':float(np.nansum(np.where(selected,occupancy,0))/np.nansum(occupancy)),
                       tag+'_selected':n,tag+'_planned_selected':planned,tag+'_high_pressure':positives})
    return result


def main(a):
    sys.path.insert(0,str(a.source/'exp/common'))
    from exp.common.audit import load_panel
    from exp.common.summarize import selected_jobs, write_json
    from exp.common.evaluate import load_events
    from exp.common.paired import intervals
    from exp.common.controls import grid_groups
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'private_predictions').mkdir()
    raw,raw_dates,attrs=load_panel(a.bundle)
    with np.load(a.inputs/'parking_panel.npz') as z:p={k:z[k] for k in z.files}
    dates=p['dates'];weekday=pd.to_datetime(dates).dayofweek.to_numpy()
    shape=p['occupancy'].shape;days,hours,nodes=shape
    ratio=p['occupancy']/p['capacity'];valid=np.isfinite(ratio)
    train=(dates>='2020-10-01')&(dates<='2020-11-19')
    training_valid=valid&train[:,None,None]
    counts=training_valid.sum(0)
    sums=np.nansum(np.where(training_valid,ratio,np.nan),axis=0)
    means=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    calendar=np.empty(shape,np.float64)
    for day in range(7):
        eligible=training_valid&(weekday[:,None,None]==day)
        sm=np.nansum(np.where(eligible,ratio,np.nan),axis=0);ct=eligible.sum(0)
        calendar[weekday==day]=(sm+5*means)/(ct+5)
    features=[calendar]
    lag_values=[]
    for lag in (7,14):
        past=np.full(shape,np.nan);past[lag:]=ratio[:-lag]
        missing=~np.isfinite(past);filled=np.where(missing,means[None],past)
        lag_values.append(filled);features.extend([filled,missing.astype(float)])
    for hour in (1,2):features.append(np.broadcast_to((np.arange(3)==hour)[None,:,None],shape))
    for day in range(1,7):features.append(np.broadcast_to((weekday==day)[:,None,None],shape))
    base=np.stack(features,axis=-1)
    jobs={}
    wanted={'Shared-LSTM_T14_Identity_full_MAE','Matched-LSTM_T14_Identity_full_MAE',
            'GraphGRU_T14_Identity_full_MAE','GraphGRU_T14_NAICS-2_full_MAE',
            'Fixed-MTGNN_T14_Identity_full_MAE','Fixed-MTGNN_T14_NAICS-2_full_MAE'}
    for path,meta,_ in selected_jobs([a.old/'v1/queue',a.old/'v2/queue',a.extension/'matched_queue']):
        if meta['id'] in wanted:jobs.setdefault(meta['id'],[]).append((meta['seed'],path))
    assert set(jobs)==wanted and all(len(v)==5 for v in jobs.values())
    jobs['NLinear-ridge_T14']=[(-1,a.old/'dependent_20261114/baselines/NLinear_ridge_T14_H7.npz')]
    exemplar=load_events(next(iter(jobs.values()))[0][1])
    def indices(data):
        origins=np.asarray(data['forecast_origin'],dtype='datetime64[D]')
        relative=(origins-np.datetime64(dates[0])).astype(int)
        return relative[:,None]+np.array(LEADS)[None,:]-1
    ex_indices=indices(exemplar);validation=exemplar['dataset_split']=='validation'
    vix=ex_indices[validation,-1];vmask=valid[vix]
    base_fits=[ridge_fit(base[training_valid],ratio[training_valid],lam) for lam in PENALTIES]
    scores=[float(np.abs(ridge_predict(f,base[vix])-ratio[vix])[vmask].mean()) for f in base_fits]
    selected=int(np.argmin(scores));base_fit=base_fits[selected]
    write_json(a.out/'parking_only_fit.json',dict(penalty=PENALTIES[selected],validation_MAE=scores,
                                                mean=base_fit[0].tolist(),scale=base_fit[1].tolist(),weights=base_fit[2].tolist()))
    records=[];paired=[];screening=[];matching=[];models={};fitted=[]
    for period in ('december','january'):
        if period=='december':example=exemplar
        else:
            key=next(iter(jobs));seed=jobs[key][0][0]
            example=load_events(a.old/'consolidated_final_20261114/january/private_predictions'/(key+f'_seed{seed}.npz'))
        ix=indices(example)[example['dataset_split']=='test']
        y=ratio[ix];capacity=p['capacity'][ix];occ=p['occupancy'][ix]
        models[period]=dict(indices=ix,y=y,capacity=capacity,occupancy=occ,baselines={})
        for name,pred in [('Parking calendar',calendar[ix]),('Parking lag7',lag_values[0][ix]),('Parking ridge',ridge_predict(base_fit,base[ix]))]:
            models[period]['baselines'][name]=pred
            for j,lead in enumerate(LEADS):
                records.append(dict(period=period,model=name,seed=-1,radius_m=0,lead=lead,
                                    **metrics(pred[:,j],y[:,j],capacity[:,j],occ[:,j],p['block_id'],counts>0)))
    for radius in (500,250,1000):
        operator,match_counts=spatial_operator(attrs,p['latitude'],p['longitude'],radius)
        matching.append(dict(radius_m=radius,minimum=int(min(match_counts)),median=float(np.median(match_counts)),maximum=int(max(match_counts)),unmatched=int(np.sum(np.array(match_counts)==0))))
        observed=np.full((days,nodes),np.nan)
        offset=int((np.datetime64(raw_dates[0])-np.datetime64(dates[0])).astype(int))
        observed[offset:offset+len(raw)]=(operator@np.nan_to_num(raw,nan=0.).T).T
        log_observed=np.broadcast_to(np.log1p(observed)[:,None,:],shape)
        augmented=np.concatenate([base,log_observed[...,None]],axis=-1)
        fits=[ridge_fit(augmented[training_valid],ratio[training_valid],lam) for lam in PENALTIES]
        loss_cache={period:{} for period in models}
        selection_cache={period:{} for period in models}
        for key,runs in jobs.items():
            for seed,path in sorted(runs):
                data=load_events(path);ix=indices(data)
                total=(operator@data['predicted_visits'][:,np.array(LEADS)-1].reshape(-1,len(attrs)).T).T.reshape(len(ix),len(LEADS),nodes)
                signal=np.broadcast_to(np.log1p(total)[:,:,None,:],(len(ix),len(LEADS),3,nodes))
                vx=np.concatenate([base[ix[validation,-1]],signal[validation,-1,...,None]],axis=-1)
                scores=[float(np.abs(ridge_predict(f,vx)-ratio[vix])[vmask].mean()) for f in fits]
                k=int(np.argmin(scores));fit=fits[k]
                fitted.append(dict(model=key,seed=seed,radius_m=radius,penalty=PENALTIES[k],validation_MAE=scores,
                                   weights=fit[2].tolist(),mean=fit[0].tolist(),scale=fit[1].tolist(),
                                   december_prediction_sha256=sha256(path.read_bytes()).hexdigest()))
                for period in ('december','january'):
                    if period=='january':
                        if key=='NLinear-ridge_T14':name='NLinear_ridge_T14_seed-1.npz'
                        else:name=key+f'_seed{seed}.npz'
                        folder=(a.extension/'matched_evaluation/january' if key.startswith('Matched-LSTM') else a.old/'consolidated_final_20261114/january')
                        jp=folder/'private_predictions'/name;data=load_events(jp);ix=indices(data)
                        total=(operator@data['predicted_visits'][:,np.array(LEADS)-1].reshape(-1,len(attrs)).T).T.reshape(len(ix),len(LEADS),nodes)
                        signal=np.broadcast_to(np.log1p(total)[:,:,None,:],(len(ix),len(LEADS),3,nodes))
                        fitted[-1]['january_prediction_sha256']=sha256(jp.read_bytes()).hexdigest()
                    test=data['dataset_split']=='test';target=ix[test]
                    np.testing.assert_array_equal(target,models[period]['indices'])
                    x=np.concatenate([base[target],signal[test,...,None]],axis=-1)
                    pred=ridge_predict(fit,x).astype(np.float32);assert np.isfinite(pred).all()
                    m=models[period];y=m['y'];capacity=m['capacity'];occ=m['occupancy']
                    name=f'{period}_{key}_seed{seed}_r{radius}'
                    np.savez_compressed(a.out/'private_predictions'/(name+'.npz'),prediction=pred,target_dates=dates[target],block_id=p['block_id'],leads=np.array(LEADS))
                    loss=np.where(np.isfinite(y[:,-1]),np.abs(pred[:,-1]-y[:,-1]),0.).astype(float)
                    loss_cache[period].setdefault(key,[]).append(loss)
                    if radius==500:
                        selection_cache[period].setdefault(key,[]).append(np.stack([
                            select_budget(pred[:,-1],np.broadcast_to(counts>0,y[:,-1].shape),p['block_id'],fraction)
                            for fraction in (.1,.2)]))
                    for j,lead in enumerate(LEADS):
                        records.append(dict(period=period,model=key,seed=seed,radius_m=radius,lead=lead,
                                            **metrics(pred[:,j],y[:,j],capacity[:,j],occ[:,j],p['block_id'],counts>0)))
                print(json.dumps(dict(model=key,seed=seed,radius_m=radius,penalty=PENALTIES[k],status='complete')),flush=True)
        for period,cache in loss_cache.items():
            m=models[period];y=m['y'][:,-1];mask=np.isfinite(y)
            losses={key:np.mean(value,axis=0) for key,value in cache.items()}
            losses['Parking ridge']=np.where(mask,np.abs(m['baselines']['Parking ridge'][:,-1]-y),0.)
            if radius==500:
                selections={key:np.mean(value,axis=0) for key,value in selection_cache[period].items()}
                selections['Parking ridge']=np.stack([select_budget(m['baselines']['Parking ridge'][:,-1],np.broadcast_to(counts>0,mask.shape),p['block_id'],f) for f in (.1,.2)])
            contrasts=[(key,'Parking ridge') for key in cache]
            contrasts += [(f'{name}_T14_NAICS-2_full_MAE',f'{name}_T14_Identity_full_MAE') for name in ('GraphGRU','Fixed-MTGNN')]
            for model,reference in contrasts:
                difference=losses[model]-losses[reference]
                for km in (1.,3.,5.):
                    paired.append(dict(period=period,radius_m=radius,lead=7,model=model,reference=reference,cell_km=km,
                                       **intervals(difference,mask,grid_groups(p['latitude'],p['longitude'],km))))
                    if radius==500:
                        high=(y>=.85)&mask
                        paid=np.where(mask,m['occupancy'][:,-1],0.)
                        for budget,fraction in enumerate((.1,.2)):
                            delta=selections[model][budget]-selections[reference][budget]
                            for metric,weights in [('high_pressure_recall',high),('paid_occupancy_share',paid)]:
                                row=intervals(delta*weights,weights,grid_groups(p['latitude'],p['longitude'],km))
                                row['difference']=row.pop('delta_MAE')
                                screening.append(dict(period=period,model=model,reference=reference,cell_km=km,budget_fraction=fraction,metric=metric,**row))
        pd.DataFrame(records).to_csv(a.out/'metrics_by_seed.csv',index=False)
        pd.DataFrame(paired).to_csv(a.out/'paired_intervals.csv',index=False)
        pd.DataFrame(screening).to_csv(a.out/'screening_intervals.csv',index=False)
        write_json(a.out/'transport_fits.json',fitted)
    table=pd.DataFrame(records);keys=['period','model','radius_m','lead']
    grouped=table.groupby(keys,sort=False)
    summary=grouped[[c for c in table.columns if c not in keys+['seed']]].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary.to_csv(a.out/'metrics_summary.csv',index=False)
    write_json(a.out/'complete.json',dict(status='complete',transport_fits=len(fitted),records=len(records),
                                         matched_POIs=matching,protocol_sha256=sha256(a.protocol.read_bytes()).hexdigest(),
                                         source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),completed_at=time.time()))


if __name__=='__main__':
    p=ArgumentParser()
    for field in ('source','old','extension','inputs','bundle','out','protocol'):p.add_argument('--'+field,type=Path,required=True)
    main(p.parse_args())
