"""Evaluate dated parking inputs and historical-forecast mapping on AutoDL4."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json, sys, time
import numpy as np
import pandas as pd
from exp.exp8.inputs import features, candidates_for_dates
from exp.exp8.audit_coverage import report

LEADS=np.array([1,3,7])
PENALTIES=(.0001,.01,1.,100.)


def main(a):
    sys.path[:0]=[str(a.source/'exp/common'),str(a.helpers)]
    from exp.common.audit import load_panel
    from exp.common.summarize import selected_jobs, write_json
    from exp.common.evaluate import load_events
    from exp.exp8.evaluate import ridge_fit, ridge_predict, spatial_operator, select_budget
    from exp.common.paired import intervals
    from exp.common.controls import grid_groups
    raw,raw_dates,attrs=load_panel(a.bundle)
    with np.load(a.inputs/'parking_panel.npz') as z:p=dict(z)
    f=features(p,(14,21));dates=f['dates'];base=f['base'];ratio=f['ratio']
    train=f['training'];shape=ratio.shape;days,hours,nodes=shape
    train_days=np.flatnonzero((dates>='2020-10-01')&(dates<='2020-11-19'))
    assert len(train_days)==50
    jobs={};wanted={'Shared-LSTM_T14_Identity_full_MAE','Matched-LSTM_T14_Identity_full_MAE',
        'GraphGRU_T14_Identity_full_MAE','GraphGRU_T14_NAICS-2_full_MAE',
        'Fixed-MTGNN_T14_Identity_full_MAE','Fixed-MTGNN_T14_NAICS-2_full_MAE'}
    for path,meta,_ in selected_jobs([a.old/'v1/queue',a.old/'v2/queue',a.extension/'matched_queue']):
        if meta['id'] in wanted:jobs.setdefault(meta['id'],[]).append((meta['seed'],path))
    assert set(jobs)==wanted and all(len(v)==5 for v in jobs.values())
    jobs['NLinear-ridge_T14']=[(-1,a.old/'dependent_20261114/baselines/NLinear_ridge_T14_H7.npz')]
    heldout_names={'GraphGRU_T14_Identity_full_MAE':'GraphGRU',
                   'Matched-LSTM_T14_Identity_full_MAE':'Matched-LSTM','NLinear-ridge_T14':'NLinear-ridge'}
    historical=json.loads((a.historical/'complete.json').read_text())
    assert historical['status']=='complete' and historical['validation_end']=='2020-09-02'
    def indices(data):
        origins=data['forecast_origin'].astype('datetime64[D]')
        relative=(origins-np.datetime64(dates[0])).astype(int)
        return relative[:,None]+LEADS[None,:]-1
    def test_mask(data):
        return (data['dataset_split']=='test')&(data['forecast_origin'].astype('datetime64[D]')>=np.datetime64('2020-12-11'))
    def january_path(key,seed):
        name='NLinear_ridge_T14_seed-1.npz' if key=='NLinear-ridge_T14' else key+f'_seed{seed}.npz'
        folder=a.extension/'matched_evaluation/january' if key.startswith('Matched-LSTM') else a.old/'consolidated_final_20261114/january'
        return folder/'private_predictions'/name
    first=next(iter(jobs));seed,path=jobs[first][0];example=load_events(path)
    validation=example['dataset_split']=='validation';vix=indices(example)[validation,-1]
    validation_end=str(dates[vix].astype('datetime64[D]').max())
    assert validation_end=='2020-12-03'
    vscope=candidates_for_dates(f,vix,'training_calendar');vmask=np.isfinite(ratio[vix])&vscope
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'private_predictions').mkdir()
    def fit_grid(x):return [ridge_fit(x[train],ratio[train],lam) for lam in PENALTIES]
    def choose(fits,x):
        scores=[float(np.abs(ridge_predict(fit,x)-ratio[vix])[vmask].mean()) for fit in fits]
        k=int(np.argmin(scores));return fits[k],k,scores
    fits=fit_grid(base);base_fit,k,scores=choose(fits,base[vix])
    fitted=[dict(model='Parking ridge',mapping='parking_only',seed=-1,radius_m=0,penalty=PENALTIES[k],
                 validation_MAE=scores,mean=base_fit[0].tolist(),scale=base_fit[1].tolist(),weights=base_fit[2].tolist())]
    models={};records=[];selections=[];paired=[];screening=[];provenance=[];availability=[]
    def summarize(pred,m,meta):
        for j,lead in enumerate(LEADS):
            y=m['y'][:,j];scope=m['scope'][:,j];valid=np.isfinite(y)&scope
            err=np.abs(pred[:,j]-y)
            row=dict(**meta,lead=int(lead),ratio_MAE=float(err[valid].mean()),
                count_MAE=float((err*m['capacity'][:,j])[valid].mean()),events=int(valid.sum()),origins=len(y))
            for budget in (.1,.2):
                selected=select_budget(pred[:,j],scope,p['block_id'],budget)
                selections.append(dict(**meta,lead=int(lead),budget_fraction=budget,**report(selected,scope,y)))
            records.append(row)
    for period,data in [('december',example),('january',load_events(january_path(first,seed)))]:
        test=test_mask(data);ix=indices(data)[test];origins=data['forecast_origin'][test].astype('datetime64[D]')
        scope=np.stack([candidates_for_dates(f,ix[:,j],'training_calendar') for j in range(3)],axis=1)
        m=dict(indices=ix,y=ratio[ix],capacity=p['capacity'][ix],scope=scope,origins=origins,baselines={})
        models[period]=m
        for lag in (14,21):
            source=dates[ix].astype('datetime64[D]')-np.timedelta64(lag,'D')
            slack=(origins[:,None]-np.timedelta64(8,'D')-source).astype(int)
            assert slack.min()>=0
            availability.append(dict(period=period,lag=lag,minimum_slack_days=int(slack.min()),
                first_origin=str(origins.min()),last_origin=str(origins.max()),origins=len(origins)))
        for name,pred in [('Parking calendar',f['calendar'][ix]),('Parking lag14',f['lags'][0][ix]),('Parking ridge',ridge_predict(base_fit,base[ix]))]:
            m['baselines'][name]=pred
            summarize(pred,m,dict(period=period,model=name,mapping='parking_only',seed=-1,radius_m=0))
            np.savez_compressed(a.out/'private_predictions'/f'{period}_{name.replace(" ","_")}.npz',
                prediction=pred,target_dates=dates[ix],block_id=p['block_id'],leads=LEADS)
    for radius in (500,250,1000):
        operator,counts=spatial_operator(attrs,p['latitude'],p['longitude'],radius)
        observed=np.full((days,nodes),np.nan)
        offset=int((np.datetime64(raw_dates[0])-np.datetime64(dates[0])).astype(int))
        observed[offset:offset+len(raw)]=(operator@np.nan_to_num(raw,nan=0.).T).T
        def augment(signal):return np.concatenate([base,np.broadcast_to(np.log1p(signal)[:,None,:,None],(*shape,1))],axis=-1)
        observed_fits=fit_grid(augment(observed))
        losses={period:{} for period in models};selected_masks={period:{} for period in models}
        for key,runs in jobs.items():
            for seed,path in sorted(runs):
                all_data={'december':load_events(path),'january':load_events(january_path(key,seed))}
                signals={}
                for period,data in all_data.items():
                    np.testing.assert_array_equal(data['poi_id'],attrs.placekey.to_numpy(dtype=str))
                    total=(operator@data['predicted_visits'][:,LEADS-1].reshape(-1,len(attrs)).T).T.reshape(len(data['forecast_origin']),3,nodes)
                    signals[period]=np.broadcast_to(np.log1p(total)[:,:,None,:],(len(total),3,3,nodes))
                np.testing.assert_array_equal(all_data['december']['dataset_split']=='validation',validation)
                vx=np.concatenate([base[vix],signals['december'][validation,-1,...,None]],axis=-1)
                mapping_fits={'observed_visits':observed_fits}
                if key in heldout_names:
                    hp=a.historical/f'{heldout_names[key]}_seed{seed}.npz'
                    with np.load(hp) as z:
                        np.testing.assert_array_equal(z['poi_id'],attrs.placekey.to_numpy(dtype=str))
                        np.testing.assert_array_equal(z['target_dates'],dates[train_days])
                        assert z['forecast_origins'].astype('datetime64[D]').min()>np.datetime64('2020-09-02')
                        historical_signal=np.full((days,nodes),np.nan)
                        historical_signal[train_days]=(operator@z['predictions'].T).T
                    mapping_fits['historical_forecasts']=fit_grid(augment(historical_signal))
                    provenance.append(dict(kind='historical_forecast',model=key,seed=seed,radius_m=radius,sha256=sha256(hp.read_bytes()).hexdigest()))
                for mapping,fit_options in mapping_fits.items():
                    fit,k,scores=choose(fit_options,vx)
                    fitted.append(dict(model=key,mapping=mapping,seed=seed,radius_m=radius,penalty=PENALTIES[k],
                        validation_MAE=scores,mean=fit[0].tolist(),scale=fit[1].tolist(),weights=fit[2].tolist()))
                    for period,data in all_data.items():
                        test=test_mask(data);ix=indices(data)[test];m=models[period]
                        np.testing.assert_array_equal(ix,m['indices'])
                        x=np.concatenate([base[ix],signals[period][test,...,None]],axis=-1)
                        pred=ridge_predict(fit,x).astype(np.float32);assert np.isfinite(pred).all()
                        meta=dict(period=period,model=key,mapping=mapping,seed=seed,radius_m=radius)
                        summarize(pred,m,meta)
                        name=f'{period}_{key}_seed{seed}_r{radius}_{mapping}.npz'
                        np.savez_compressed(a.out/'private_predictions'/name,prediction=pred,
                            target_dates=dates[ix],block_id=p['block_id'],leads=LEADS)
                        valid=np.isfinite(m['y'][:,-1])&m['scope'][:,-1]
                        losses[period].setdefault((key,mapping),[]).append(np.where(valid,np.abs(pred[:,-1]-m['y'][:,-1]),0.))
                        if radius==500:
                            selected_masks[period].setdefault((key,mapping),[]).append(np.stack([
                                select_budget(pred[:,-1],m['scope'][:,-1],p['block_id'],b) for b in (.1,.2)]))
                    print(json.dumps(dict(model=key,seed=seed,radius_m=radius,mapping=mapping,status='complete')),flush=True)
                if radius==500:
                    for period,fp in [('december',path),('january',january_path(key,seed))]:
                        provenance.append(dict(kind='final_forecast',model=key,seed=seed,period=period,sha256=sha256(fp.read_bytes()).hexdigest()))
        for period,m in models.items():
            y=m['y'][:,-1];scope=m['scope'][:,-1];valid=np.isfinite(y)&scope
            cache={k:np.mean(v,axis=0) for k,v in losses[period].items()}
            baseline=('Parking ridge','parking_only')
            cache[baseline]=np.where(valid,np.abs(m['baselines']['Parking ridge'][:,-1]-y),0.)
            contrasts=[(k,baseline) for k in losses[period]]
            contrasts += [((k,'historical_forecasts'),(k,'observed_visits')) for k in heldout_names]
            contrasts += [((f'{k}_T14_NAICS-2_full_MAE','observed_visits'),(f'{k}_T14_Identity_full_MAE','observed_visits')) for k in ('GraphGRU','Fixed-MTGNN')]
            if radius==500:
                sel={k:np.mean(v,axis=0) for k,v in selected_masks[period].items()}
                sel[baseline]=np.stack([select_budget(m['baselines']['Parking ridge'][:,-1],scope,p['block_id'],b) for b in (.1,.2)])
            for model,reference in contrasts:
                meta=dict(period=period,radius_m=radius,lead=7,model=model[0],mapping=model[1],reference=reference[0],reference_mapping=reference[1])
                for km in (1.,3.,5.):
                    groups=grid_groups(p['latitude'],p['longitude'],km)
                    paired.append(dict(**meta,cell_km=km,**intervals(cache[model]-cache[reference],valid,groups)))
                    if radius==500:
                        high=(y>=.85)&valid;unknown=scope&~valid
                        for j,budget in enumerate((.1,.2)):
                            delta=sel[model][j]-sel[reference][j]
                            known=float((delta*high).sum());lower=known+float((np.minimum(delta,0)*unknown).sum());upper=known+float((np.maximum(delta,0)*unknown).sum())
                            ci=intervals(delta*high,high,groups);ci['conditional_recall_change']=ci.pop('delta_MAE')
                            screening.append(dict(**meta,cell_km=km,budget_fraction=budget,observed_hit_change=known,
                                hit_change_lower=lower,hit_change_upper=upper,**ci))
        pd.DataFrame(records).to_csv(a.out/'metrics_by_seed.csv',index=False)
        pd.DataFrame(selections).to_csv(a.out/'selections_by_seed.csv',index=False)
        pd.DataFrame(paired).to_csv(a.out/'paired_intervals.csv',index=False)
        pd.DataFrame(screening).to_csv(a.out/'screening_intervals.csv',index=False)
        write_json(a.out/'mapping_fits.json',fitted)
    for name,rows,keys in [('metrics',records,['period','model','mapping','radius_m','lead']),
                           ('selections',selections,['period','model','mapping','radius_m','lead','budget_fraction'])]:
        frame=pd.DataFrame(rows);columns=[c for c in frame if c not in keys+['seed']]
        summary=frame.groupby(keys)[columns].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
        summary.reset_index().to_csv(a.out/(name+'_summary.csv'),index=False)
    write_json(a.out/'complete.json',dict(status='complete',new_neural_fits=0,mapping_fit_selections=len(fitted),
        ridge_candidate_fits=4+3*4+3*11*4,records=len(records),selection_records=len(selections),
        scenario='Parking publication delay; supplied POI forecasts at each origin',availability=availability,
        candidate_rule='training_calendar',validation_end=validation_end,first_test_origin='2020-12-11',
        input_forecasts=provenance,source_sha256={q.name:sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/exp8/inputs.py'),(Path(__file__).resolve().parents[2] / 'exp/exp8/audit_coverage.py'),a.protocol]},
        completed_at=time.time()))
    print(json.dumps(dict(status='complete',mapping_fit_selections=len(fitted),records=len(records))),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for n in ('source','helpers','old','extension','inputs','bundle','historical','out','protocol'):
        p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
