"""Audit stored parking selections, observation coverage and identification bounds."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import sys,json
import numpy as np
import pandas as pd
from exp.exp8.inputs import features, candidates_for_dates


def report(selected,candidates,y):
    observed=np.isfinite(y)&candidates;high=(y>=.85)&observed
    missing=candidates&~observed
    h=int(high.sum());k=int((selected&high).sum())
    ms=int((selected&missing).sum());mu=int((~selected&missing).sum())
    planned=int(selected.sum());scored=int((selected&observed).sum())
    return dict(candidate_events=int(candidates.sum()),observed_candidate_events=int(observed.sum()),
                planned=planned,scored=scored,observed_hits=k,observed_high_pressure=h,
                missing_selected=ms,missing_unselected=mu,
                selected_coverage=scored/planned if planned else np.nan,
                conditional_recall=k/h if h else np.nan,
                recall_lower=k/(h+mu) if h+mu else np.nan,
                recall_upper=(k+ms)/(h+ms) if h+ms else np.nan,
                observed_hits_per_100_planned=100*k/planned if planned else np.nan,
                precision_lower=k/planned if planned else np.nan,
                precision_upper=(k+ms)/planned if planned else np.nan)


def main(a):
    sys.path[:0]=[str(a.source/'exp/common'),str(a.helpers)]
    from exp.exp8.evaluate import ridge_predict, select_budget
    from exp.common.paired import intervals
    from exp.common.controls import grid_groups
    with np.load(a.inputs/'parking_panel.npz') as z:panel=dict(z)
    data=features(panel,(7,14));dates=data['dates'];rows=[];coverage=[];pairs=[];hashes=[]
    previous=pd.read_csv(a.corrected/'metrics_by_seed.csv')
    fit=json.loads((a.previous/'parking_only_fit.json').read_text())
    fitted=tuple(np.array(fit[k]) for k in ('mean','scale','weights'))
    for period in ['december','january']:
        forecasts=[];target=None
        for file in sorted((a.previous/'private_predictions').glob(period+'_*_r500.npz')):
            name,tail=file.stem[len(period)+1:].rsplit('_seed',1);seed=int(tail.split('_r')[0])
            with np.load(file) as z:
                np.testing.assert_array_equal(z['block_id'],panel['block_id'])
                index=(z['target_dates'].astype('datetime64[D]')-np.datetime64(dates[0])).astype(int)[:,-1]
                if target is None:target=index
                else:np.testing.assert_array_equal(index,target)
                forecasts.append((name,seed,z['prediction'][:,-1]))
            hashes.append(dict(file=file.name,sha256=sha256(file.read_bytes()).hexdigest()))
        assert len(forecasts)==31
        forecasts += [('Parking calendar',-1,data['calendar'][target]),
                      ('Parking lag7',-1,data['lags'][0][target]),
                      ('Parking ridge',-1,ridge_predict(fitted,data['base'][target]))]
        y=data['ratio'][target]
        for rule in ['original','training_calendar']:
            candidates=candidates_for_dates(data,target,rule)
            for day in range(7):
                for hour,clock in enumerate([10,14,18]):
                    included=data['weekday'][target]==day;scope=candidates[included,hour]
                    observed=np.isfinite(y[included,hour])&scope
                    coverage.append(dict(period=period,rule=rule,weekday=day,hour=clock,
                                         dates=int(included.sum()),candidate_events=int(scope.sum()),
                                         observed_events=int(observed.sum())))
            selections={}
            for name,seed,pred in forecasts:
                for budget in [.1,.2]:
                    selected=select_budget(pred,candidates,panel['block_id'],budget)
                    result=report(selected,candidates,y)
                    if rule=='original':
                        old=previous[(previous.period==period)&(previous.model==name)&(previous.seed==seed)&
                                     (previous.lead==7)&(previous.radius_m==(0 if seed==-1 and name.startswith('Parking ') else 500))].iloc[0]
                        tag=f'budget{int(budget*100)}'
                        assert result['planned']==int(old[tag+'_planned_selected'])
                        assert result['scored']==int(old[tag+'_selected'])
                        assert abs(result['conditional_recall']-old[tag+'_recall'])<1e-12
                    rows.append(dict(period=period,rule=rule,model=name,seed=seed,budget_fraction=budget,**result))
                    selections.setdefault((name,budget),[]).append(selected)
            masks={key:np.mean(value,axis=0) for key,value in selections.items()}
            high=(y>=.85)&np.isfinite(y)&candidates
            unknown=candidates&~np.isfinite(y)
            groups=grid_groups(panel['latitude'],panel['longitude'],3.)
            for budget in [.1,.2]:
                for name in sorted({n for n,b in masks if not n.startswith('Parking ')}):
                    delta=masks[(name,budget)]-masks[('Parking ridge',budget)]
                    known=float((delta*high).sum());minus=float((-np.minimum(delta,0)*unknown).sum());plus=float((np.maximum(delta,0)*unknown).sum())
                    interval=intervals(delta*high,high,groups)
                    pairs.append(dict(period=period,rule=rule,model=name,reference='Parking ridge',budget_fraction=budget,
                                      observed_hit_change=known,hit_change_lower=known-minus,hit_change_upper=known+plus,
                                      conditional_recall_change=interval['delta_MAE'],joint_low=interval['joint_low'],joint_high=interval['joint_high']))
    a.out.mkdir(parents=True,exist_ok=False)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'selections_by_seed.csv',index=False)
    keys=['period','rule','model','budget_fraction'];cols=[c for c in frame if c not in keys+['seed']]
    summary=frame.groupby(keys)[cols].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(a.out/'selections_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(a.out/'coverage_by_weekday_hour.csv',index=False)
    pd.DataFrame(pairs).to_csv(a.out/'paired_hit_bounds.csv',index=False)
    audit=dict(status='complete',stored_forecasts_verified=len(hashes),selection_rows=len(rows),
               original_selection_counts_and_recall_reproduced=True,prediction_hashes=hashes,
               source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/exp8/inputs.py')]})
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k not in ['prediction_hashes','source_sha256']}),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for n in ['source','helpers','inputs','previous','corrected','out']:p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
