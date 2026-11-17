"""Check saved forecast metrics, date boundaries and complete experiment axes."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd


def main(a):
    with np.load(a.panel) as z:p=dict(z)
    table=pd.read_csv(a.results/'metrics_by_seed.csv')
    selections=pd.read_csv(a.results/'selections_by_seed.csv')
    audit=json.loads((a.results/'complete.json').read_text())
    assert audit['status']=='complete' and len(table)==774 and len(selections)==1548
    assert audit['ridge_candidate_fits']==148 and audit['new_neural_fits']==0
    ratio=p['occupancy']/p['capacity'];dates=p['dates'].astype('datetime64[D]')
    weekdays=pd.to_datetime(dates).dayofweek.to_numpy()
    training=(dates>=np.datetime64('2020-10-01'))&(dates<=np.datetime64('2020-11-19'))
    counts=np.stack([(np.isfinite(ratio)&(training&(weekdays==d))[:,None,None]).sum(0) for d in range(7)])
    checked=0;max_error=0.;hashes=[]
    for path in sorted((a.results/'private_predictions').glob('*.npz')):
        period,rest=path.stem.split('_',1)
        if '_seed' in rest:
            model,tail=rest.rsplit('_seed',1);seed,tail=tail.split('_r',1);radius,mapping=tail.split('_',1)
            seed=int(seed);radius=int(radius)
        else:model=rest.replace('_',' ');seed=-1;radius=0;mapping='parking_only'
        with np.load(path) as z:
            pred=z['prediction'];targets=z['target_dates'].astype('datetime64[D]')
            np.testing.assert_array_equal(z['block_id'],p['block_id']);np.testing.assert_array_equal(z['leads'],[1,3,7])
        origins=targets-np.array([0,2,6],dtype='timedelta64[D]')
        np.testing.assert_array_equal(origins,np.broadcast_to(origins[:,:1],origins.shape))
        assert origins.min()>=np.datetime64('2020-12-11')
        for lag in [14,21]:assert np.all(targets-np.timedelta64(lag,'D')<=origins-np.timedelta64(8,'D'))
        ix=(targets-dates[0]).astype(int);y=ratio[ix]
        scope=counts[weekdays[ix]]>=3
        holidays=np.isin(targets,np.array(['2020-12-25','2021-01-01','2021-01-18'],dtype='datetime64[D]'))
        scope[(weekdays[ix]==6)|holidays]=False
        group=table[(table.period==period)&(table.model==model)&(table.seed==seed)&(table.radius_m==radius)&(table.mapping==mapping)]
        assert len(group)==3
        for j,lead in enumerate([1,3,7]):
            valid=np.isfinite(y[:,j])&scope[:,j];mae=float(np.abs(pred[:,j]-y[:,j])[valid].mean())
            row=group[group.lead==lead].iloc[0];err=abs(mae-row.ratio_MAE);max_error=max(max_error,err)
            assert err<1e-12 and int(valid.sum())==row.events
            for budget in [.1,.2]:
                chosen=np.zeros_like(valid)
                for day in range(len(y)):
                    for hour in range(3):
                        ids=np.flatnonzero(scope[day,j,hour]);k=int(np.ceil(len(ids)*budget))
                        order=sorted(ids,key=lambda b:(-pred[day,j,hour,b],p['block_id'][b]))
                        chosen[day,hour,order[:k]]=True
                s=selections[(selections.period==period)&(selections.model==model)&(selections.seed==seed)&(selections.radius_m==radius)&(selections.mapping==mapping)&(selections.lead==lead)&(selections.budget_fraction==budget)]
                assert len(s)==1;s=s.iloc[0]
                assert chosen.sum()==s.planned and (chosen&valid).sum()==s.scored
                assert (chosen&valid&(y[:,j]>=.85)).sum()==s.observed_hits
            checked+=1
        hashes.append(dict(file=path.name,sha256=sha256(path.read_bytes()).hexdigest()))
    assert checked==774 and len(hashes)==258
    report=dict(status='verified',prediction_files=len(hashes),metric_rows=checked,selection_rows=len(selections),
                max_metric_difference=max_error,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),predictions=hashes)
    dest=a.results/'verification.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'}))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--results',type=Path,required=True)
    main(parser.parse_args())
