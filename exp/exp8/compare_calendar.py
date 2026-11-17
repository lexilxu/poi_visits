"""Compare the prespecified calendar control using preserved forecasts."""
from argparse import ArgumentParser
from pathlib import Path
import sys,json
from hashlib import sha256
import numpy as np
import pandas as pd
from exp.exp8.inputs import features, candidates_for_dates


def main(a):
    sys.path[:0]=[str(a.source/'exp/common'),str(a.helpers)]
    from exp.common.paired import intervals
    from exp.common.controls import grid_groups
    from exp.exp8.evaluate import select_budget
    with np.load(a.panel) as z:p=dict(z)
    data=features(p,(14,21));rows=[];hits=[]
    for period in ['december','january']:
        with np.load(a.results/'private_predictions'/f'{period}_Parking_calendar.npz') as z:
            calendar=z['prediction'][:,-1];ix=(z['target_dates'].astype('datetime64[D]')-np.datetime64(data['dates'][0])).astype(int)[:,-1]
        y=data['ratio'][ix];scope=candidates_for_dates(data,ix,'training_calendar');valid=np.isfinite(y)&scope
        closs=np.where(valid,np.abs(calendar-y),0.);high=(y>=.85)&valid;unknown=scope&~valid
        csel=np.stack([select_budget(calendar,scope,p['block_id'],b) for b in [.1,.2]])
        loss={};selections={}
        paths=list((a.results/'private_predictions').glob(period+'_*_r*.npz'))
        paths.append(a.results/'private_predictions'/f'{period}_Parking_ridge.npz')
        for path in paths:
            with np.load(path) as z:pred=z['prediction'][:,-1]
            if '_seed' in path.stem:
                model,tail=path.stem[len(period)+1:].rsplit('_seed',1);seed,tail=tail.split('_r',1);radius,mapping=tail.split('_',1);radius=int(radius)
            else:model='Parking ridge';mapping='parking_only';radius=0
            key=(model,mapping,radius)
            loss.setdefault(key,[]).append(np.where(valid,np.abs(pred-y),0.))
            if radius in [0,500]:selections.setdefault(key,[]).append(np.stack([select_budget(pred,scope,p['block_id'],b) for b in [.1,.2]]))
        for key,values in loss.items():
            model,mapping,radius=key;meta=dict(period=period,model=model,mapping=mapping,radius_m=radius,lead=7,reference='Parking calendar')
            for km in [1.,3.,5.]:
                groups=grid_groups(p['latitude'],p['longitude'],km)
                rows.append(dict(**meta,cell_km=km,**intervals(np.mean(values,axis=0)-closs,valid,groups)))
                if key in selections:
                    difference=np.mean(selections[key],axis=0)-csel
                    for j,budget in enumerate([.1,.2]):
                        delta=difference[j];known=float((delta*high).sum())
                        ci=intervals(delta*high,high,groups);ci['conditional_recall_change']=ci.pop('delta_MAE')
                        hits.append(dict(**meta,cell_km=km,budget_fraction=budget,observed_hit_change=known,
                            hit_change_lower=known+float((np.minimum(delta,0)*unknown).sum()),
                            hit_change_upper=known+float((np.maximum(delta,0)*unknown).sum()),**ci))
    a.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(a.out/'paired_intervals.csv',index=False)
    pd.DataFrame(hits).to_csv(a.out/'screening_intervals.csv',index=False)
    (a.out/'complete.json').write_text(json.dumps(dict(status='complete',paired_rows=len(rows),screening_rows=len(hits),
        new_fits=0,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    p=ArgumentParser()
    for n in ['source','helpers','panel','results','out']:p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
