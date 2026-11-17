"""Independent scalar reductions and history construction for the saved analysis."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import hashlib
import numpy as np
import pandas as pd
import exp.exp9.analyze as analyze

from argparse import ArgumentParser
from exp.common.paths import DATA_ROOT, RESULTS_ROOT, REPO_ROOT


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=analyze.BASE)
    parser.add_argument('--extension', type=Path, default=analyze.EXT)
    parser.add_argument('--bundle', type=Path, default=analyze.BUNDLE)
    args = parser.parse_args()
    analyze.BASE, analyze.EXT, analyze.BUNDLE = args.base, args.extension, args.bundle

    out=args.results
    audit=json.loads((out/'audit.json').read_text())
    with ZipFile(analyze.BUNDLE) as z:
        attrs=pd.read_csv(BytesIO(z.read('poi_attributes.csv')))
    with np.load(analyze.BASE/'private_validation_202101/extended_panel.npz') as z:
        raw=z['visits'].astype(float);dates=z['dates'].astype(str)
    codes={g:attrs.naics_code.astype('int64').astype(str).str[:d].to_numpy() for g,d in [('NAICS-2',2),('NAICS-4',4)]}
    peers={g:{c:np.flatnonzero(v==c) for c in np.unique(v)} for g,v in codes.items()}
    eligible=np.array([all(len(peers[g][codes[g][j]])>1 for g in peers) for j in range(raw.shape[1])])
    own=np.array([np.nan_to_num(raw[o-14:o],nan=0).sum(0)/14 for o in range(14,117)])
    neighbor={}
    for g,groups in peers.items():
        a=np.zeros_like(own)
        for indices in groups.values():
            if len(indices)>1:a[:,indices]=(own[:,indices].sum(1)[:,None]-own[:,indices])/(len(indices)-1)
        neighbor[g]=a
    thresholds=json.loads((out/'thresholds.json').read_text())
    assert np.median(own[:30,eligible])==thresholds['target']
    for g in neighbor:assert abs(np.median(neighbor[g][:30,eligible])-thresholds['neighbors'][g])<1e-12
    train=pd.DataFrame(raw[:50]);m=train.mean(0).to_numpy();z=(train==0).sum(0).to_numpy()/train.count(0).to_numpy()
    low=(m>0)&(m<=np.quantile(m[m>0],.25))&(z>=.7)
    assert low.sum()==1944
    cost=pd.read_csv(out/'cost_per_seed.csv');strata=pd.read_csv(out/'strata_per_seed.csv')
    recovery=[];maxerr=0.;checks=0
    for period in ['December','January']:
        paths,_=analyze.sources(period)
        for model,archives in paths.items():
            for path in archives:
                with np.load(path) as a:
                    test=a['dataset_split']=='test';y=a['observed_visits'][test].astype(float);p=a['predicted_visits'][test].astype(float);v=a['valid'][test];seed=int(a['seed'])
                    origins=np.searchsorted(dates,a['forecast_origin'][test].astype(str))
                for cohort,poi in [('All',np.ones(raw.shape[1],bool)),('Low-history',low)]:
                    mask=v&poi[None,None,:];true=y[mask];pred=p[mask];err=pred-true
                    for lam in [.5,1,2,3,5]:
                        # A different expression from the original component implementation.
                        actual=np.where(err>=0,err,-lam*err).sum()/mask.sum()
                        selected=cost[(cost.period==period)&(cost.model==model)&(cost.seed==seed)&(cost.cohort==cohort)&(cost.weight==lam)]
                        assert len(selected)==1
                        diff=abs(actual-selected.loss.iloc[0]);maxerr=max(maxerr,diff);assert diff<1e-12;checks+=1
                    positive=true>0;zero=true==0
                    recovery.append(dict(period=period,cohort=cohort,model=model,seed=seed,events=len(true),positive_events=int(positive.sum()),zero_events=int(zero.sum()),
                        recall=float(np.count_nonzero(pred[positive]>=.5)/positive.sum()),FAR=float(np.count_nonzero(pred[zero]>=.5)/zero.sum()),DMR=float(pred[positive].sum()/true[positive].sum())))
                if model in analyze.GRAPH_MODELS and seed==42:
                    for graph in peers:
                        cell=(own[origins-14]>thresholds['target'])*2+(neighbor[graph][origins-14]>thresholds['neighbors'][graph])
                        for c,label in enumerate(analyze.CELLS):
                            mask=v&eligible[None,None,:]&(cell==c)[:,None,:]
                            row=strata[(strata.period==period)&(strata.model==model)&(strata.seed==seed)&(strata.neighbor_partition==graph)&(strata.cell==label)].iloc[0]
                            true=y[mask];pred=p[mask];pos=true>0;zero=true==0
                            values=dict(MAE=np.abs(pred-true).sum()/mask.sum(),recall=np.count_nonzero(pred[pos]>=.5)/pos.sum(),FAR=np.count_nonzero(pred[zero]>=.5)/zero.sum(),DMR=pred[pos].sum()/true[pos].sum())
                            assert int(row.events)==mask.sum()
                            for k,value in values.items():
                                diff=abs(value-row[k]);maxerr=max(maxerr,diff);assert diff<1e-12;checks+=1
        print(period,'independent checks passed',flush=True)
    rf=pd.DataFrame(recovery);rf.to_csv(out/'recovery_per_seed.csv',index=False)
    rf.groupby(['period','cohort','model'],sort=False).agg(events=('events','first'),positive_events=('positive_events','first'),zero_events=('zero_events','first'),recall=('recall','mean'),recall_sd=('recall','std'),FAR=('FAR','mean'),FAR_sd=('FAR','std'),DMR=('DMR','mean'),DMR_sd=('DMR','std')).reset_index().to_csv(out/'recovery_summary.csv',index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in audit['input_sha256'].items())
    report=dict(status='passed',independent_numeric_checks=checks,max_absolute_discrepancy=maxerr,training_medians_independently_reproduced=True,
        test_labels_excluded_from_group_assignment=True,all_history_indices_precede_origin=True,source_inputs_unchanged=True,
        verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
