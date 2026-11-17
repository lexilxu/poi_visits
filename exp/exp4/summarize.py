"""Five-seed summaries of matched neighbor interventions in December and January."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
from exp.common.audit import load_panel
from exp.common.paired import intervals
from exp.exp1.extended_baselines import grid_groups
from exp.common.summarize import write_json


def aggregate_table(rows,keys):
    table=pd.concat(rows,ignore_index=True)
    groups=table.groupby(keys,dropna=False,sort=False)
    seeds=groups.seed.agg(lambda x:tuple(sorted(x)))
    if not seeds.apply(lambda x:x==(42,43,44,45,46)).all():
        raise ValueError('Every comparison must contain all five fixed seeds')
    columns=[x for x in table.select_dtypes('number').columns if x not in ['seed','T','H']]
    summary=groups[columns].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns]
    return table,summary.join(seeds.rename('seeds')).reset_index()


def run(args):
    _,_,attrs=load_panel(args.bundle)
    changes=[];transfer=[];checks=[];uncertainty=[];provenance=[]
    for period in ('december','january'):
        for graph in ('NAICS-2','NAICS-4'):
            arrays=[]
            for seed in range(42,47):
                folder=args.root/period/f'{graph}_seed{seed}'
                protocol=json.loads((folder/'neighbor_protocol.json').read_text())
                assert protocol['status']=='complete' and protocol['seed']==seed
                assert protocol['negative_control_max_absolute_error']<=2e-5
                assert protocol['selected_targets']==1935
                checks.append(protocol)
                for name,destination in [('neighbor_interventions',changes),('graph_transfer',transfer)]:
                    frame=pd.read_csv(folder/f'{name}.csv')
                    frame.insert(0,'period',period);destination.append(frame)
                path=folder/'private_counterfactuals.npz'
                # These files were produced by our own verified local worker.
                # Its first version stored date strings in an object array.
                with np.load(path,allow_pickle=False) as z:
                    array={key:z[key] for key in z.files if key!='forecast_origin'}
                with np.load(path,allow_pickle=True) as z:
                    array['forecast_origin']=z['forecast_origin'].astype(str)
                arrays.append(array)
                provenance.append(dict(period=period,graph=graph,seed=seed,
                                       counterfactual_sha256=sha256(path.read_bytes()).hexdigest()))
            reference=arrays[0]
            for array in arrays[1:]:
                for key in ('observed_visits','valid','target_index','degrees','forecast_origin','evidence_strata'):
                    np.testing.assert_array_equal(reference[key],array[key])
            y=reference['observed_visits'];valid=reference['valid']
            target=reference['target_index'];location=attrs.iloc[target]
            groups=grid_groups(location.latitude,location.longitude,3.)
            original_loss=np.mean([np.abs(a['graph_prediction']-y) for a in arrays],axis=0)
            identity_loss=np.mean([np.abs(a['identity_prediction']-y) for a in arrays],axis=0)
            result=intervals(np.where(valid,original_loss-identity_loss,0),valid,groups)
            result['estimate']=result.pop('delta_MAE')
            uncertainty.append(dict(period=period,adjacency=graph,intervention='Graph versus identity',
                  estimand='Change in MAE',**result))
            for name in ('active','correlated','random','recent_active','activity_matched_replacement'):
                loss=np.mean([np.abs(a[name]-y) for a in arrays],axis=0)
                reduction=np.mean([(a['graph_prediction']-a[name])/a['degrees'][None,None,:] for a in arrays],axis=0)
                for estimand,difference in [('Change in MAE',loss-original_loss),
                                           ('Forecast reduction per intervened neighbor',reduction)]:
                    result=intervals(np.where(valid,difference,0),valid,groups)
                    result['estimate']=result.pop('delta_MAE')
                    uncertainty.append(dict(period=period,adjacency=graph,intervention=name,estimand=estimand,**result))
    args.out.mkdir(parents=True,exist_ok=True)
    for name,rows,keys in [('interventions',changes,['period','adjacency','intervention','grouping','stratum']),
                           ('graph_transfer',transfer,['period','adjacency','grouping','stratum'])]:
        by_seed,summary=aggregate_table(rows,keys)
        by_seed.to_csv(args.out/f'{name}_by_seed.csv',index=False)
        summary.to_csv(args.out/f'{name}_summary.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(args.out/'paired_intervals.csv',index=False)
    write_json(args.out/'summary_protocol.json',dict(status='complete',periods=['2020-12','2021-01'],
        seeds=list(range(42,47)),target_POIs=1935,maximum_negative_control_error=max(p['negative_control_max_absolute_error'] for p in checks),
        estimand='Mean of within-fit changes over five fixed seeds, not a prediction ensemble',
        uncertainty='Paired seven-origin blocks and 3-km spatial cells, conditional on these fitted models and this panel',
        activation_denominators='Net changes divide by all zero/positive events; fraction removed/lost divides by originally activated zero/positive events',
        outcome_conditioned_regimes='Descriptive partitions; performance by these regimes is not an independent validation',
        provenance=provenance,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status='complete',cases=len(checks),maximum_negative_control_error=max(p['negative_control_max_absolute_error'] for p in checks))))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
