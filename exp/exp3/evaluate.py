"""Original descriptive transfer strata using corrected frozen predictions."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,socket,sys
import numpy as np
import pandas as pd


def main(args):
    sys.path[:0]=[str(args.source),str(args.source/'exp/common'),str(args.source/'exp/common')]
    from exp.common.original_protocol import load_panel, make_windows, group_codes, group_average
    from exp.common.audit import EXPECTED_BUNDLE_SHA256
    panel=load_panel(args.bundle,EXPECTED_BUNDLE_SHA256);w=make_windows(panel.raw,28,14)
    low=(w.history_mean>0)&(w.history_mean<=np.quantile(w.history_mean[w.history_mean>0],.25))&(w.zero_ratio>=.7)
    truth=w.y[w.test_start:];y=truth[:,:,low];positive=y>0
    history=panel.filled[w.history_idx[w.test_start:]].mean(1)
    groups=group_codes(panel.attrs)
    np.testing.assert_array_equal(groups['NAICS-4'],groups['Top category'])
    rows=[];cells=[];maps=[];metadata=[];sources=[]
    def forecast(graph,seed):
        run=args.root/'runs'/f"original_masked_T28_H14_{graph.replace(' ','_')}_history_seed{seed}"
        record=json.loads((run/'complete.json').read_text());file=run/'predictions.npz'
        assert record['status']=='complete' and sha256(file.read_bytes()).hexdigest()==record['predictions_sha256']
        assert record['config']['adjacency']==graph and record['config']['seed']==seed
        with np.load(file,allow_pickle=False) as z:
            selected=z['dataset_split']=='test'
            np.testing.assert_array_equal(z['observed_visits'][selected],truth)
            np.testing.assert_array_equal(z['poi_id'],panel.attrs.placekey.to_numpy(dtype=str))
            np.testing.assert_array_equal(z['forecast_origin'][selected],np.asarray(panel.dates)[w.origins[w.test_start:]])
            assert z['valid'][selected].all()
            p=z['predicted_visits'][selected][:,:,low]
        sources.append(dict(path=str(file),sha256=record['predictions_sha256']))
        return p
    for seed in args.seeds:
        identity=forecast('Identity',seed)
        for graph in ['NAICS-2','NAICS-4']:
            pred=forecast(graph,seed);labels=groups[graph];size=np.bincount(labels)[labels]
            neighbor_mean=(group_average(history,labels)*size-history)/np.maximum(size-1,1)
            target=np.broadcast_to(history[:,None,low],y.shape)
            neighbor=np.broadcast_to(neighbor_mean[:,None,low],y.shape)
            lift=pred-identity;gain=np.abs(y-identity)-np.abs(y-pred)
            target_cut=float(np.median(target));neighbor_cut=float(np.median(neighbor))
            low_t=target<target_cut;high_n=neighbor>=neighbor_cut
            false=(~positive)&(pred>=.5);attempted=low_t&high_n&(lift>=.1)
            regimes={'Spurious transfer':attempted&false,'Substitutive transfer':attempted,
                'Confirmatory transfer':~low_t&high_n&(lift>=.1),'Redundant self-history':~low_t&(lift<.1),
                'Conservative zero':~positive&(pred<.5),'Other':np.ones_like(positive)}
            remaining=np.ones_like(positive)
            for label,subset in regimes.items():
                keep=subset&remaining;remaining&=~keep
                cols=dict(graph_lift=lift,target_evidence=target,neighbor_evidence=neighbor,
                    mean_mae_gain=gain,positive_event_share=positive,false_activation_share=false)
                rows.append(dict(seed=seed,adjacency=graph,regime=label,n_events=int(keep.sum()),
                    **{name:float(value[keep].mean()) if keep.any() else None for name,value in cols.items()}))
            assert not remaining.any()
            for low_target in [True,False]:
                for high_neighbor in [False,True]:
                    keep=(low_t==low_target)&(high_n==high_neighbor)
                    cells.append(dict(seed=seed,adjacency=graph,target='Low' if low_target else 'High',
                        neighbor='High' if high_neighbor else 'Low',events=int(keep.sum()),
                        mean_mae_gain=float(gain[keep].mean()) if keep.any() else None))
            for j,poi in enumerate(np.flatnonzero(low)):
                maps.append(dict(seed=seed,adjacency=graph,placekey=panel.attrs.placekey.iloc[poi],
                    latitude=float(panel.attrs.latitude.iloc[poi]),longitude=float(panel.attrs.longitude.iloc[poi]),
                    mean_mae_gain=float(gain[:,:,j].mean()),graph_lift=float(lift[:,:,j].mean()),
                    positive_share=float(positive[:,:,j].mean()),attempted_share=float(attempted[:,:,j].mean()),
                    spurious_share=float((attempted&false)[:,:,j].mean())))
            metadata.append(dict(seed=seed,adjacency=graph,low_history_pois=int(low.sum()),events=int(y.size),
                target_median=target_cut,neighbor_median=neighbor_cut,
                mean_gain_positive=float(gain[positive].mean()),mean_gain_zero=float(gain[~positive].mean())))
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(args.out/'transfer_regimes_by_seed.csv',index=False)
    pd.DataFrame(cells).to_csv(args.out/'evidence_cells_by_seed.csv',index=False)
    pd.DataFrame(maps).to_csv(args.out/'spatial_transfer_by_seed.csv',index=False)
    report=dict(status='complete' if args.seeds==[42,43,44,45,46] else 'partial_seed_export',seeds=args.seeds,
        metadata=metadata,sources=sources,alias={'Top category':'NAICS-4'},
        thresholds='Original descriptive medians of test-input target and neighbor history. Lift threshold 0.1 and activation threshold 0.5.',
        estimand='Prediction contrasts between separately fitted graph configurations, with disjoint regime assignment in the original priority order.',
        code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],seeds=args.seeds,regime_rows=len(rows))))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for field in ['source','root','bundle','out']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46]);main(p.parse_args())
