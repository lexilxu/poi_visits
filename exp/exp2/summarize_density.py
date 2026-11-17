"""Verify and reduce all corrected density fits, reusing the full-cohort runs."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,socket,sys
import numpy as np
import pandas as pd
from exp.exp1.summarize import correlation


def main(args):
    sys.path[:0]=[str(args.source),str(args.source/'exp/common'),
        str(args.source/'exp/common'),str(args.source/'exp/common')]
    from exp.common.original_protocol import load_panel, make_windows
    from exp.common.audit import EXPECTED_BUNDLE_SHA256
    from exp.exp2.density_protocol import selected_indices
    panel=load_panel(args.bundle,EXPECTED_BUNDLE_SHA256)
    observed=np.isfinite(panel.raw[:57]);ratio=(panel.raw[:57]>0).sum(0)/observed.sum(0)
    rows=[];sources=[]
    subsets=['Full']+[f'{group}{pct}' for group in ['Dense','Recent-dense','Persist-dense'] for pct in [10,25,50,75]]
    for label in subsets:
        if label=='Full':ids=np.arange(len(panel.attrs));selection=None
        else:ids,selection=selected_indices(panel.raw,label)
        w=make_windows(panel.raw[:,ids],28,14);truth=w.y[w.test_start:]
        ha=np.broadcast_to(w.history_mean,truth.shape);ha_mae=float(np.abs(truth-ha).mean())
        for seed in range(42,47):
            run=(args.root/'runs'/f'original_masked_T28_H14_Identity_history_seed{seed}' if label=='Full'
                else args.root/'density/runs'/f'{label}_seed{seed}')
            record=json.loads((run/'complete.json').read_text());file=run/'predictions.npz'
            assert record['status']=='complete' and record['config']['seed']==seed
            assert record.get('density')==selection
            assert sha256(file.read_bytes()).hexdigest()==record['predictions_sha256']
            with np.load(file,allow_pickle=False) as saved:
                test=saved['dataset_split']=='test'
                np.testing.assert_array_equal(saved['poi_id'],panel.attrs.placekey.iloc[ids].to_numpy(dtype=str))
                np.testing.assert_array_equal(saved['observed_visits'][test],truth)
                np.testing.assert_array_equal(saved['forecast_origin'][test],np.asarray(panel.dates)[w.origins[w.test_start:]])
                assert saved['valid'][test].all()
                prediction=saved['predicted_visits'][test]
            error=prediction.astype(float)-truth;mae=float(np.abs(error).mean())
            assert abs(mae-record['test_MAE'])<1e-5
            rows.append(dict(subset=label,seed=seed,n_pois=len(ids),nonzero_ratio=float(ratio[ids].mean()),
                mae=mae,rmse=float(np.sqrt(np.square(error).mean())),
                rse=float(np.linalg.norm(error.ravel())/np.linalg.norm((truth-truth.mean()).ravel())),
                corr=correlation(truth,prediction),ha_mae=ha_mae,skill_percent=100*(1-mae/ha_mae)))
            sources.append(dict(subset=label,seed=seed,path=str(file),sha256=record['predictions_sha256']))
    args.out.mkdir(parents=True,exist_ok=False)
    table=pd.DataFrame(rows)
    table.to_csv(args.out/'density_sensitivity_by_seed.csv',index=False)
    table[table.seed==42].to_csv(args.out/'density_sensitivity_primary.csv',index=False)
    table.groupby('subset',sort=False)[['mae','rmse','rse','corr','skill_percent']].agg(['mean','std']).to_csv(args.out/'density_sensitivity_summary.csv')
    (args.out/'complete.json').write_text(json.dumps(dict(status='complete',fits=65,subsets=13,seeds=list(range(42,47)),
        reused_full_cohort_fits=5,sources=sources,code_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(status='complete',fits=65,subsets=13)))


if __name__=='__main__':
    p=ArgumentParser()
    for key in ['source','root','bundle','out']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
