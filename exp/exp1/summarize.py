"""Recompute original-table metrics on AutoDL4 from hash-verified predictions.

The original four comparator rows remain tied to their original prediction files.
The corrected EvolveGCN row is selected by validation only after all distinct
adjacencies for that setting/seed are available. Partial snapshots are explicit.
"""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,socket,sys,time
import numpy as np
import pandas as pd


def correlation(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom else float('nan')


def metrics(y,p,ha,selected):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64);ha=np.asarray(ha,dtype=np.float64)
    error=p-y;mae=float(np.abs(error).mean());baseline=float(np.abs(y-ha).mean())
    low_y=y[:,:,selected];low_p=p[:,:,selected];low_ha=ha[:,:,selected];positive=low_y>0
    mass=low_y[positive].sum();dmr=low_p[positive].sum()/mass;dmr_ha=low_ha[positive].sum()/mass
    return dict(mae=mae,rmse=float(np.sqrt(np.square(error).mean())),
                rse=float(np.linalg.norm(error.ravel())/np.linalg.norm((y-y.mean()).ravel())),
                corr=correlation(y,p),skill_percent=100*(1-mae/baseline),
                anomaly_corr=correlation(y-ha,p-ha),lift=float((low_p-low_ha).mean()),
                dmr_model=float(dmr),dmr_ha=float(dmr_ha),delta_dmr=float(dmr-dmr_ha),
                far_0_5=float((low_p[~positive]>=.5).mean()),mean_prediction=float(low_p.mean()),
                low_history_pois=int(selected.sum()),events=int(y.size))


def main(args):
    sys.path[:0]=[str(args.source),str(args.source/'exp/common'),str(args.source/'exp/common')]
    from exp.common.original_protocol import load_panel, make_windows, model_key, group_codes
    from exp.common.audit import EXPECTED_BUNDLE_SHA256
    panel=load_panel(args.bundle,EXPECTED_BUNDLE_SHA256)
    groups=group_codes(panel.attrs)
    assert np.array_equal(groups['NAICS-4'],groups['Top category'])
    args.out.mkdir(parents=True,exist_ok=False)
    all_rows=[];main_rows=[];checks=[];selection=[];cache={}
    for t,h in [(14,7),(28,7),(28,14),(28,28)]:
        w=make_windows(panel.raw,t,h);y=w.y[w.test_start:]
        assert w.valid[w.test_start:].all()
        old=args.legacy/f'T{t}_H{h}'
        ha=np.load(old/(model_key('Historical average','None')+'.npy'),allow_pickle=False)
        selected=(w.history_mean>0)&(w.history_mean<=np.quantile(w.history_mean[w.history_mean>0],.25))&(w.zero_ratio>=.7)
        cache[(t,h)]=(w,y,ha,selected)
        mtg=['Identity','NAICS-2','Top category','Distance KNN']
        best=min(mtg,key=lambda adj:json.loads((old/(model_key('MTGNN',adj)+'.json')).read_text())['validation_loss'])
        for model,adj in [('Historical average','None'),('AR-ridge','None'),('LSTM','None'),('MTGNN',best)]:
            file=old/(model_key(model,adj)+'.npy');prediction=np.load(file,allow_pickle=False)
            assert prediction.shape==y.shape and np.isfinite(prediction).all()
            main_rows.append(dict(T=t,H=h,model=model,adjacency=adj,seed=42,
                                  source='preserved_original_prediction',**metrics(y,prediction,ha,selected)))
            checks.append(dict(path=str(file),sha256=sha256(file.read_bytes()).hexdigest(),kind='preserved_original'))
    completed=[]
    for file in sorted((args.root/'runs').glob('*/complete.json')):
        record=json.loads(file.read_text());cfg=record['config']
        if cfg['protocol']!='original_masked' or cfg['static']:continue
        prediction_file=file.parent/'predictions.npz'
        checkpoint_file=file.parent/'best_weights.pt'
        assert sha256(prediction_file.read_bytes()).hexdigest()==record['predictions_sha256']
        assert sha256(checkpoint_file.read_bytes()).hexdigest()==record['checkpoint_sha256']
        w,y,ha,selected=cache[(cfg['T'],cfg['H'])]
        with np.load(prediction_file,allow_pickle=False) as saved:
            test=saved['dataset_split']=='test'
            prediction=saved['predicted_visits'][test]
            np.testing.assert_array_equal(saved['poi_id'],panel.attrs.placekey.to_numpy(dtype=str))
            np.testing.assert_array_equal(saved['forecast_origin'][test],np.asarray(panel.dates)[w.origins[w.test_start:]])
            np.testing.assert_array_equal(saved['observed_visits'][test],y)
            assert saved['valid'][test].all()
        scores=metrics(y,prediction,ha,selected)
        assert abs(scores['mae']-record['test_MAE'])<1e-5
        row=dict(T=cfg['T'],H=cfg['H'],model='EvolveGCN',adjacency=cfg['adjacency'],seed=cfg['seed'],
                 source='corrected_EvolveGCN_H',validation_loss=record['validation_loss'],**scores)
        all_rows.append(row);completed.append(cfg)
        checks.append(dict(path=str(prediction_file),sha256=record['predictions_sha256'],kind='corrected_prediction'))
    expected={'Identity','NAICS-2','NAICS-4','Distance KNN','Distance RBF','Visit correlation'}
    for t,h in cache:
        rows=[r for r in all_rows if (r['T'],r['H'],r['seed'])==(t,h,42)]
        complete={r['adjacency'] for r in rows}==expected
        state=dict(T=t,H=h,primary_seed=42,available_graphs=len(rows),expected_graphs=len(expected),complete=complete)
        if complete:
            identity=next(r for r in rows if r['adjacency']=='Identity')
            best=min((r for r in rows if r['adjacency']!='Identity'),key=lambda r:r['validation_loss'])
            main_rows.extend([dict(identity,series='identity'),dict(best,series='best non-identity')])
            state.update(selected_nonidentity=best['adjacency'],selection_metric='validation masked MSE')
        selection.append(state)
    pd.DataFrame(all_rows).to_csv(args.out/'corrected_by_seed.csv',index=False)
    pd.DataFrame(main_rows).to_csv(args.out/'original_table_replacement_snapshot.csv',index=False)
    report=dict(status='complete_primary_forecast_table' if all(s['complete'] for s in selection) else 'partial_snapshot',
                created_utc=time.time(),completed_corrected_forecast_fits=len(completed),selection=selection,
                alias={'Top category':'NAICS-4'},checks=checks,
                unchanged_comparators='Historical average, AR-ridge, LSTM, MTGNN use the original saved predictions.',
                scope='Forecast metrics only; density, intervention, attribution and multi-seed uncertainty remain separate gates.')
    (args.out/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['status','completed_corrected_forecast_fits','selection']}),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for key in ['source','root','bundle','legacy','out']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
