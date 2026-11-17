"""Frozen corrected EvolveGCN forecasts on the existing January holdout."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,time
import numpy as np
from exp.common.evolvegcn_io import load_checkpoint, predict


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def evaluate(args):
    if args.out.exists():raise FileExistsError('Existing evaluation is immutable: '+str(args.out))
    manifest=json.loads(args.manifest.read_text())
    assert manifest['status']=='complete'
    panel_hash=digest(args.panel)
    assert panel_hash==manifest['panel_sha256']
    loaded=load_checkpoint(args.source,args.run,args.bundle)
    cfg=loaded['config'];old=loaded['panel'];record=loaded['record']
    assert manifest['original_bundle_sha256']==record['bundle_sha256']
    with np.load(args.panel,allow_pickle=False) as z:
        raw=z['visits'];dates=z['dates'];ids=z['poi_id']
        np.testing.assert_array_equal(ids,old.attrs.placekey.to_numpy(dtype=str))
        np.testing.assert_array_equal(raw[:92],old.raw)
        np.testing.assert_array_equal(dates[:92],np.asarray(old.dates))
        assert str(z['original_bundle_sha256'])==record['bundle_sha256']
    assert raw.shape==(123,9884)
    np.testing.assert_array_equal(dates[92:],np.arange('2021-01-01','2021-02-01',dtype='datetime64[D]').astype(str))
    origins=np.arange(92,124-cfg['H'])
    history=origins[:,None]+np.arange(-cfg['T'],0)
    target=origins[:,None]+np.arange(cfg['H'])
    assert (history<origins[:,None]).all() and target.max()==122
    own=np.nan_to_num(raw[history],nan=0.).astype(np.float32)
    forecast=predict(loaded,own,batch_size=cfg['batch_size'])
    assert forecast.shape==(len(origins),cfg['H'],9884)
    assert np.isfinite(forecast).all() and (forecast>=0).all()
    acceptance=loaded.get('numerical_replay_acceptance')
    if acceptance is not None:
        assert acceptance['january_panel_sha256']==panel_hash
        assert sha256(forecast.tobytes()).hexdigest()==acceptance['january_prediction_sha256']
    valid=np.isfinite(raw[target]);truth=np.nan_to_num(raw[target],nan=0.)
    error=np.abs(forecast.astype(np.float64)-truth)
    metrics=[]
    for name,cohort in [('All',np.ones(9884,dtype=bool)),('Low-history',loaded['low_history'])]:
        selected=valid&cohort[None,None,:]
        positive=selected&(truth>0);zero=selected&(truth==0)
        metrics.append(dict(cohort=name,pois=int(cohort.sum()),events=int(selected.sum()),
            MAE=float(error[selected].mean()),RMSE=float(np.sqrt(np.square(error[selected]).mean())),
            DMR=float(forecast[positive].sum(dtype=np.float64)/truth[positive].sum(dtype=np.float64)) if positive.any() else None,
            FAR_0_5=float((forecast[zero]>=.5).mean()) if zero.any() else None,
            observed_positive_events=int(positive.sum()),observed_zero_events=int(zero.sum())))
    args.out.mkdir(parents=True,exist_ok=False)
    # Validation predictions remain identical to those used to select this fit.
    val=loaded['split']=='validation'
    np.savez_compressed(args.out/'predictions.npz',
        predicted_visits=np.concatenate([loaded['prediction'][val],forecast]),
        observed_visits=np.concatenate([old.raw[loaded['target'][val]],raw[target]]),
        valid=np.concatenate([loaded['valid'][val],valid]),
        forecast_origin=np.r_[np.asarray(old.dates)[loaded['origins'][val]],dates[origins]],
        dataset_split=np.asarray(['validation']*int(val.sum())+['test']*len(origins)),
        poi_id=ids,horizon_step=np.arange(1,cfg['H']+1),seed=cfg['seed'],T=cfg['T'],H=cfg['H'],
        protocol=cfg['protocol'],train_end=loaded['train_end'],period='2021-01')
    summary=dict(status='complete',architecture='EvolveGCN-H',config=cfg,period='2021-01',
        forecast_origins=len(origins),frozen_parameters=True,metrics=metrics,
        original_replay=loaded['replay_audit'],checkpoint_sha256=record['checkpoint_sha256'],
        original_predictions_sha256=record['predictions_sha256'],
        panel_sha256=panel_hash,manifest_sha256=digest(args.manifest),
        predictions_sha256=digest(args.out/'predictions.npz'),
        code_sha256={p.name:digest(p) for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn_io.py'),(Path(__file__).resolve().parents[2] / 'exp/common/deterministic_group.py')]},
        missing_history_values_filled_zero=int(np.isnan(raw[history]).sum()),
        missing_label_events_excluded=int((~valid).sum()),completed_utc=time.time(),
        evaluation='Same-city later-month rolling-origin holdout. Original model selection and scaling remain frozen.')
    (args.out/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(run=args.run.name,origins=len(origins),metrics=metrics)),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    for key in ['source','run','bundle','panel','manifest','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    evaluate(parser.parse_args())
