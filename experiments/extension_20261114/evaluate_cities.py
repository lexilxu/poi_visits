"""Replay city checkpoints and score the frozen earlier-period control design."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import sys
import numpy as np
import pandas as pd


def main(a):
    sys.path.insert(0,str(a.source/'experiments/revision'))
    import torch
    from train_controlled import make_model
    from training_protocol import Protocol,feature_mask
    from run_baseline_checks import predictions,metrics
    from run_extended_baselines import nlinear
    from audit_protocol import low_history
    from summarize_queue import selected_jobs,write_json
    from evaluate_saved import load_events
    from paired_seed_analysis import intervals
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    a.out.mkdir(parents=True,exist_ok=False);private=a.out/'private_predictions';private.mkdir()
    jobs=selected_jobs([a.queue]);assert len(jobs)==20
    rows=[];paired=[];audits=[];fit_records=[]
    for city in ('Seattle','Portland'):
        path=a.panels/(city+'.npz');digest=sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:raw,dates,poi_id=z['visits'],z['dates'],z['poi_id']
        protocol=Protocol(14,7)
        baselines,origins,history,target,split,valid,recent=predictions(raw,dates,14,7,protocol)
        linear,linear_fit,weights=nlinear(raw,history,target,split,valid)
        baselines={k:v for k,v in baselines.items() if k in ('Zero','Calendar HA','Weekday HA','Seasonal naive','Recent mean')}
        baselines['NLinear-ridge']=linear
        np.savez_compressed(private/(city+'_NLinear_weights.npz'),weights=weights,scale=linear_fit['train_scale'])
        write_json(a.out/(city+'_baseline_fits.json'),dict(nlinear=linear_fit,recent_mean=recent))
        test=split=='test';selected=split!='train';target_test=raw[target[test]];valid_test=valid[test]
        low,threshold=low_history(raw[:50]);cohorts={'All':np.ones(len(poi_id),bool),'Low-history':low}
        losses={}
        for model,pred in baselines.items():
            np.savez_compressed(private/(city+'_'+model.replace(' ','_')+'.npz'),predicted_visits=pred[selected],observed_visits=raw[target[selected]],valid=valid[selected],forecast_origin=dates[origins[selected]],dataset_split=split[selected],poi_id=poi_id,horizon_step=np.arange(1,8),T=14,H=7,seed=-1,panel_sha256=digest)
            losses[model]=[np.where(valid_test,np.abs(pred[test]-target_test),0.)]
            for cohort,nodes in cohorts.items():rows.append(dict(city=city,model=model,seed=-1,cohort=cohort,**metrics(target_test,pred[test],valid_test,nodes)))
        for prediction,meta,_ in jobs:
            if not meta['id'].startswith(city+'_'):continue
            data=load_events(prediction);fit=json.loads((prediction.parent/'fit.json').read_text())
            assert fit['bundle_sha256']==digest and str(data['bundle_sha256'])==digest
            np.testing.assert_array_equal(data['poi_id'],poi_id)
            np.testing.assert_array_equal(data['forecast_origin'],dates[origins[selected]])
            np.testing.assert_array_equal(data['observed_visits'],raw[target[selected]])
            saved=torch.load(prediction.parent/'checkpoint.pt',map_location='cpu',weights_only=False)
            model=make_model(meta['model'],7,None,feature_mask('full'),len(poi_id),14,(lambda x:x,lambda x:x),torch.device('cuda:0'),.1).cuda().eval()
            model.load_state_dict(saved['state_dict'],strict=True)
            ids=np.flatnonzero(selected);result=[]
            with torch.no_grad():
                for batch in np.array_split(ids,int(np.ceil(len(ids)/4))):
                    own=np.nan_to_num(raw[history[batch]],nan=0.).astype(np.float32)
                    x=torch.tensor((own-saved['visit_mean'])/saved['visit_std'],device='cuda:0')
                    result.append((model(x,x)*saved['visit_std']+saved['visit_mean']).clamp_min(0).cpu().numpy())
            replay=np.concatenate(result);np.testing.assert_allclose(replay,data['predicted_visits'],rtol=2e-4,atol=2e-4)
            audits.append(dict(id=meta['id'],seed=meta['seed'],strict_replay_passed=True,max_absolute_error=float(np.max(np.abs(replay-data['predicted_visits']))),prediction_sha256=sha256(prediction.read_bytes()).hexdigest()))
            pred=data['predicted_visits'][data['dataset_split']=='test']
            losses.setdefault(meta['model'],[]).append(np.where(valid_test,np.abs(pred-target_test),0.))
            for cohort,nodes in cohorts.items():rows.append(dict(city=city,model=meta['model'],seed=meta['seed'],cohort=cohort,**metrics(target_test,pred,valid_test,nodes)))
            fit_records.append(dict(city=city,model=meta['model'],seed=meta['seed'],parameters=fit['parameters'],seconds=fit['seconds'],validation_MAE=fit['validation_MAE'],lr=fit['lr']))
            del model;torch.cuda.empty_cache()
            print(json.dumps(dict(city=city,model=meta['model'],seed=meta['seed'],status='replayed_evaluated')),flush=True)
        averaged={k:np.mean(v,axis=0) for k,v in losses.items()}
        for model,reference in [('GraphGRU','Matched-LSTM'),('GraphGRU','NLinear-ridge'),('Matched-LSTM','NLinear-ridge')]:
            for cohort,nodes in cohorts.items():
                mask=valid_test&nodes[None,None,:]
                delta=np.where(mask,averaged[model]-averaged[reference],0.)
                result=intervals(delta,mask,np.zeros(len(poi_id),dtype=int))
                paired.append(dict(city=city,model=model,reference=reference,cohort=cohort,delta_MAE=result['delta_MAE'],low=result['origin_7day_low'],high=result['origin_7day_high']))
        write_json(a.out/(city+'_cohort.json'),dict(POIs=len(poi_id),low_history=int(low.sum()),positive_mean_q25=threshold,valid_test_events=int(valid_test.sum()),panel_sha256=digest))
    table=pd.DataFrame(rows);table.to_csv(a.out/'metrics_by_seed.csv',index=False)
    grouped=table.groupby(['city','model','cohort'],sort=False)
    cols=[k for k in table.columns if k not in ('city','model','cohort','seed')]
    summary=grouped[cols].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(a.out/'metrics_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(a.out/'paired_intervals.csv',index=False)
    pd.DataFrame(fit_records).to_csv(a.out/'fit_resources.csv',index=False)
    write_json(a.out/'replay_audit.json',audits)
    write_json(a.out/'complete.json',dict(status='complete',strictly_replayed_checkpoints=len(audits),neural_fits=len(list(a.queue.rglob('fit.json'))),complete_configurations=len(list(a.queue.rglob('selected_runs.json'))),temporal_pairs=len(paired)))


if __name__=='__main__':
    p=ArgumentParser()
    for f in ('source','queue','panels','out'):p.add_argument('--'+f,type=Path,required=True)
    main(p.parse_args())
