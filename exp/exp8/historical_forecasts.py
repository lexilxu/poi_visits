"""Reuse earlier Seattle temporal models for genuinely held-out mapping inputs."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json,sys
import numpy as np


def main(a):
    sys.path.insert(0,str(a.source/'exp/common'))
    import torch
    from exp.common.controlled_training import make_model
    from exp.common.protocol import Protocol, feature_mask
    from exp.common.summarize import selected_jobs
    from exp.common.evaluate import load_events
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.25)
    with np.load(a.inputs/'july_december_original_cohort.npz') as z:
        raw=z['visits'];dates=z['dates'].astype(str);poi_ids=z['poi_id'].astype(str)
    with np.load(a.panels/'Seattle.npz') as z:
        early=z['visits'];early_dates=z['dates'].astype(str);early_ids=z['poi_id'].astype(str)
    assert early_dates[49]=='2020-08-19' and early_dates[63]=='2020-09-02'
    target=np.flatnonzero((dates>='2020-10-01')&(dates<='2020-11-19'));origins=target-6
    assert len(target)==50 and dates[origins].astype('datetime64[D]').min()>np.datetime64('2020-09-02')
    history=origins[:,None]+np.arange(-14,0)
    own=np.nan_to_num(raw[history],nan=0.).astype(np.float32)
    eo,eh,et,es,ev=Protocol(14,7,50,64).windows(early);selected=es!='train'
    audit=[];a.out.mkdir(parents=True,exist_ok=False)
    for path,meta,_ in selected_jobs([a.queue]):
        if not meta['id'].startswith('Seattle_'):continue
        assert meta['model'] in ['GraphGRU','Matched-LSTM']
        saved=torch.load(path.parent/'checkpoint.pt',map_location='cpu',weights_only=False)
        model=make_model(meta['model'],7,None,feature_mask('full'),len(early_ids),14,
                         (lambda x:x,lambda x:x),torch.device('cuda:0'),.1).cuda().eval()
        model.load_state_dict(saved['state_dict'],strict=True)
        def predict(values):
            outputs=[]
            with torch.no_grad():
                for start in range(0,len(values),4):
                    x=torch.tensor((values[start:start+4]-saved['visit_mean'])/saved['visit_std'],device='cuda:0')
                    outputs.append((model(x,x)*saved['visit_std']+saved['visit_mean']).clamp_min(0).cpu().numpy())
            return np.concatenate(outputs)
        stored=load_events(path);ix=np.flatnonzero(selected)[:4]
        replay=predict(np.nan_to_num(early[eh[ix]],nan=0.).astype(np.float32))
        np.testing.assert_allclose(replay,stored['predicted_visits'][:4],rtol=2e-4,atol=2e-4)
        forecast=predict(own)[:,6]
        name=f'{meta["model"]}_seed{meta["seed"]}.npz'
        np.savez_compressed(a.out/name,predictions=forecast,target_dates=dates[target],forecast_origins=dates[origins],poi_id=poi_ids)
        audit.append(dict(model=meta['model'],seed=meta['seed'],checkpoint_sha256=sha256((path.parent/'checkpoint.pt').read_bytes()).hexdigest(),
                          strict_spot_replay=True,max_replay_error=float(np.abs(replay-stored['predicted_visits'][:4]).max()),
                          forecast_sha256=sha256((a.out/name).read_bytes()).hexdigest()))
        print(json.dumps(audit[-1]),flush=True);del model;torch.cuda.empty_cache()
    assert len(audit)==10
    weights_path=a.evaluation/'private_predictions/Seattle_NLinear_weights.npz'
    with np.load(weights_path) as z:weights=z['weights'];scale=float(z['scale'])
    def linear(values):
        x=values.transpose(0,2,1).astype(float);last=x[...,-1]
        design=np.concatenate([(x[...,:-1]-last[...,None])/scale,np.ones((*last.shape,1))],axis=-1)
        return np.maximum(np.einsum('bnt,th->bhn',design,weights)*scale+last[:,None,:],0.)
    with np.load(a.evaluation/'private_predictions/Seattle_NLinear-ridge.npz',allow_pickle=False) as z:ref=dict(z)
    np.testing.assert_array_equal(ref['poi_id'],early_ids)
    np.testing.assert_array_equal(ref['forecast_origin'],early_dates[eo[selected]])
    replay=linear(np.nan_to_num(early[eh[selected]],nan=0.))
    np.testing.assert_allclose(replay,ref['predicted_visits'],rtol=1e-10,atol=1e-10)
    path=a.out/'NLinear-ridge_seed-1.npz'
    np.savez_compressed(path,predictions=linear(own)[:,6],target_dates=dates[target],forecast_origins=dates[origins],poi_id=poi_ids)
    audit.append(dict(model='NLinear-ridge',seed=-1,weights_sha256=sha256(weights_path.read_bytes()).hexdigest(),strict_replay=True,
                      forecast_sha256=sha256(path.read_bytes()).hexdigest()))
    report=dict(status='complete',new_neural_fits=0,forecasts=audit,training_end='2020-08-19',validation_end='2020-09-02',
                mapping_targets=['2020-10-01','2020-11-19'],target_POIs=len(poi_ids),forecast_lead=7,
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='complete',forecast_models=len(audit),target_POIs=len(poi_ids))),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for n in ['source','inputs','panels','queue','evaluation','out']:p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
