"""Load and replay a corrected checkpoint against its frozen source and inputs."""
from exp.common.provenance import source_hash
from pathlib import Path
from hashlib import sha256
import json,socket,sys
import numpy as np
import torch


def load_checkpoint(source,run,bundle):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    source,run,bundle=map(Path,(source,run,bundle))
    torch.set_num_threads(4)
    sys.path[:0]=[str(source),str(source/'exp/common'),
                  str(source/'exp/common'),str(source/'exp/common')]
    from exp.common.evolvegcn import EvolveGCNForecast
    from exp.common.evolvegcn_training import graph_operator
    from exp.common.original_protocol import load_panel, make_windows
    from exp.common.protocol import Protocol
    from exp.common.audit import EXPECTED_BUNDLE_SHA256
    record=json.loads((run/'complete.json').read_text());cfg=record['config']
    legacy=record.get('prediction_arithmetic')=='legacy_numpy'
    torch.set_float32_matmul_precision('high' if legacy else 'highest')
    assert record['status']=='complete' and record['architecture']=='EvolveGCN-H'
    model_path='exp/common/evolvegcn.py'
    assert sha256((Path(__file__).with_name("evolvegcn.py")).read_bytes()).hexdigest()==source_hash(record, model_path)
    for name,field in [('best_weights.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:
        assert sha256((run/name).read_bytes()).hexdigest()==record[field]
    checkpoint=torch.load(run/'best_weights.pt',map_location='cpu',weights_only=True)
    panel=load_panel(bundle,EXPECTED_BUNDLE_SHA256)
    if cfg.get('density'):raise ValueError('This loader expects the full canonical POI cohort')
    if cfg['protocol']=='original_masked':
        w=make_windows(panel.raw,cfg['T'],cfg['H'])
        origins,history,target,mask=w.origins,w.history_idx,w.target_idx,w.valid
        split=np.asarray(['train']*w.n_train+['validation']*w.n_val+['test']*(len(origins)-w.test_start));end=57
        low=(w.history_mean>0)&(w.history_mean<=np.quantile(w.history_mean[w.history_mean>0],.25))&(w.zero_ratio>=.7)
    else:
        origins,history,target,split,mask=Protocol(cfg['T'],cfg['H'],50,64).windows(panel.raw);end=50
        from exp.common.audit import low_history
        low,_=low_history(panel.raw[:end])
    selected=split!='train'
    origins,history,target,split,mask=origins[selected],history[selected],target[selected],split[selected],mask[selected]
    graph,graph_record=graph_operator(cfg['adjacency'],panel,end,torch.device('cuda:0'))
    assert graph_record==checkpoint['graph']==record['graph']
    acceptance=None;canonical=None
    acceptance_path=run.parents[1]/'replay_acceptance'/(run.name+'.json')
    if acceptance_path.exists():
        from exp.common.deterministic_group import DeterministicGroupGraph
        acceptance=json.loads(acceptance_path.read_text())
        assert acceptance['status']=='accepted_numerical_replay'
        assert acceptance['checkpoint_sha256']==record['checkpoint_sha256']
        assert acceptance['original_predictions_sha256']==record['predictions_sha256']
        assert acceptance['source_model_sha256']==source_hash(record, model_path)
        assert acceptance['operator_sha256']==sha256((Path(__file__).resolve().parents[2] / 'exp/common/deterministic_group.py').read_bytes()).hexdigest()
        assert type(graph).__name__=='GroupGraph'
        assert acceptance['independent_process_predictions_bitwise_equal']
        for name in ['max_cohort_MAE_change','max_lead_cohort_MAE_change','max_day_cohort_MAE_change']:
            assert acceptance['bounds'][name]<5e-5
        assert acceptance['bounds']['max_rate_change_pp']<.01
        canonical_path=Path(acceptance['canonical_predictions'])
        assert sha256(canonical_path.read_bytes()).hexdigest()==acceptance['canonical_file_sha256']
        with np.load(canonical_path) as saved:canonical=saved['december'].copy()
        graph=DeterministicGroupGraph(graph)
    model=EvolveGCNForecast(cfg['H'],graph,static=checkpoint['state_dict'].get('static')).cuda()
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    with np.load(run/'predictions.npz',allow_pickle=False) as saved:
        expected=saved['predicted_visits']
        np.testing.assert_array_equal(saved['poi_id'],panel.attrs.placekey.to_numpy(dtype=str))
        np.testing.assert_array_equal(saved['forecast_origin'],np.asarray(panel.dates)[origins])
        np.testing.assert_array_equal(saved['dataset_split'],split)
        np.testing.assert_array_equal(saved['observed_visits'],panel.raw[target])
        np.testing.assert_array_equal(saved['valid'],mask)
    own=panel.filled[history]
    loaded=dict(model=model,record=record,config=cfg,panel=panel,mean=checkpoint['mean'],std=checkpoint['std'],
                origins=origins,history=history,target=target,split=split,valid=mask,own=own,prediction=expected,
                low_history=low,train_end=end,run=run)
    if legacy:
        groups=[np.flatnonzero(split==name) for name in ['validation','test']]
        batches=[ids[start:start+cfg['batch_size']] for ids in groups for start in range(0,len(ids),cfg['batch_size'])]
    else:
        batches=np.array_split(np.arange(len(own)),int(np.ceil(len(own)/cfg['batch_size'])))
    loaded['replay_batches']=batches
    replay=predict(loaded,own,batch_size=cfg['batch_size'],batches=batches)
    strict=bool(np.allclose(replay,expected,rtol=1e-5,atol=1e-5))
    if acceptance is None:
        np.testing.assert_allclose(replay,expected,rtol=1e-5,atol=1e-5)
    else:
        np.testing.assert_array_equal(replay,canonical)
    loaded['replayed_prediction']=replay
    loaded['numerical_replay_acceptance']=acceptance
    loaded['replay_max_absolute_error']=float(np.abs(replay-expected).max())
    keep=mask&(split=='test')[:,None,None]
    truth=np.nan_to_num(panel.raw[target],nan=0.)
    loaded['replay_audit']=dict(max_absolute_error=loaded['replay_max_absolute_error'],
        mean_absolute_error=float(np.abs(replay-expected)[keep].mean()),
        test_MAE_change=float((np.abs(replay-truth)-np.abs(expected-truth))[keep].mean()),
        activation_disagreements_at_0_5=int(((replay>=.5)!=(expected>=.5))[keep].sum()),
        test_events=int(keep.sum()),absolute_tolerance=1e-5,relative_tolerance=1e-5,
        strict_replay_passed=strict,numerical_replay_verified=acceptance is not None,
        acceptance_sha256=sha256(acceptance_path.read_bytes()).hexdigest() if acceptance is not None else None)
    return loaded


def origin_batch_size(loaded,index):
    matches=[len(ids) for ids in loaded['replay_batches'] if index in ids]
    if len(matches)!=1:raise ValueError('Origin must belong to one archived prediction batch')
    return matches[0]


def predict(loaded,own,batch_size=4,batches=None):
    out=[]
    if batches is None:
        batches=[np.arange(start,min(start+batch_size,len(own))) for start in range(0,len(own),batch_size)]
    legacy=loaded['record'].get('prediction_arithmetic')=='legacy_numpy'
    with torch.no_grad():
        for ids in batches:
            x=torch.as_tensor((own[ids]-loaded['mean'])/loaded['std'],device='cuda',dtype=torch.float32)
            prediction=loaded['model'](x)
            if legacy:
                out.append(np.maximum(prediction.cpu().float().numpy()*loaded['std']+loaded['mean'],0))
            else:
                out.append(torch.clamp(prediction*loaded['std']+loaded['mean'],min=0.).cpu().numpy())
    return np.concatenate(out)


def world_forward(loaded,normalized_worlds):
    """Use eight physical worlds for every intervention forward, padding the tail."""
    count=len(normalized_worlds)
    if not 1<=count<=8:raise ValueError('Canonical intervention batches contain at most eight worlds')
    padded=torch.cat([normalized_worlds,normalized_worlds[-1:].expand(8-count,-1,-1)],dim=0)
    values=loaded['model'](padded)[:count]
    if loaded['record'].get('prediction_arithmetic')=='legacy_numpy':
        return np.maximum(values.detach().cpu().float().numpy()*loaded['std']+loaded['mean'],0)
    return torch.clamp(values*loaded['std']+loaded['mean'],min=0.).detach().cpu().numpy()


def predict_worlds(loaded,own):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(own),8):
            x=torch.as_tensor((own[start:start+8]-loaded['mean'])/loaded['std'],device='cuda',dtype=torch.float32)
            outputs.append(world_forward(loaded,x))
    return np.concatenate(outputs)


def targeted_worlds(loaded,history,targets,peer_sets,donor_sets=None,batch_size=8,return_baseline=False):
    """Recompute all node states and weights in one counterfactual per target."""
    targets=np.asarray(targets);results=[];baselines=[]
    if not 1<=batch_size<=8:raise ValueError('Intervention chunks use one to eight targets')
    base=torch.as_tensor((history-loaded['mean'])/loaded['std'],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        for start in range(0,len(targets),batch_size):
            stop=min(start+batch_size,len(targets));worlds=base[None].expand(stop-start,-1,-1).clone()
            if return_baseline:
                unperturbed=world_forward(loaded,worlds)
                baselines.append(unperturbed[np.arange(stop-start),:,targets[start:stop]])
            for row in range(start,stop):
                peers=np.asarray(peer_sets[row],dtype=np.int64)
                if len(peers)==0:continue
                if targets[row] in peers:raise ValueError('Neighbor intervention includes its target')
                if donor_sets is None:worlds[row-start,:,peers]=(0-loaded['mean'])/loaded['std']
                else:
                    donors=np.asarray(donor_sets[row],dtype=np.int64)
                    assert len(peers)==len(donors)
                    worlds[row-start,:,peers]=base[:,donors]
            output=world_forward(loaded,worlds)
            results.append(output[np.arange(stop-start),:,targets[start:stop]])
    after=np.concatenate(results).T
    return (after,np.concatenate(baselines).T) if return_baseline else after
