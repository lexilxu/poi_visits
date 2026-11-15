"""Translate parent embeddings and compensate bias, preserving the predictor."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import sys
import json
import numpy as np
import pandas as pd


def load_fitted(directory,bundle):
    import torch
    from audit_protocol import load_panel
    from training_protocol import Protocol,feature_mask
    from train_controlled import make_model
    from train_formal import static_features
    fit=json.loads((directory/'fit.json').read_text())
    saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    raw,dates,attrs=load_panel(bundle)
    origins,history,target,split,valid=Protocol(14,7).windows(raw);keep=split!='train'
    own=np.nan_to_num(raw[history[keep]],nan=0.).astype(np.float32)
    static,_=static_features(attrs)
    static[:,:4]=(static[:,:4]-saved['numeric_mean'])/saved['numeric_std']
    model=make_model('GNN-static',7,static,feature_mask('full'),len(attrs),14,(lambda x:x,lambda x:x),torch.device('cuda:0'),.1).cuda().eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    def predict(m):
        result=[]
        with torch.no_grad():
            for ids in np.array_split(np.arange(len(own)),int(np.ceil(len(own)/saved['args']['batch_size']))):
                x=torch.tensor((own[ids]-saved['visit_mean'])/saved['visit_std'],device='cuda:0',dtype=m.project.weight.dtype)
                result.append((m(x,x)*saved['visit_std']+saved['visit_mean']).clamp_min(0).cpu().numpy())
        return np.concatenate(result)
    prediction=predict(model)
    with np.load(directory/'predictions.npz') as z:
        np.testing.assert_array_equal(z['forecast_origin'],dates[origins[keep]])
        archive_difference=np.abs(prediction-z['predicted_visits'])
        archive_check=dict(strict_replay_passed=bool(np.allclose(prediction,z['predicted_visits'],rtol=2e-4,atol=2e-4)),
                           max_difference=float(archive_difference.max()),mean_difference=float(archive_difference.mean()))
    model=model.double();prediction=predict(model)
    return dict(model=model,fit=fit,attrs=attrs,dates=dates,origins=origins[keep],split=split[keep],own=own,
                mean=saved['visit_mean'],std=saved['visit_std'],prediction=prediction,predict=predict,archive_check=archive_check)


def attribution64(model,history,origins,mean,std):
    import torch
    from attribution_audit import gradient_attribution
    n=len(model.static);t=history.shape[1]
    pairs=np.stack(np.meshgrid(origins,np.arange(n),indexing='ij'),axis=-1).reshape(-1,2)
    absolute=np.zeros((n,t+52),np.float64);encoded=model.encode_static().detach()
    with torch.backends.cudnn.flags(enabled=False):
        for start in range(0,len(pairs),512):
            steps,nodes=pairs[start:start+512].T
            h=torch.as_tensor((history[steps,:,nodes]-mean)/std,device='cuda:0',dtype=torch.float64)
            points=torch.cat([h,encoded[nodes]],dim=1).detach()
            values,_=gradient_attribution(model,points,t,mean,std)
            np.add.at(absolute,nodes,values.abs().sum(1).detach().cpu().numpy())
    return absolute


def main(a):
    sys.path.insert(0,str(a.source/'experiments/revision'))
    import torch
    from checkpoint_tools import selected_directories
    from summarize_attribution import profile,agreement
    from summarize_queue import write_json
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.25)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'private_attribution').mkdir()
    rows=[];before_shares=[];after_shares=[];eligible=[]
    directories=selected_directories(a.queue,'GNN-static','Identity','full',14);assert len(directories)==5
    for directory in directories:
        loaded=load_fitted(directory,a.bundle);model=loaded['model'];seed=loaded['fit']['seed'];t=14
        missing=loaded['attrs'].parent_placekey.fillna('').astype(str).eq('').to_numpy()
        category=np.unique(model.static.detach().cpu().numpy()[missing,8]);assert len(category)==1
        prior=a.attribution/f'seed{seed}'
        protocol=json.loads((prior/'attribution_protocol.json').read_text())
        test=np.flatnonzero(loaded['split']=='test');origins=test[np.unique(np.linspace(0,len(test)-1,8,dtype=int))]
        assert protocol['origins']==loaded['dates'][loaded['origins'][origins]].tolist()
        previous=prior/'private_attribution/Gradient_x_Input.npz'
        with np.load(previous) as z:
            np.testing.assert_array_equal(z['poi_id'],loaded['attrs'].placekey.to_numpy(str))
        before=attribution64(model,loaded['own'],origins,loaded['mean'],loaded['std'])
        with torch.no_grad():
            shift=model.embeddings[4].weight[int(category[0])].clone()
            model.embeddings[4].weight.sub_(shift)
            model.project.bias.add_(model.project.weight[:,37:45]@shift)
        after_prediction=loaded['predict'](model)
        np.testing.assert_allclose(after_prediction,loaded['prediction'],rtol=1e-10,atol=1e-10)
        delta=np.abs(after_prediction-loaded['prediction'])
        after=attribution64(model,loaded['own'],origins,loaded['mean'],loaded['std'])
        assert (after[missing,t+36:t+44]==0).all()
        sa,ra,_,va=profile(before,t);sb,rb,_,vb=profile(after,t)
        both=va&vb
        row=dict(seed=seed,max_prediction_difference=float(delta.max()),mean_prediction_difference=float(delta.mean()),
                 archived_FP32_replay=loaded['archive_check'],paired_arithmetic='FP64',
                 missing_parent_POIs=int(missing.sum()),missing_parent_regime_before=int(((ra==3)&missing&both).sum()),
                 missing_parent_regime_after=int(((rb==3)&missing&both).sum()),
                 checkpoint_sha256=sha256((directory/'checkpoint.pt').read_bytes()).hexdigest(),
                 original_attribution_sha256=sha256(previous.read_bytes()).hexdigest(),
                 **agreement(before,after,t))
        rows.append(row);before_shares.append(sa);after_shares.append(sb);eligible.append(both)
        np.savez_compressed(a.out/'private_attribution'/f'seed{seed}.npz',absolute=after,original_absolute=before,poi_id=loaded['attrs'].placekey.to_numpy(str),shift=shift.cpu().numpy())
        write_json(a.out/f'seed{seed}.json',row);print(json.dumps(row),flush=True)
        del model;torch.cuda.empty_cache()
    valid=np.logical_and.reduce(eligible)
    sa=np.mean(before_shares,axis=0);sb=np.mean(after_shares,axis=0)
    def regime(shares):
        standardized=(shares[valid]-shares[valid].mean(0))/np.maximum(shares[valid].std(0),1e-12)
        return standardized.argmax(1)
    ra,rb=regime(sa),regime(sb)
    report=dict(status='complete',seeds=5,eligible_POIs=int(valid.sum()),regime_agreement=float((ra==rb).mean()),
                parent_regime_before=int((ra==3).sum()),parent_regime_after=int((rb==3).sum()),
                missing_parent_regime_before=int(((ra==3)&missing[valid]).sum()),
                missing_parent_regime_after=int(((rb==3)&missing[valid]).sum()),
                mean_parent_share_before=float(sa[valid,3].mean()),mean_parent_share_after=float(sb[valid,3].mean()),
                max_prediction_difference=max(r['max_prediction_difference'] for r in rows),
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    pd.DataFrame(rows).to_csv(a.out/'by_seed.csv',index=False);write_json(a.out/'complete.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for f in ('source','queue','bundle','attribution','out'):p.add_argument('--'+f,type=Path,required=True)
    main(p.parse_args())
