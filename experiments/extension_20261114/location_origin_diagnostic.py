"""Prediction-preserving coordinate-origin test of the original static checkpoint.

Run on AutoDL4 only. No training, no changed checkpoints, no output overwrite.
"""
from pathlib import Path
from argparse import ArgumentParser
import hashlib
import json
import sys
import time


def main(a):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(a.repo))
    sys.path.insert(0, str(a.formal))
    from train_formal import GraphGRU, load_panel, make_windows, static_features
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.35)
    a.out.mkdir(parents=True, exist_ok=False)
    checkpoint = a.formal/'run/T14_H7/gnn-static__identity.pt'
    archived = a.formal/'run/T14_H7/gnn-static__identity.npy'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    panel = load_panel(a.formal/'approved_panel.zip',
                       '12d6414990eeb5703b1179152b5993518dcfa68361e39e09ae621919950695ec')
    w = make_windows(panel.raw, 14, 7)
    static, _ = static_features(panel.attrs)
    model = GraphGRU(7, static).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cuda:0', weights_only=True), strict=True)
    model.requires_grad_(False)
    fit = json.loads((checkpoint.with_suffix('.json')).read_text())
    batch_size = int(fit['batch_size'])
    history = (panel.filled[w.history_idx[w.test_start:]]-w.mean)/w.std
    n = len(panel.attrs)
    origins = [panel.dates[i] for i in w.origins[w.test_start:]]

    def predict(m):
        parts = []
        with torch.no_grad():
            for i in range(0,len(history),batch_size):
                x = torch.as_tensor(history[i:i+batch_size], device='cuda:0',dtype=m.project.weight.dtype)
                parts.append((m(x,x)*w.std+w.mean).clamp_min(0).cpu().numpy())
        return np.concatenate(parts)

    fp32 = predict(model)
    old = np.load(archived, allow_pickle=False)
    replay = dict(max_abs=float(np.abs(fp32-old).max()),
                  mean_abs=float(np.abs(fp32-old).mean()),
                  atol=0.002, rtol=0.001,
                  passed=bool(np.allclose(fp32,old,atol=.002,rtol=.001)))
    assert replay['passed'], replay
    model = model.double()
    before_prediction = predict(model)
    # Median coordinates depend only on the fixed metadata, not forecast outcomes.
    shift = model.static[:,:2].median(dim=0).values.detach().clone()
    before_encoded = model.encode_static().detach().clone()

    def attribution(m):
        m.gru.train()  # Required by cuDNN backward; one layer and zero dropout.
        raw = np.zeros((n,5),np.float64)
        encoded_static=m.encode_static().detach()
        weights=m.project.weight
        static_projection=F.linear(encoded_static,weights[:,1:])
        for i in range(0,len(history),batch_size):
            hx=torch.as_tensor(history[i:i+batch_size],device='cuda:0',dtype=torch.float64)
            projected=(F.linear(hx[...,None],weights[:,:1],m.project.bias)+static_projection).detach().requires_grad_(True)
            last=hx[:,-1,:].detach().requires_grad_(True)
            encoded=torch.cat((hx[...,None],encoded_static[None,None].expand(len(hx),14,-1,-1)),dim=-1)
            logits=m.forecast_projected(projected,last)
            for h in range(7):
                dprojected,dlast=torch.autograd.grad(logits[:,h].sum(),(projected,last),retain_graph=h<6)
                gradient=dprojected@weights
                gradient[:,-1,:,0]+=dlast
                scores=(encoded*gradient).abs().sum(1).cpu().numpy()
                group=np.stack((scores[...,0],scores[...,1:3].sum(-1),
                    scores[...,5:37].sum(-1)+scores[...,45:53].sum(-1),
                    scores[...,37:45].sum(-1),scores[...,3:5].sum(-1)),axis=-1)
                raw+=group.sum(0)
            print(json.dumps({'phase':'attribution','origin_count':min(i+batch_size,len(history)),
                              'total_origins':len(history)}),flush=True)
        m.eval()
        return raw

    before=attribution(model)
    with torch.no_grad():
        model.static[:,:2].sub_(shift)
        model.project.bias.add_(model.project.weight[:,1:3]@shift)
    after_prediction=predict(model)
    np.testing.assert_allclose(after_prediction,before_prediction,atol=1e-9,rtol=1e-9)
    after=attribution(model)
    eligible=(before.sum(1)>0)&(after.sum(1)>0)
    def profiles(raw):
        shares=raw[eligible]/raw[eligible].sum(1,keepdims=True)
        standardized=(shares-shares.mean(0))/np.maximum(shares.std(0),1e-12)
        return shares,standardized.argmax(1)
    bs,br=profiles(before);ars,ar=profiles(after)
    unchanged_groups=[0,2,3,4]
    np.testing.assert_allclose(before[:,unchanged_groups],after[:,unchanged_groups],atol=1e-8,rtol=1e-8)
    pd.DataFrame({'placekey':panel.attrs.placekey.to_numpy()[eligible],
                  'location_share_before':bs[:,1],'location_share_after':ars[:,1],
                  'regime_before':br,'regime_after':ar}).to_csv(a.out/'profiles.csv',index=False)
    np.savez_compressed(a.out/'absolute_attribution.npz',before=before,after=after,
                        placekey=panel.attrs.placekey.to_numpy(str),shift=shift.cpu().numpy())
    report=dict(status='complete',checkpoint_sha256=checkpoint_hash,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                formal_source_sha256=hashlib.sha256((a.formal/'train_formal.py').read_bytes()).hexdigest(),
                numerical_scope='FP64 paired evaluation of original stored parameters; FP32 replay separately checked',
                forecast_origin_dates=origins,POIs=n,eligible_POIs=int(eligible.sum()),seed=42,
                T=14,H=7,batch_size=batch_size,shift_latitude_longitude=shift.cpu().tolist(),
                archived_FP32_replay=replay,
                maximum_paired_prediction_change=float(np.abs(before_prediction-after_prediction).max()),
                mean_location_share_before=float(bs[:,1].mean()),mean_location_share_after=float(ars[:,1].mean()),
                location_dominant_before=int((bs.argmax(1)==1).sum()),location_dominant_after=int((ars.argmax(1)==1).sum()),
                standardized_regime_agreement=float((br==ar).mean()),
                regimes_before=np.bincount(br,minlength=5).tolist(),regimes_after=np.bincount(ar,minlength=5).tolist(),
                group_order=['history','location','identity','parent','others'],
                output_attribution='Original normalized pre-clipping output; absolute values before lag aggregation',
                checkpoint_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_hash,
                completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    assert report['checkpoint_unchanged']
    (a.out/'complete.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=ArgumentParser()
    for name in ['formal','repo','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
