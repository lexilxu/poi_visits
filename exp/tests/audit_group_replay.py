"""Independent-process numerical audit of categorical graph replay candidates.

This exports diagnostic candidates and comparisons. It grants no replay waiver
and never changes model selection, fitted parameters, or archived predictions.
"""
from argparse import ArgumentParser
from exp.common.provenance import source_hash
from pathlib import Path
from hashlib import sha256
import json, socket, sys, time
import numpy as np
import torch
from exp.common.deterministic_group import DeterministicGroupGraph


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def reporting_change(candidate,reference,truth,valid,low,target_dates):
    candidate,reference,truth=[x.astype(np.float64) for x in (candidate,reference,truth)]
    error=np.abs(candidate-reference)
    loss=np.abs(candidate-truth)-np.abs(reference-truth)
    rows=[]
    for name,nodes in [('All',np.ones(truth.shape[-1],bool)),('Low-history',low)]:
        mask=valid&nodes[None,None,:]
        details=dict(cohort=name,events=int(mask.sum()),delta_MAE=float(loss[mask].mean()),
            lead_delta_MAE=[float(loss[:,h][mask[:,h]].mean()) for h in range(truth.shape[1])],
            day_delta_MAE={str(day):float(loss[mask&(target_dates==day)[:,:,None]].mean()) for day in np.unique(target_dates)},
            thresholds=[])
        positive=mask&(truth>0);zero=mask&(truth==0)
        for threshold in [.1,.25,.5,1.,2.]:
            rates=[]
            for forecast in [candidate,reference]:
                active=forecast>=threshold
                tp=int((active&positive).sum());fp=int((active&zero).sum())
                rates.append(np.array([fp/max(1,int(zero.sum())),tp/max(1,int(positive.sum())),tp/max(1,tp+fp)]))
            differences=100*(rates[0]-rates[1])
            details['thresholds'].append(dict(threshold=threshold,FAR_pp=float(differences[0]),
                recall_pp=float(differences[1]),precision_pp=float(differences[2]),
                event_classification_disagreements=int(((candidate>=threshold)!=(reference>=threshold))[mask].sum())))
        rows.append(details)
    return dict(max_abs_prediction_difference=float(error.max()),mean_abs_prediction_difference=float(error[valid].mean()),
        strict_allclose_1e_5=bool(np.allclose(candidate,reference,rtol=1e-5,atol=1e-5)),cohorts=rows)


def main(a):
    if not torch.cuda.is_available():raise RuntimeError('AutoDL4 required')
    sys.path[:0]=[str(a.source),str(a.source/'exp/common'),
                  str(a.source/'exp/common'),str(a.source/'exp/common')]
    from exp.common.original_protocol import load_panel, make_windows
    from exp.common.audit import EXPECTED_BUNDLE_SHA256, low_history
    from exp.common.protocol import Protocol
    from exp.common.evolvegcn_training import graph_operator
    from exp.common.evolvegcn import EvolveGCNForecast
    panel=load_panel(a.bundle,EXPECTED_BUNDLE_SHA256)
    manifest=json.loads(a.january_manifest.read_text())
    assert manifest['status']=='complete' and manifest['panel_sha256']==digest(a.january_panel)
    with np.load(a.january_panel) as z:
        full=z['visits'].copy();dates=z['dates'].copy()
        np.testing.assert_array_equal(full[:92],panel.raw)
        np.testing.assert_array_equal(z['poi_id'],panel.attrs.placekey.to_numpy(dtype=str))
    cases=json.loads(a.cases.read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    rows=[]
    for name in cases:
        began=time.monotonic();run=a.root/'runs'/name
        record=json.loads((run/'complete.json').read_text());cfg=record['config']
        assert cfg['adjacency'] in ['NAICS-2','NAICS-4'] and not cfg['static']
        for filename,key in [('best_weights.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:
            assert digest(run/filename)==record[key]
        model_file='exp/common/evolvegcn.py'
        assert digest(a.source/model_file)==source_hash(record, model_file)
        checkpoint=torch.load(run/'best_weights.pt',map_location='cpu',weights_only=True)
        legacy=record.get('prediction_arithmetic')=='legacy_numpy'
        torch.set_float32_matmul_precision('high' if legacy else 'highest')
        if cfg['protocol']=='original_masked':
            w=make_windows(panel.raw,cfg['T'],cfg['H'])
            origins,history,target,valid=w.origins,w.history_idx,w.target_idx,w.valid
            split=np.asarray(['train']*w.n_train+['validation']*w.n_val+['test']*(len(origins)-w.test_start))
            low=(w.history_mean>0)&(w.history_mean<=np.quantile(w.history_mean[w.history_mean>0],.25))&(w.zero_ratio>=.7)
            end=57
        else:
            origins,history,target,split,valid=Protocol(cfg['T'],cfg['H'],50,64).windows(panel.raw)
            low,_=low_history(panel.raw[:50]);end=50
        keep=split!='train';origins,history,target,valid,split=[v[keep] for v in [origins,history,target,valid,split]]
        with np.load(run/'predictions.npz') as z:
            reference=z['predicted_visits'].copy()
            np.testing.assert_array_equal(z['forecast_origin'],np.asarray(panel.dates)[origins])
            np.testing.assert_array_equal(z['observed_visits'],panel.raw[target])
            np.testing.assert_array_equal(z['valid'],valid)
        graph,graph_record=graph_operator(cfg['adjacency'],panel,end,torch.device('cuda'))
        assert graph_record==record['graph']==checkpoint['graph']
        model=EvolveGCNForecast(cfg['H'],DeterministicGroupGraph(graph)).cuda()
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        mean,std=checkpoint['mean'],checkpoint['std']
        def predict(own,batches):
            outputs=[]
            with torch.no_grad():
                for ids in batches:
                    x=torch.as_tensor((own[ids]-mean)/std,device='cuda',dtype=torch.float32)
                    y=model(x)
                    outputs.append(np.maximum(y.cpu().numpy()*std+mean,0) if legacy else torch.clamp(y*std+mean,min=0).cpu().numpy())
            return np.concatenate(outputs)
        if legacy:
            groups=[np.flatnonzero(split==s) for s in ['validation','test']]
            batches=[ids[j:j+cfg['batch_size']] for ids in groups for j in range(0,len(ids),cfg['batch_size'])]
        else:batches=np.array_split(np.arange(len(origins)),int(np.ceil(len(origins)/cfg['batch_size'])))
        candidate=predict(panel.filled[history],batches)
        test=split=='test'
        changes=reporting_change(candidate[test],reference[test],panel.raw[target[test]],valid[test],low,np.asarray(panel.dates)[target[test]])
        jan_origins=np.arange(92,124-cfg['H']);jan_history=jan_origins[:,None]+np.arange(-cfg['T'],0)
        jan=predict(np.nan_to_num(full[jan_history],nan=0),[np.arange(j,min(j+cfg['batch_size'],len(jan_origins))) for j in range(0,len(jan_origins),cfg['batch_size'])])
        for key,value in model.state_dict().items():assert torch.equal(value.cpu(),checkpoint['state_dict'][key])
        np.savez_compressed(a.out/(name+'.npz'),december=candidate,january=jan,
            december_origin=np.asarray(panel.dates)[origins],january_origin=dates[jan_origins])
        row=dict(run=name,checkpoint_sha256=record['checkpoint_sha256'],original_prediction_sha256=record['predictions_sha256'],
            december_prediction_sha256=sha256(candidate.tobytes()).hexdigest(),january_prediction_sha256=sha256(jan.tobytes()).hexdigest(),
            validation_strict_allclose_1e_5=bool(np.allclose(candidate[~test],reference[~test],rtol=1e-5,atol=1e-5)),
            changes=changes,parameters_unchanged=True,seconds=time.monotonic()-began)
        rows.append(row)
        (a.out/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(cases),current=name)))
        print(json.dumps(dict(completed=len(rows),total=len(cases),run=name,strict=changes['strict_allclose_1e_5'])),flush=True)
    report=dict(status='diagnostic_only',rows=rows,source_sha256={p.name:digest(p) for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/common/deterministic_group.py')]},
        cases_sha256=digest(a.cases),january_panel_sha256=digest(a.january_panel),
        scope='Fixed checkpoint and graph; deterministic FP64 group accumulation; saved predictions and model selection unchanged.')
    (a.out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    p=ArgumentParser()
    for k in ['source','root','bundle','january-panel','january-manifest','cases','out']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
