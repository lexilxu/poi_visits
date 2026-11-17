"""Original primary-seed figure inputs from verified corrected predictions."""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,socket,sys
import numpy as np
import pandas as pd
from exp.exp1.summarize import metrics


def main(args):
    sys.path[:0]=[str(args.source),str(args.source/'exp/common'),str(args.source/'exp/common')]
    from exp.common.original_protocol import load_panel, make_windows, model_key, group_codes, group_average
    from exp.common.audit import EXPECTED_BUNDLE_SHA256
    panel=load_panel(args.bundle,EXPECTED_BUNDLE_SHA256);sources=[];horizons=[];recovery=[];choices=[]
    graphs=['Identity','NAICS-2','NAICS-4','Distance KNN','Distance RBF','Visit correlation']
    mtg_graphs=['Identity','NAICS-2','Top category','Distance KNN']
    def legacy(t,h,model,graph):
        folder=args.legacy/f'T{t}_H{h}';path=folder/(model_key(model,graph)+'.npy')
        prediction=np.load(path,allow_pickle=False)
        sources.append(dict(kind='preserved_comparator',path=str(path),sha256=sha256(path.read_bytes()).hexdigest()))
        report=json.loads((folder/(model_key(model,graph)+'.json')).read_text())
        return prediction,report
    def corrected(t,h,graph,w):
        folder=args.root/'runs'/f"original_masked_T{t}_H{h}_{graph.replace(' ','_')}_history_seed{args.seed}"
        record=json.loads((folder/'complete.json').read_text());path=folder/'predictions.npz'
        assert record['status']=='complete' and record['config']['seed']==args.seed
        assert record['config']['adjacency']==graph
        assert sha256(path.read_bytes()).hexdigest()==record['predictions_sha256']
        assert sha256((folder/'best_weights.pt').read_bytes()).hexdigest()==record['checkpoint_sha256']
        with np.load(path,allow_pickle=False) as z:
            test=z['dataset_split']=='test'
            np.testing.assert_array_equal(z['poi_id'],panel.attrs.placekey.to_numpy(dtype=str))
            np.testing.assert_array_equal(z['forecast_origin'][test],np.asarray(panel.dates)[w.origins[w.test_start:]])
            np.testing.assert_array_equal(z['observed_visits'][test],w.y[w.test_start:])
            assert z['valid'][test].all()
            prediction=z['predicted_visits'][test]
        sources.append(dict(kind='corrected_EvolveGCN',path=str(path),sha256=record['predictions_sha256'],checkpoint_sha256=record['checkpoint_sha256']))
        return prediction,record
    cache={}
    for t,h in [(14,7),(28,7),(28,14),(28,28)]:
        w=make_windows(panel.raw,t,h);truth=w.y[w.test_start:]
        ha,_=legacy(t,h,'Historical average','None')
        low=(w.history_mean>0)&(w.history_mean<=np.quantile(w.history_mean[w.history_mean>0],.25))&(w.zero_ratio>=.7)
        current={graph:corrected(t,h,graph,w) for graph in graphs}
        mtg={graph:legacy(t,h,'MTGNN',graph) for graph in mtg_graphs}
        best=min((g for g in graphs if g!='Identity'),key=lambda g:current[g][1]['validation_loss'])
        best_mtg=min(mtg,key=lambda g:mtg[g][1]['validation_loss'])
        choices.append(dict(T=t,H=h,EvolveGCN_adjacency=best,MTGNN_adjacency=best_mtg,selection='validation loss only'))
        baselines=[('Historical average','None',ha)]+[(name,'None',legacy(t,h,name,'None')[0]) for name in ['AR-ridge','LSTM']]
        if t==28:
            curves=[(name,name,graph,pred) for name,graph,pred in baselines]+[
                ('MTGNN (best)','MTGNN',best_mtg,mtg[best_mtg][0]),
                ('EvolveGCN (identity)','EvolveGCN','Identity',current['Identity'][0]),
                ('EvolveGCN (best non-identity)','EvolveGCN',best,current[best][0])]
            for series,model,graph,pred in curves:
                horizons.append(dict(T=t,H=h,series=series,model=model,adjacency=graph,seed=args.seed,
                    test_origins=len(truth),**metrics(truth,pred,ha,low)))
        if (t,h)==(14,7):
            rows=baselines+ [('MTGNN',g,mtg[g][0]) for g in mtg_graphs]+[
                ('EvolveGCN','Top category' if g=='NAICS-4' else g,current[g][0]) for g in graphs]
            for model,graph,pred in rows:
                recovery.append(dict(model=model,adjacency=graph,seed=args.seed,**metrics(truth,pred,ha,low)))
        if (t,h)==(28,14):cache=dict(w=w,low=low,truth=truth,identity=current['Identity'][0],graph=current['NAICS-2'][0])
    w,low=cache['w'],cache['low'];y=cache['truth'][:,:,low]
    gain=np.abs(y-cache['identity'][:,:,low])-np.abs(y-cache['graph'][:,:,low])
    history=panel.filled[w.history_idx[w.test_start:]].mean(1)
    labels=group_codes(panel.attrs)['NAICS-2'];size=np.bincount(labels)[labels]
    neighbors=(group_average(history,labels)*size-history)/np.maximum(size-1,1)
    target=np.broadcast_to(history[:,None,low],y.shape);neighbor=np.broadcast_to(neighbors[:,None,low],y.shape)
    edges=[np.unique(np.quantile(v,np.linspace(0,1,7))) for v in [target,neighbor]]
    bx,by=[np.searchsorted(e[1:-1],v,side='right') for v,e in zip([target,neighbor],edges)]
    cells=[];outcomes=[]
    for outcome,status in [('Positive observed visits',y>0),('Zero observed visits',y==0)]:
        for j,(loy,hiy) in enumerate(zip(edges[1][:-1],edges[1][1:])):
            for i,(lox,hix) in enumerate(zip(edges[0][:-1],edges[0][1:])):
                keep=status&(bx==i)&(by==j);n=int(keep.sum())
                cells.append(dict(outcome=outcome,x_bin=i,y_bin=j,target_low=lox,target_high=hix,
                    neighbor_low=loy,neighbor_high=hiy,n_events=n,display_cell=n>=30,
                    mean_mae_gain=float(gain[keep].mean()) if n else None))
        selected=[row for row in cells if row['outcome']==outcome]
        assert sum(row['n_events'] for row in selected)==int(status.sum())
        weighted=sum(row['n_events']*row['mean_mae_gain'] for row in selected if row['n_events'])/int(status.sum())
        np.testing.assert_allclose(weighted,float(gain[status].mean()),rtol=1e-5,atol=1e-6)
        outcomes.append(dict(outcome=outcome,n_events=int(status.sum()),mean_mae_gain=float(gain[status].mean())))
    assert sum(c['n_events'] for c in cells)==y.size
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(horizons).to_csv(args.out/'07_horizon_skill.csv',index=False)
    pd.DataFrame(recovery).to_csv(args.out/'01_recovery_activation.csv',index=False)
    pd.DataFrame(cells).to_csv(args.out/'02_graph_transfer_cells.csv',index=False)
    report=dict(status='complete_primary_figure_inputs',seed=args.seed,sources=sources,selection=choices,
        outcome_summary=outcomes,target_edges=edges[0].tolist(),neighbor_edges=edges[1].tolist(),
        conventions='Original test-input sextile boundaries pooled over both outcomes; cells with fewer than30events have no displayed estimate. Top category reuses the identical NAICS-4 fit.',
        series=6,horizon_points=len(horizons),recovery_points=len(recovery),outcome_cells=len(cells),
        code_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/exp1/summarize.py')]})
    (args.out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','seed','horizon_points','recovery_points','outcome_cells','outcome_summary']}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for field in ['source','root','legacy','bundle','out']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);main(p.parse_args())
