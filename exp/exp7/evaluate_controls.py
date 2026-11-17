"""Frozen-parameter January validation, with original validation thresholds.

Replays every checkpoint against its archived November/December predictions
before producing January forecasts. No fitting uses January outcomes.
"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import time
import numpy as np
from exp.common.audit import load_panel, EXPECTED_BUNDLE_SHA256
from exp.common.protocol import Protocol, feature_mask
from exp.exp1.baselines import predictions
from exp.exp1.extended_baselines import tsb_states
from exp.common.summarize import selected_jobs, evaluate_arrays, merge, write_json
from exp.common.evaluate import load_events


def extended_panel(bundle,path):
    raw,dates,attrs=load_panel(bundle)
    with np.load(path,allow_pickle=False) as z:
        extended=z['visits']; extended_dates=z['dates']; ids=z['poi_id']
    np.testing.assert_array_equal(ids,attrs.placekey.to_numpy(dtype=str))
    np.testing.assert_array_equal(extended[:len(raw)],raw)
    np.testing.assert_array_equal(extended_dates[:len(raw)],dates)
    assert extended.shape==(123,9884)
    np.testing.assert_array_equal(extended_dates[92:],
        np.arange('2021-01-01','2021-02-01',dtype='datetime64[D]').astype(str))
    return extended,extended_dates,attrs


def replay_model(directory,raw,dates,attrs,device,graph_dir,diagnostic_out=None):
    import torch
    import pandas as pd
    from scipy import sparse
    from exp.common.controlled_training import make_model, model_static_features
    from exp.common.original_protocol import static_features, group_average
    from exp.exp1.train_mtgnn import sparse_operator, clique_operator
    from exp.common.controls import propagate_neighbors
    fit=json.loads((directory/'fit.json').read_text(encoding='utf8'))
    assert fit['bundle_sha256']==EXPECTED_BUNDLE_SHA256
    saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    args=saved['args']; t,h=fit['T'],fit['H']
    if args.get('attributes_file'):
        from exp.common.inputs import replace_attributes
        attrs,digest=replace_attributes(attrs,args['attributes_file'])
        assert digest==fit['metadata_sha256']
    origins=np.r_[np.arange(50,64-h+1),np.arange(64,92-h+1),np.arange(92,len(raw)-h+1)]
    history=origins[:,None]+np.arange(-t,0)
    own=np.nan_to_num(raw[history],nan=0.).astype('float32'); graph=own
    operators=(lambda x:x,lambda x:x)
    if args.get('sparse_graph'):
        path=Path(args['sparse_graph']);assert sha256(path.read_bytes()).hexdigest()==fit['graph_sha256']
        adjacency=sparse.load_npz(path)
        transition=adjacency+sparse.eye(len(attrs),dtype=np.float32,format='csr')
        transition=sparse.diags(1/np.asarray(transition.sum(1)).ravel())@transition
        graph=(transition@np.nan_to_num(raw,nan=0.).T).T[history].astype('float32')
        operators=(sparse_operator(adjacency,torch.device(device)),sparse_operator(adjacency.T,torch.device(device)))
    elif args.get('graph_file'):
        path=Path(args['graph_file'])
        if not path.exists():path=graph_dir/path.name
        with np.load(path,allow_pickle=False) as z:
            peers=z['neighbors']; np.testing.assert_array_equal(z['poi_id'],attrs.placekey.to_numpy(dtype=str))
        assert sha256(peers.astype('<i4').tobytes()).hexdigest()==fit['graph_sha256']
        graph=propagate_neighbors(own,peers).astype('float32')
        degree=(peers>=0).sum(1); row=np.repeat(np.arange(len(peers)),degree); col=peers[peers>=0]
        a=sparse.csr_matrix((1/degree[row],(row,col)),shape=(len(peers),len(peers)))
        operators=(sparse_operator(a,torch.device(device)),sparse_operator(a.T,torch.device(device)))
    elif fit['adjacency'] in ('NAICS-2','NAICS-4'):
        digits=2 if fit['adjacency']=='NAICS-2' else 4
        labels=pd.factorize(attrs.naics_code.astype('int64').astype(str).str[:digits])[0]
        filled=np.nan_to_num(raw,nan=0.)
        graph=((filled+group_average(filled,labels))*.5)[history].astype('float32')
        op=clique_operator(labels,torch.device(device));operators=(op,op)
    static=None
    if fit['model'] in ('GNN-static','GNN-parent'):
        static,_=model_static_features(attrs,fit['model'])
        static[:,:4]=(static[:,:4]-saved['numeric_mean'])/saved['numeric_std']
    long_input=t>14
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(.4 if long_input else .25)
    model=make_model(fit['model'],h,static,feature_mask(fit['variant']),len(attrs),t,
                     operators,torch.device(device),fit['dropout'])
    model.load_state_dict(saved['state_dict'],strict=True);model=model.to(device).eval()
    # Preserve every fitted model's batch shapes for exact replay.
    # cuDNN's T=28 workspace needs a larger allowance; releasing cached workspaces
    # after each batch avoids retaining both batch-3 and batch-4 allocations.
    evaluation_batch_size=args['batch_size']
    pred=np.empty((len(origins),h,len(attrs)),dtype='float32')
    with torch.no_grad():
        for partition in (np.flatnonzero(origins<92),np.flatnonzero(origins>=92)):
            for ids in np.array_split(partition,int(np.ceil(len(partition)/evaluation_batch_size))):
                inputs=[torch.as_tensor((v[ids]-saved['visit_mean'])/saved['visit_std'],
                                         device=device,dtype=torch.float32) for v in (graph,own)]
                pred[ids]=(model(*inputs)*saved['visit_std']+saved['visit_mean']).clamp_min(0).cpu().numpy()
                if long_input and device.startswith('cuda'):torch.cuda.empty_cache()
    old=load_events(directory/'predictions.npz'); before=origins<92
    np.testing.assert_array_equal(dates[origins[before]],old['forecast_origin'])
    strict_match=np.allclose(pred[before],old['predicted_visits'],rtol=2e-4,atol=2e-4)
    numerical_check={}
    if diagnostic_out is not None or not strict_match:
        repeats=[]
        for _ in range(3):
            again=[]
            with torch.no_grad():
                partition=np.flatnonzero(origins<92)
                for ids in np.array_split(partition,int(np.ceil(len(partition)/evaluation_batch_size))):
                    inputs=[torch.as_tensor((v[ids]-saved['visit_mean'])/saved['visit_std'],
                               device=device,dtype=torch.float32) for v in (graph,own)]
                    again.append((model(*inputs)*saved['visit_std']+saved['visit_mean']).clamp_min(0).cpu().numpy())
                    if long_input and device.startswith('cuda'):torch.cuda.empty_cache()
            repeats.append(np.concatenate(again))
        def differences(a,b):
            delta=np.abs(a-b).astype(float)
            return dict(max=float(delta.max()),mean=float(delta.mean()),
                        p99=float(np.quantile(delta,.99)),
                        exceeding_tolerance=int((delta>2e-4+2e-4*np.abs(b)).sum()))
        numerical_check=dict(against_saved=differences(pred[before],old['predicted_visits']),
                            repeated_forward=[differences(r,pred[before]) for r in repeats])
        if diagnostic_out is not None:
            write_json(diagnostic_out,dict(model=fit['model'],adjacency=fit['adjacency'],seed=fit['seed'],**numerical_check))
    if not strict_match:
        # CUDA index_add_ in the archived clique operator uses atomic FP32 sums.
        # Repeated calls above measure that kernel's variation without refitting.
        # This exception is confined to that operator and bounded well below
        # the reported visit-count precision; all other adapters remain strict.
        if fit['model']!='Fixed-MTGNN' or fit['adjacency'] not in ('NAICS-2','NAICS-4') or fit['T']!=14:
            np.testing.assert_allclose(pred[before],old['predicted_visits'],rtol=2e-4,atol=2e-4)
        delta=np.abs(pred[before]-old['predicted_visits'])
        repeated_mean=max(r['mean'] for r in numerical_check['repeated_forward'])
        repeated_max=max(r['max'] for r in numerical_check['repeated_forward'])
        if not (delta.mean()<1e-5 and delta.max()<.01 and
                np.max(delta/(1+np.abs(old['predicted_visits'])))<.002 and
                delta.mean()<max(1e-6,10*repeated_mean) and delta.max()<max(.002,10*repeated_max)):
            raise AssertionError('Clique replay exceeds measured FP32 aggregation variation: '+json.dumps(numerical_check))
        numerical_check.update(reason='Repeated forward calls expose nondeterministic CUDA FP32 index_add_',
             maximum_mean_absolute_difference=1e-5,maximum_absolute_difference=.01,
             maximum_scaled_difference=.002)
    replay_error=float(np.abs(pred[before]-old['predicted_visits']).max())
    # Reuse the original validation forecasts bit-for-bit for calibration.
    pred[before]=old['predicted_visits']
    keep=(origins<64)|(origins>=92)
    return origins[keep],pred[keep],dict(replay_max_absolute_error=replay_error,
              evaluation_batch_size=evaluation_batch_size,
              strict_replay_passed=bool(strict_match),numerical_replay_check=numerical_check,
              checkpoint_sha256=sha256((directory/'checkpoint.pt').read_bytes()).hexdigest())


def baseline_models(raw,dates,t,folder):
    models,origins,history,target,split,valid,_=predictions(raw,dates,t,7,Protocol(T=t))
    own=np.nan_to_num(raw[history],nan=0.).transpose(0,2,1).astype(float)
    with np.load(folder/f'AR_ridge_T{t}_weights.npz') as z: weights=z['weights']
    models['AR-ridge']=np.maximum(0,np.einsum('bnt,nth->bhn',own,weights[:,:t])+weights[:,t].T[None])
    with np.load(folder/f'nlinear_T{t}_H7_weights.npz') as z:
        weights=z['weights'];scale=float(z['scale'])
    last=own[:,:,-1]
    design=np.concatenate(((own[:,:,:-1]-last[...,None])/scale,np.ones((*last.shape,1))),axis=-1)
    models['NLinear ridge']=np.maximum(0,np.einsum('bnt,th->bhn',design,weights)*scale+last[:,None,:])
    fits=json.loads((folder.parent/'extended_baseline_protocol.json').read_text())['fits']
    fit=next(f for f in fits if f['T']==t and f['model']=='TSB' and 'alpha' in f)
    models['TSB']=np.broadcast_to(tsb_states(raw,fit['alpha'],fit['beta'])[origins,None,:],models['Zero'].shape)
    kept=(split=='validation')|(origins>=92)
    for name,forecast in models.items():
        path=folder/f'{name.replace(" ","_").replace("-","_")}_T{t}_H7.npz'
        if not path.exists():path=folder/f'{name.replace(" ","_")}_T{t}_H7.npz'
        old=load_events(path)
        before=(split!='train')&(origins+7<=92)
        np.testing.assert_array_equal(dates[origins[before]],old['forecast_origin'])
        np.testing.assert_allclose(forecast[before],old['predicted_visits'],rtol=2e-6,atol=2e-6)
        yield name,origins[kept],forecast[kept].astype('float32')


def save_and_evaluate(out,origins,pred,meta,raw,dates,attrs,extra):
    h=meta['H'];target=origins[:,None]+np.arange(h)
    assert np.all((origins<64)|(origins>=92)) and np.sum(origins>=92)==25
    assert np.isfinite(pred).all() and np.min(pred)>=0
    data=dict(model=meta['model'],adjacency=meta['adjacency'],seed=meta['seed'],
              forecast_origin=dates[origins].astype(str),horizon_step=np.arange(1,h+1),
              poi_id=attrs.placekey.to_numpy(dtype=str),observed_visits=raw[target],
              predicted_visits=pred,valid=np.isfinite(raw[target]),
              dataset_split=np.where(origins<64,'validation','test'),train_end=50,T=meta['T'],H=h,
              protocol='frozen_january',bundle_sha256=EXPECTED_BUNDLE_SHA256)
    name=meta['id'].replace(' ','_')+f"_seed{meta['seed']}"
    path=out/'private_predictions'/f'{name}.npz'; path.parent.mkdir(exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**data)
    temporary.replace(path)
    report=evaluate_arrays(data,meta,raw,attrs,meta['stage'] in ('baseline','core','loss','adaptive'))
    report.update(extra,source_prediction_sha256=sha256(path.read_bytes()).hexdigest(),
                  period='2021-01',validation_period='2020-11-20/2020-12-03',frozen_parameters=True)
    write_json(out/'run_records'/f'{name}.json',report)
    row=next(r for r in report['metrics'] if r['cohort']=='All' and r['calibration']=='raw')
    print(json.dumps(dict(id=meta['id'],seed=meta['seed'],January_MAE=row['MAE'])),flush=True)


def main(args):
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.25)
    raw,dates,attrs=extended_panel(args.bundle,args.panel)
    args.out.mkdir(parents=True,exist_ok=True)
    cache=args.out/'run_records';cache.mkdir(exist_ok=True)
    for t in (14,28):
        for name,origins,pred in baseline_models(raw,dates,t,args.baselines):
            meta=dict(id=f'{name}_T{t}',stage='baseline',model=name,T=t,H=7,adjacency='None',
                       variant='full',loss='deterministic',graph=None,k=None,seed=-1,
                       selected_lr=None,validation_MAE=None,best_epoch=None)
            if not (cache/(meta['id'].replace(' ','_')+'_seed-1.json')).exists():
                save_and_evaluate(args.out,origins,pred,meta,raw,dates,attrs,dict(replay_passed=True))
    merge(cache,args.out)
    start=time.monotonic()
    while True:
        changed=False
        for path,meta,_ in selected_jobs(args.queues):
            if meta['stage']=='density':continue
            if (cache/(meta['id'].replace(' ','_')+f"_seed{meta['seed']}.json")).exists():continue
            origins,pred,extra=replay_model(path.parent,raw,dates,attrs,args.device,args.graph_dir)
            save_and_evaluate(args.out,origins,pred,meta,raw,dates,attrs,extra);changed=True
        if changed:merge(cache,args.out)
        states=[json.loads((q/'queue_status.json').read_text()) if (q/'queue_status.json').exists() else {} for q in args.queues]
        if not args.watch or all(s.get('status') in ('complete','stopped') for s in states):break
        if time.monotonic()-start>args.max_hours*3600:break
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
        time.sleep(10)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--queues',nargs='+',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--graph-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--watch',action='store_true')
    p.add_argument('--max-hours',type=float,default=12)
    main(p.parse_args())
