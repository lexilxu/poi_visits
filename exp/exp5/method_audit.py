"""Gradient x Input and reference-sensitive integrated gradients for GNN-static.

Attribution uses eight prespecified test origins, all seven leads, and all POIs
for Gradient x Input. IG uses a fixed uniform sample of 1,024 POIs, shared across
seeds and methods. Legal observed POI records define two reference endpoints.
"""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from exp.common.checkpoints import load_gru, encoded_forecast, selected_directories


GROUPS=('history','location','identity','parent','others')
SEMANTICS=('category','subcategory','naics2','naics4')


def group_indices(t):
    return [np.arange(t),t+np.array([0,1]),t+np.r_[4:36,44:52],
            t+np.arange(36,44),t+np.array([2,3])]


def semantic_indices(t):
    return [t+np.arange(4,12),t+np.arange(44,52),t+np.arange(12,20),t+np.arange(20,28)]


def shares_and_regimes(absolute,t):
    totals=np.stack([absolute[:,indices].sum(1) for indices in group_indices(t)],axis=1)
    shares=np.divide(totals,totals.sum(1,keepdims=True),out=np.zeros_like(totals),where=totals.sum(1,keepdims=True)>0)
    standardized=(shares-shares.mean(0))/np.maximum(shares.std(0),1e-12)
    return shares,standardized.argmax(1)


def reference_nodes(raw,attrs,train_end):
    mean=np.nanmean(raw[:train_end],axis=0)
    geographic=attrs[['latitude','longitude']].to_numpy(float)
    geographic=(geographic-np.median(geographic,0))/np.maximum(np.std(geographic,0),1e-8)
    chosen=[]
    for quantile in (.25,.75):
        value=np.quantile(mean[mean>0],quantile)
        score=np.abs(np.log1p(mean)-np.log1p(value))
        candidates=np.argsort(score,kind='stable')[:max(10,int(.01*len(mean)))]
        selected=candidates[np.argmin(np.square(geographic[candidates]).sum(1))]
        chosen.append(int(selected))
    if chosen[0]==chosen[1]:
        raise ValueError('Reference endpoints must be distinct')
    return chosen


def gradient_attribution(model,points,t,mean,std,reference=None,steps=32):
    """Return signed per-lead attribution and endpoint completeness residuals."""
    import torch
    def forward(value):
        return encoded_forecast(model,value[:,:t],value[:,t:],mean,std)
    if reference is None:
        value=points.detach().requires_grad_(True)
        forecast=forward(value)
        values=[]
        for lead in range(forecast.shape[1]):
            derivative=torch.autograd.grad(forecast[:,lead].sum(),value,retain_graph=lead+1<forecast.shape[1])[0]
            values.append(value.detach()*derivative)
        return torch.stack(values,dim=1),None
    difference=points-reference
    integral=torch.zeros((len(points),model.decode.out_features,points.shape[1]),device=points.device)
    # Midpoint quadrature avoids evaluating the ReLU derivative at an endpoint.
    for step in range(steps):
        value=(reference+(step+.5)/steps*difference).detach().requires_grad_(True)
        forecast=forward(value)
        for lead in range(forecast.shape[1]):
            derivative=torch.autograd.grad(forecast[:,lead].sum(),value,retain_graph=lead+1<forecast.shape[1])[0]
            integral[:,lead]+=derivative/steps
    attribution=integral*difference[:,None,:]
    with torch.no_grad():
        delta=forward(points)-forward(reference.expand_as(points))
        residual=attribution.sum(-1)-delta
    return attribution,(residual,delta)


def accumulate(model,history,static,node_ids,origin_ids,mean,std,device,reference=None,steps=32,batch=512):
    import torch
    t=history.shape[1]
    pairs=np.stack(np.meshgrid(origin_ids,node_ids,indexing='ij'),axis=-1).reshape(-1,2)
    absolute=np.zeros((len(static),t+52),dtype=np.float64)
    residuals=[]; scales=[]
    encoded=model.encode_static().detach()
    # cuDNN's inference-mode recurrent kernel does not support backward.
    with torch.backends.cudnn.flags(enabled=False):
        for start in range(0,len(pairs),batch):
            origins,nodes=pairs[start:start+batch].T
            h=torch.as_tensor((history[origins,:,nodes]-mean)/std,device=device,dtype=torch.float32)
            points=torch.cat([h,encoded[nodes]],dim=1).detach()
            baseline=None if reference is None else reference.expand_as(points)
            values,check=gradient_attribution(model,points,t,mean,std,baseline,steps)
            summed=values.abs().sum(1).detach().cpu().numpy()
            np.add.at(absolute,nodes,summed)
            if check is not None:
                residuals.append(check[0].detach().cpu().numpy().ravel())
                scales.append(check[1].detach().cpu().numpy().ravel())
    info={}
    if residuals:
        residual=np.abs(np.concatenate(residuals)); scale=np.abs(np.concatenate(scales))
        normalized=residual/(1+scale)
        info={'completeness_mean_absolute':float(residual.mean()),
              'completeness_p95_scaled':float(np.quantile(normalized,.95)),
              'completeness_max_scaled':float(normalized.max()),'quadrature_steps':steps}
    return absolute,info


def compare(a,b,t):
    shares_a,regime_a=shares_and_regimes(a,t)
    shares_b,regime_b=shares_and_regimes(b,t)
    result={'regime_agreement':float((regime_a==regime_b).mean())}
    for index,name in enumerate(GROUPS):
        correlation=spearmanr(shares_a[:,index],shares_b[:,index]).statistic
        result[name+'_rank_spearman']=float(correlation) if np.isfinite(correlation) else None
        result[name+'_share_mean_A']=float(shares_a[:,index].mean())
        result[name+'_share_mean_B']=float(shares_b[:,index].mean())
    return result


def run(directory,bundle,out,device,steps=32,sample_size=1024,extended_panel=None):
    import torch
    loaded=load_gru(directory,bundle,device)
    if extended_panel is not None:
        with np.load(extended_panel,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['visits'][:92],loaded['raw'])
            np.testing.assert_array_equal(z['poi_id'],loaded['attrs'].placekey.to_numpy(dtype=str))
            loaded['raw']=z['visits'];loaded['dates']=z['dates']
        origins=np.arange(92,len(loaded['raw'])-loaded['fit']['H']+1)
        history=origins[:,None]+np.arange(-loaded['fit']['T'],0)
        targets=origins[:,None]+np.arange(loaded['fit']['H'])
        loaded.update(origins=origins,history=history,target=targets,
                       split=np.repeat('test',len(origins)),
                       own=np.nan_to_num(loaded['raw'][history],nan=0.),
                       valid=np.isfinite(loaded['raw'][targets]))
    fit=loaded['fit']
    assert fit['model']=='GNN-static' and fit['variant']=='full'
    model=loaded['model']; t=fit['T']; n=len(loaded['attrs'])
    test=np.flatnonzero(loaded['split']=='test')
    origins=test[np.unique(np.linspace(0,len(test)-1,8,dtype=int))]
    if not loaded['valid'][origins].all():
        raise ValueError('This attribution sampling protocol requires observed outcomes at sampled events')
    rng=np.random.default_rng(20261114)
    sample=np.sort(rng.choice(n,min(sample_size,n),replace=False))
    out.mkdir(parents=True,exist_ok=False)
    private=out/'private_attribution'; private.mkdir()
    gx,info=accumulate(model,loaded['own'],model.static,np.arange(n),origins,
                       loaded['mean'],loaded['std'],device)
    encoded=model.encode_static().detach()
    reference_ids=reference_nodes(loaded['raw'],loaded['attrs'],fit['train_end'])
    results={}; checks=[]
    for index,j in enumerate(reference_ids):
        visits=np.nan_to_num(loaded['raw'][fit['train_end']-t:fit['train_end'],j],nan=0.)
        h=torch.as_tensor((visits-loaded['mean'])/loaded['std'],device=device,dtype=torch.float32)
        reference=torch.cat([h,encoded[j]]).detach()[None,:]
        ig,check=accumulate(model,loaded['own'],model.static,sample,origins,
                            loaded['mean'],loaded['std'],device,reference,steps)
        # Refine numerical integration, without choosing a result using test labels.
        if check['completeness_p95_scaled']>.01:
            ig,check=accumulate(model,loaded['own'],model.static,sample,origins,
                                loaded['mean'],loaded['std'],device,reference,steps*4)
        if check['completeness_p95_scaled']>.02:
            raise RuntimeError('IG numerical completeness insufficient: '+json.dumps(check))
        results[f'IG_reference_{index+1}']=ig
        checks.append({'reference_number':index+1,**check})
    rows=[]; method_profiles={'Gradient x Input':gx,**results}
    for name,values in method_profiles.items():
        selected=np.arange(n) if name=='Gradient x Input' else sample
        shares,regime=shares_and_regimes(values[selected],t)
        semantic=np.stack([values[selected][:,indexes].sum(1) for indexes in semantic_indices(t)],axis=1)
        semantic=np.divide(semantic,semantic.sum(1,keepdims=True),out=np.zeros_like(semantic),where=semantic.sum(1,keepdims=True)>0)
        for index,group in enumerate(GROUPS):
            rows.append(dict(seed=fit['seed'],method=name,group=group,POIs=len(selected),
                             mean_share=float(shares[:,index].mean()),median_share=float(np.median(shares[:,index])),
                             regime_fraction=float((regime==index).mean())))
        for index,group in enumerate(SEMANTICS):
            rows.append(dict(seed=fit['seed'],method=name,group=group,POIs=len(selected),
                             mean_share=float(semantic[:,index].mean()),median_share=float(np.median(semantic[:,index])),
                             regime_fraction=None))
        np.savez_compressed(private/(name.replace(' ','_')+'.npz'),poi_index=selected,
                            absolute=values[selected],shares=shares,regime=regime,
                            semantic_shares=semantic,poi_id=loaded['attrs'].placekey.to_numpy(dtype=str)[selected])
    comparisons=[]
    for name,values in results.items():
        comparisons.append(dict(seed=fit['seed'],method_A='Gradient x Input',method_B=name,
                                **compare(gx[sample],values[sample],t)))
    a,b=results.values()
    comparisons.append(dict(seed=fit['seed'],method_A='IG_reference_1',method_B='IG_reference_2',
                            **compare(a[sample],b[sample],t)))
    pd.DataFrame(rows).to_csv(out/'attribution_profiles.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'attribution_agreement.csv',index=False)
    (out/'attribution_protocol.json').write_text(json.dumps({
        'status':'complete','seed':fit['seed'],'T':t,'H':fit['H'],'all_pois':n,'IG_sample_size':len(sample),
        'period':'2021-01' if extended_panel else '2020-12',
        'checkpoint_sha256':sha256((directory/'checkpoint.pt').read_bytes()).hexdigest(),
        'analysis_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'IG_sample_sha256':sha256(sample.astype('<i4').tobytes()).hexdigest(),
        'origins':loaded['dates'][loaded['origins'][origins]].tolist(),
        'reference_rule':'Observed POIs nearest the 25th/75th positive training-mean quantiles, then nearest geographic median among the closest 1 percent; their final training input windows and complete static records define endpoints.',
        'reference_training_means':[float(np.nanmean(loaded['raw'][:fit['train_end'],j])) for j in reference_ids],
        'categorical_path':'Straight line between learned embeddings of observed categories; intermediate embeddings are not interpreted as discrete categories.',
        'numeric_inputs':'Continuous static features standardized on training-eligible POIs; visits standardized using first 50 days.',
        'scope':'Absolute attributions summed separately over leads and sampled origins; method comparisons use identical sampled POIs and events.',
        'completeness':checks
    },indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True); parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=32); parser.add_argument('--sample-size',type=int,default=1024)
    parser.add_argument('--watch',action='store_true');parser.add_argument('--max-hours',type=float,default=12)
    parser.add_argument('--extended-panel',type=Path)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.2)
    start=time.monotonic()
    while True:
        directories=selected_directories(args.queue,'GNN-static','Identity','full',14)
        for directory in directories:
            seed=json.loads((directory/'fit.json').read_text())['seed']; destination=args.out/f'seed{seed}'
            if (destination/'attribution_protocol.json').exists():continue
            run(directory,args.bundle,destination,args.device,args.steps,args.sample_size,args.extended_panel)
            print(json.dumps({'seed':seed,'status':'complete'}),flush=True)
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
        if len(directories)==5 or not args.watch or time.monotonic()-start>args.max_hours*3600:break
        time.sleep(10)
