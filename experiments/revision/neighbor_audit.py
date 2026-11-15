"""Matched-size interventions on fitted semantic GraphGRU models.

All target POIs, forecast events, and fitted parameters are held fixed across
interventions. Selection uses training data or the available input window;
regime cutoffs use validation inputs, never test outcomes.
"""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from audit_protocol import low_history
from checkpoint_tools import load_gru, predict_gru, selected_directories


def neighbor_sets(training,labels,targets,k,seed,degree_caps=None):
    filled=np.nan_to_num(training,nan=0.)
    count=np.isfinite(training).sum(0)
    mean=np.divide(np.nansum(training,0),count,out=np.zeros(len(labels)),where=count>0)
    frequency=np.divide((training>0).sum(0),count,out=np.zeros(len(labels)),where=count>0)
    rng=np.random.default_rng(seed)
    sets={name:np.full((len(targets),k),-1,dtype=int) for name in ('active','correlated','random')}
    degrees=[]
    for row,j in enumerate(targets):
        peers=np.flatnonzero((labels==labels[j]) & (np.arange(len(labels))!=j))
        size=min(k,len(peers),int(degree_caps[row]) if degree_caps is not None else k); degrees.append(size)
        if not size:
            continue
        sets['active'][row,:size]=peers[np.argsort(-mean[peers],kind='stable')[:size]]
        paired=np.isfinite(training[:,j,None]) & np.isfinite(training[:,peers])
        counts=paired.sum(0)
        x=np.where(paired,training[:,j,None],0.)
        z=np.where(paired,training[:,peers],0.)
        xm=np.divide(x.sum(0),counts,out=np.zeros(len(peers)),where=counts>0)
        zm=np.divide(z.sum(0),counts,out=np.zeros(len(peers)),where=counts>0)
        x=np.where(paired,x-xm,0.);z=np.where(paired,z-zm,0.)
        correlation=(x*z).sum(0)/np.maximum(np.linalg.norm(x,axis=0)*np.linalg.norm(z,axis=0),1e-8)
        correlation[counts<3]=0.
        sets['correlated'][row,:size]=peers[np.argsort(-correlation,kind='stable')[:size]]
        sets['random'][row,:size]=rng.choice(peers,size,replace=False)
    # Match donor histories on training mean, frequency, and variability.
    variability=np.nanstd(training,axis=0)
    features=np.column_stack([np.log1p(mean),frequency,np.log1p(variability)])
    features=(features-features.mean(0))/np.maximum(features.std(0),1e-8)
    tree=cKDTree(features)
    donors=np.full_like(sets['active'],-1)
    distances=[]
    cache={}
    for row,j in enumerate(targets):
        for slot,peer in enumerate(sets['active'][row]):
            if peer<0:
                continue
            key=(int(labels[j]),int(peer))
            if key not in cache:
                for size in (32,128,512,len(labels)):
                    distance,candidates=tree.query(features[peer],k=min(size,len(labels)))
                    allowed=np.flatnonzero(labels[np.atleast_1d(candidates)]!=labels[j])
                    if len(allowed):
                        index=allowed[0]
                        cache[key]=(int(np.atleast_1d(candidates)[index]),float(np.atleast_1d(distance)[index]))
                        break
                if key not in cache:
                    raise ValueError('No disconnected donor exists')
            donors[row,slot],distance=cache[key]
            distances.append(distance)
    return sets,donors,np.asarray(degrees),dict(
        donor_standardized_distance_mean=float(np.mean(distances)),
        donor_standardized_distance_p95=float(np.quantile(distances,.95)))


def clique_change(own,labels,targets,peers,donors=None):
    """Exact change to 0.5 own + 0.5 group mean, with graph weights fixed.

    own is (origins,T,nodes); peers can be (targets,k) or (origins,targets,k).
    The result applies only to the selected target outputs, not a refitted graph.
    """
    if peers.ndim==2:
        peers=np.broadcast_to(peers,(len(own),*peers.shape))
    result=np.zeros((len(own),own.shape[1],len(targets)),dtype=np.float32)
    group_size=np.bincount(labels)[labels[targets]]
    for b in range(len(own)):
        for slot in range(peers.shape[-1]):
            indexes=peers[b,:,slot]
            present=indexes>=0
            result[b,:,present]-=own[b,:,indexes[present]]
            if donors is not None:
                replace=donors[:,slot] if donors.ndim==2 else donors[b,:,slot]
                result[b,:,present]+=own[b,:,replace[present]]
    return result*(.5/group_size)[None,None,:]


def regimes(y,p,identity,target_evidence,neighbor_evidence,thresholds,lift=.1,activation=.5):
    low=(target_evidence<=thresholds[0])[:,None,:]
    high=(neighbor_evidence>thresholds[1])[:,None,:]
    up=p-identity>=lift
    substitutes=low & high & up
    spurious=substitutes & (y==0) & (p>=activation)
    conditions=[spurious,substitutes,~low & high & up,~low & ~up,(y==0)&(p<activation)]
    names=['Spurious transfer','Substitutive transfer','Confirmatory transfer',
           'Redundant self-history','Conservative zero','Other']
    output=np.full(y.shape,len(names)-1,dtype=np.int8)
    for index,condition in enumerate(conditions):
        output[(output==len(names)-1)&condition]=index
    # Pre-outcome evidence strata are separate from these outcome descriptions.
    strata=np.broadcast_to((~low).astype('int8')*2+high.astype('int8'),y.shape).copy()
    return output,names,strata


def summarize(y,before,after,valid,selection,degrees):
    keep=valid & selection
    if not keep.any():
        return dict(events=0)
    observed,original,changed=y[keep],before[keep],after[keep]
    positive=observed>0; zero=~positive
    size=np.broadcast_to(degrees[None,None,:],y.shape)[keep]
    originally_active=original>=.5
    def conditional(value,mask):
        return float(value[mask].mean()) if mask.any() else None
    return dict(events=int(keep.sum()),forecast_reduction=float((original-changed).mean()),
                forecast_reduction_per_node=float(((original-changed)/size).mean()),
                MAE_change=float((np.abs(changed-observed)-np.abs(original-observed)).mean()),
                positive_forecast_mass_lost=(float((original[positive]-changed[positive]).sum()/observed[positive].sum())
                                             if positive.any() else None),
                false_activations_removed=conditional((original>=.5).astype(float)-(changed>=.5),zero),
                positive_activations_lost=conditional((original>=.5).astype(float)-(changed>=.5),positive),
                fraction_false_activations_removed=conditional(changed<.5,zero & originally_active),
                fraction_positive_activations_lost=conditional(changed<.5,positive & originally_active),
                original_FAR=conditional(originally_active,zero),
                original_recall=conditional(originally_active,positive),
                max_absolute_forecast_change=float(np.max(np.abs(original-changed))))


def run(directory,identity_directory,bundle,out,device,k=10,extended_panel=None):
    loaded=load_gru(directory,bundle,device)
    identity=load_gru(identity_directory,bundle,device)
    if extended_panel is not None:
        from train_formal import group_average
        with np.load(extended_panel,allow_pickle=False) as z:
            raw=z['visits'];dates=z['dates']
            np.testing.assert_array_equal(raw[:92],loaded['raw'])
            np.testing.assert_array_equal(z['poi_id'],loaded['attrs'].placekey.to_numpy(dtype=str))
        for fitted in (loaded,identity):
            t,h=fitted['fit']['T'],fitted['fit']['H']
            origins=np.r_[np.arange(50,64-h+1),np.arange(92,len(raw)-h+1)]
            history=origins[:,None]+np.arange(-t,0);target=origins[:,None]+np.arange(h)
            own=np.nan_to_num(raw[history],nan=0.)
            if fitted['labels'] is None:graph=own
            else:
                filled=np.nan_to_num(raw,nan=0.)
                graph=((filled+group_average(filled,fitted['labels']))*.5)[history]
            pred=predict_gru(fitted['model'],graph,own,fitted['mean'],fitted['std'],device)
            np.testing.assert_allclose(pred[origins<64],fitted['prediction'][fitted['split']=='validation'],rtol=2e-5,atol=2e-5)
            fitted.update(raw=raw,dates=dates,origins=origins,history=history,target=target,own=own,graph=graph,
                           valid=np.isfinite(raw[target]),split=np.where(origins<64,'validation','test'),prediction=pred)
    fit=loaded['fit']
    assert fit['seed']==identity['fit']['seed'] and fit['T']==identity['fit']['T']
    labels=loaded['labels']
    if labels is None:
        raise ValueError('Non-identity semantic graph required')
    cohort,_=low_history(loaded['raw'][:fit['train_end']])
    # Isolates cannot be perturbed and are counted explicitly in the manifest.
    fine=pd.factorize(loaded['attrs'].naics_code.astype('int64').astype(str).str[:4])[0]
    fine_degree=np.bincount(fine)[fine]-1
    targets=np.flatnonzero(cohort & (fine_degree>0))
    own=loaded['own']; test=loaded['split']=='test'; val=loaded['split']=='validation'
    sets,donors,degrees,matching=neighbor_sets(loaded['raw'][:fit['train_end']],labels,targets,k,fit['seed'],fine_degree[targets])
    recent=np.full((len(own),len(targets),k),-1,dtype=int)
    for b in range(len(own)):
        score=own[b,-7:].mean(0)
        for row,j in enumerate(targets):
            peers=np.flatnonzero((labels==labels[j]) & (np.arange(len(labels))!=j))
            recent[b,row,:degrees[row]]=peers[np.argsort(-score[peers],kind='stable')[:degrees[row]]]
    target_evidence=own[:,:,targets].mean(1)
    group_size=np.bincount(labels)[labels[targets]]
    # graph = .5 own + .5 group mean, including the target in the latter term.
    group_mean=(2*loaded['graph'][:,:,targets]-own[:,:,targets]).mean(1)
    neighbor_evidence=(group_mean*group_size-own[:,:,targets].mean(1))/(group_size-1)
    thresholds=(float(np.median(target_evidence[val])),float(np.median(neighbor_evidence[val])))
    y=loaded['raw'][loaded['target'][test]][:,:,targets]
    before=loaded['prediction'][test][:,:,targets]
    mask=loaded['valid'][test][:,:,targets]
    reference=identity['prediction'][test][:,:,targets]
    assigned,names,strata=regimes(y,before,reference,target_evidence[test],neighbor_evidence[test],thresholds)
    base=dict(seed=fit['seed'],adjacency=fit['adjacency'],T=fit['T'],H=fit['H'])
    transfer=[]
    for grouping,values,labels_out in [('regime',assigned,names),('evidence',strata,
            ['Low target / low neighbor','Low target / high neighbor','High target / low neighbor','High target / high neighbor'])]:
        for index,label in enumerate(labels_out):
            selected=mask & (values==index)
            if not selected.any():
                continue
            transfer.append({**base,'grouping':grouping,'stratum':label,'events':int(selected.sum()),
                             'graph_lift':float((before-reference)[selected].mean()),
                             'MAE_gain':float((np.abs(reference-y)-np.abs(before-y))[selected].mean()),
                             'positive_prevalence':float((y[selected]>0).mean())})
    changes={name:clique_change(own[test],labels,targets,peers) for name,peers in sets.items()}
    changes['recent_active']=clique_change(own[test],labels,targets,recent[test])
    changes['activity_matched_replacement']=clique_change(own[test],labels,targets,sets['active'],donors)
    # An actual full-input intervention on a disconnected group checks isolation.
    control_own=own[test].copy()
    control_group=int(labels[targets[0]])
    disconnected=np.flatnonzero(labels!=control_group)[:k]
    control_own[:,:,disconnected]=0.
    from train_formal import group_average
    shape=control_own.shape
    control_graph=.5*(control_own+group_average(control_own.reshape(-1,len(labels)),labels).reshape(shape))
    control_pred=predict_gru(loaded['model'],control_graph,control_own,loaded['mean'],loaded['std'],device)
    control_targets=targets[labels[targets]==control_group]
    negative_error=float(np.max(np.abs(control_pred[:,:,control_targets]-loaded['prediction'][test][:,:,control_targets])))
    if negative_error>2e-5:
        raise AssertionError(f'Disconnected-node control failed: {negative_error}')
    rows=[];saved_changes={}
    attempted=(target_evidence[test]<=thresholds[0])[:,None,:] & (neighbor_evidence[test]>thresholds[1])[:,None,:] & (before-reference>=.1)
    beneficial=attempted & (y>0) & (np.abs(before-y)<np.abs(reference-y))
    harmful=attempted & (y==0) & (before>=.5)
    recovered=beneficial & (reference<.5) & (before>=.5)
    for name,change in changes.items():
        # This architecture applies independent GRUs after the fixed aggregation;
        # selecting targets is algebraically identical to the full-node forward.
        after=predict_gru(loaded['model'],loaded['graph'][test][:,:,targets]+change,
                          own[test][:,:,targets],loaded['mean'],loaded['std'],device)
        saved_changes[name]=after
        for grouping,values,labels_out in [('all',np.zeros_like(assigned),['All']),
                ('regime',assigned,names),('evidence',strata,
                ['Low target / low neighbor','Low target / high neighbor','High target / low neighbor','High target / high neighbor'])]:
            for index,label in enumerate(labels_out):
                rows.append({**base,'intervention':name,'grouping':grouping,'stratum':label,
                             **summarize(y,before,after,mask,values==index,degrees)})
        for label,selected in [('Beneficial',beneficial),('Harmful',harmful),('Recovered positives',recovered)]:
            rows.append({**base,'intervention':name,'grouping':'attempted_substitution','stratum':label,
                         **summarize(y,before,after,mask,selected,degrees)})
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'private_counterfactuals.npz',observed_visits=y,graph_prediction=before,
                        identity_prediction=reference,valid=mask,target_index=targets,degrees=degrees,
                        forecast_origin=np.asarray(loaded['dates'][loaded['origins'][test]],dtype=str),horizon_step=np.arange(1,fit['H']+1),
                        regimes=assigned,evidence_strata=strata,**saved_changes)
    pd.DataFrame(rows).to_csv(out/'neighbor_interventions.csv',index=False)
    pd.DataFrame(transfer).to_csv(out/'graph_transfer.csv',index=False)
    (out/'neighbor_protocol.json').write_text(json.dumps({
        **base,'status':'complete','selected_targets':len(targets),'low_history_pois':int(cohort.sum()),
        'period':'2021-01' if extended_panel else '2020-12',
        'size_min':int(degrees.min()),'size_max':int(degrees.max()),'mean_size':float(degrees.mean()),
        'evidence_medians_validation_only':thresholds,'graph_lift_threshold':.1,'activation_threshold':.5,
        'negative_control_max_absolute_error':negative_error,
        'negative_control_targets':len(control_targets),**matching,
        'scope':'Sensitivity of fixed GraphGRU predictions; graph-versus-identity contrasts use separate fitted models.',
        'donor_rule':'Nearest external-group history in train-standardized log mean, nonzero frequency, and log standard deviation.',
        'missing_observations':'Correlations use pairwise observed training days; means, frequencies, and variability use observed training counts.',
        'cohort':'Uniformly weighted first 50 calendar days; NAICS-4 non-isolates form the identical target/event set for both graphs.',
        'size_rule':'Each target has min(10, NAICS-4 peer count) intervened neighbors for every selection rule and both graphs.',
        'target_sha256':sha256(targets.astype('<i4').tobytes()).hexdigest(),
        'prediction_sha256':sha256((Path(directory)/'predictions.npz').read_bytes()).hexdigest(),
        'analysis_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__),Path(__file__).with_name('checkpoint_tools.py')]}
    },indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True); parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--extended-panel',type=Path)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.2)
    identities=selected_directories(args.queue,'GraphGRU','Identity',t=14)
    by_seed={json.loads((p/'fit.json').read_text())['seed']:p for p in identities}
    for adjacency in ('NAICS-2','NAICS-4'):
        for path in selected_directories(args.queue,'GraphGRU',adjacency,t=14):
            seed=json.loads((path/'fit.json').read_text())['seed']
            destination=args.out/f'{adjacency}_seed{seed}'
            if (destination/'neighbor_protocol.json').exists():
                continue
            run(path,by_seed[seed],args.bundle,destination,args.device,extended_panel=args.extended_panel)
            print(json.dumps({'adjacency':adjacency,'seed':seed,'status':'complete'}),flush=True)
