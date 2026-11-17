"""Original eight neighbor groups plus prespecified matched-history controls.

Each target receives a separate full-input counterfactual. All graph layers and
input-conditioned weight trajectories are recomputed with fitted parameters fixed.
"""
from exp.common.provenance import source_file
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from exp.common.evolvegcn_io import load_checkpoint, targeted_worlds

NAMES=['All neighbors','Recently active','High mean','Top-10 active','Random-10',
       'Correlated-10','Temporally aligned','Low mean','Matched replacement','External matched control']


def selections(lag,labels,targets,rng,tree,features):
    means=lag.mean(0);centered=lag-means;lengths=np.linalg.norm(centered,axis=0)
    members={g:np.flatnonzero(labels==g) for g in np.unique(labels)}
    groups={name:[] for name in NAMES};donors=[];distances=[]
    for target in targets:
        neighbors=members[labels[target]];neighbors=neighbors[neighbors!=target]
        if not len(neighbors):
            for name in NAMES:groups[name].append(np.empty(0,dtype=int))
            donors.append(np.empty(0,dtype=int));continue
        history=lag[:,neighbors];strength=means[neighbors];mid=np.median(strength)
        corr=(centered[:,target]@centered[:,neighbors])/np.maximum(lengths[target]*lengths[neighbors],1e-7)
        high=np.flatnonzero(strength>mid)
        if not len(high):high=np.flatnonzero(strength>=mid)
        picks=[np.arange(len(neighbors)),np.flatnonzero(history[-7:].max(0)>0),high,
               np.argsort(-strength,kind='stable')[:10],rng.choice(len(neighbors),min(10,len(neighbors)),replace=False),
               np.argsort(-corr,kind='stable')[:10],np.flatnonzero(((history>0)&(lag[:,target,None]>0)).any(0)),
               np.flatnonzero(strength<mid)]
        for name,pick in zip(NAMES[:8],picks):groups[name].append(neighbors[pick])
        active=neighbors[picks[3]];matched=[]
        for peer in active:
            for count in [32,128,512,len(labels)]:
                distance,candidates=tree.query(features[peer],k=min(count,len(labels)))
                distance,candidates=np.atleast_1d(distance),np.atleast_1d(candidates)
                allowed=(labels[candidates]!=labels[target])&~np.isin(candidates,matched)
                if allowed.any():
                    pos=np.flatnonzero(allowed)[0];matched.append(int(candidates[pos]));distances.append(float(distance[pos]));break
            else:raise ValueError('Insufficient distinct external donors')
        matched=np.asarray(matched,dtype=int)
        groups['Matched replacement'].append(active);groups['External matched control'].append(matched);donors.append(matched)
    return groups,donors,distances


def main(args):
    executor=targeted_worlds;reuse_gate=None
    if args.world_reuse_gate:
        reuse_gate=json.loads(args.world_reuse_gate.read_text());assert reuse_gate['status']=='passed'
        for name in ['reused_worlds.py','checkpoint_io.py']:
            assert reuse_gate['source_sha256'][name]==sha256(source_file(name).read_bytes()).hexdigest()
        assert reuse_gate['source_sha256']['export_neighbor_effects.py']==sha256(source_file('export_neighbor_effects.py').read_bytes()).hexdigest()
        from exp.exp4.reused_worlds import targeted_worlds_reused
        executor=targeted_worlds_reused
    loaded=load_checkpoint(args.source,args.run,args.bundle)
    identity=load_checkpoint(args.source,args.identity_run,args.bundle)
    from exp.common.original_protocol import group_codes, group_average
    cfg=loaded['config'];assert cfg['seed']==identity['config']['seed'] and cfg['T']==identity['config']['T'] and cfg['H']==identity['config']['H']
    assert cfg['adjacency'] in ['NAICS-2','NAICS-4'] and identity['config']['adjacency']=='Identity'
    np.testing.assert_array_equal(loaded['origins'],identity['origins'])
    panel=loaded['panel'];labels=group_codes(panel.attrs)[cfg['adjacency']]
    targets=np.flatnonzero(loaded['low_history'])
    test=np.flatnonzero(loaded['split']=='test')
    own=loaded['own'][test];before=loaded['prediction'][test][:,:,targets];reference=identity['prediction'][test][:,:,targets]
    truth=panel.raw[loaded['target'][test]][:,:,targets];valid=loaded['valid'][test][:,:,targets]
    node_mean=own.mean(1);sizes=np.bincount(labels)[labels]
    neighbor=(group_average(node_mean,labels)*sizes-node_mean)/np.maximum(sizes-1,1)
    thresholds=(float(np.median(node_mean[:,targets])),float(np.median(neighbor[:,targets])))
    # Original tables use descriptive test-input medians, not outcome-derived cutoffs.
    attempted=(node_mean[:,None,targets]<thresholds[0])&(neighbor[:,None,targets]>=thresholds[1])&(before-reference>=.1)
    gain=np.abs(reference-truth)-np.abs(before-truth)
    helpful=attempted&(truth>0)&(gain>0)&valid
    harmful=attempted&(truth==0)&(before>=.5)&valid
    recovered=helpful&(reference<.5)&(before>=.5)
    count_targets=len(targets) if args.max_targets is None else min(len(targets),args.max_targets)
    count_origins=len(test) if args.max_origins is None else min(len(test),args.max_origins)
    targets=targets[:count_targets];own=own[:count_origins]
    before,truth,valid,helpful,harmful,recovered=[a[:count_origins,:,:count_targets] for a in [before,truth,valid,helpful,harmful,recovered]]
    training=panel.raw[:loaded['train_end']];observed=np.isfinite(training)
    mean=np.divide(np.nansum(training,0),observed.sum(0),out=np.zeros(len(labels)),where=observed.sum(0)>0)
    frequency=np.divide((training>0).sum(0),observed.sum(0),out=np.zeros(len(labels)),where=observed.sum(0)>0)
    features=np.column_stack([np.log1p(mean),frequency,np.log1p(np.nanstd(training,axis=0))])
    features=(features-features.mean(0))/np.maximum(features.std(0),1e-8);tree=cKDTree(features)
    args.out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(cfg['seed']);counter={name:[] for name in NAMES};counts={name:[] for name in NAMES}
    paired_baselines={name:[] for name in NAMES};baseline_replay_max=0.
    distance_all=[];started=time.monotonic()
    for i,lag in enumerate(own):
        groups,donors,distances=selections(lag,labels,targets,rng,tree,features);distance_all.extend(distances)
        origin_values={}
        for name in NAMES:
            degrees=np.asarray([len(p) for p in groups[name]])
            after,paired=executor(loaded,lag,targets,groups[name],donors if name=='Matched replacement' else None,args.batch_size,return_baseline=True)
            if reuse_gate:
                probe=np.unique(np.linspace(0,len(targets)-1,16,dtype=int))
                check,check_base=targeted_worlds(loaded,lag,targets[probe],[groups[name][j] for j in probe],
                    [donors[j] for j in probe] if name=='Matched replacement' else None,8,True)
                np.testing.assert_array_equal(after[:,probe],check);np.testing.assert_array_equal(paired[:,probe],check_base)
            np.testing.assert_array_equal(after[:,degrees==0],paired[:,degrees==0])
            paired_baselines[name].append(paired)
            baseline_replay_max=max(baseline_replay_max,float(np.abs(paired-before[i]).max()))
            assert after.shape==before[i].shape and np.isfinite(after).all()
            counter[name].append(after);counts[name].append(degrees);origin_values[name]=after
            origin_values[name+'__baseline']=paired
        np.savez_compressed(args.out/f'origin_{i:02d}.npz',target_index=targets,**origin_values)
        progress=dict(completed_origins=i+1,total_origins=count_origins,targets=count_targets,seconds=time.monotonic()-started)
        (args.out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    rows=[]
    def average(values,selected):return float(values[selected].mean()) if selected.any() else None
    for name in NAMES:
        after=np.stack(counter[name]);paired=np.stack(paired_baselines[name]);degree=np.stack(counts[name]);drop=paired-after
        per_node=drop/np.maximum(degree[:,None,:],1);error=np.abs(after-truth)-np.abs(paired-truth)
        rows.append(dict(neighbor_group=name,mean_perturbed_nodes=float(degree.mean()),
            beneficial_events=int(helpful.sum()),harmful_events=int(harmful.sum()),recovered_events=int(recovered.sum()),
            beneficial_lift=average(drop,helpful),harmful_lift=average(drop,harmful),
            beneficial_per_node=average(per_node,helpful),harmful_per_node=average(per_node,harmful),
            beneficial_mae_effect=average(error,helpful),harmful_mae_effect=average(error,harmful),
            false_activations_removed=average(after<.5,harmful),recoveries_lost=average(after<.5,recovered),
            all_event_MAE_change=average(error,valid),all_event_forecast_reduction=average(drop,valid)))
    pd.DataFrame(rows).to_csv(args.out/'neighbor_effects.csv',index=False)
    report=dict(status='complete' if args.max_targets is None and args.max_origins is None else 'pilot',
                architecture='EvolveGCN-H',seed=cfg['seed'],adjacency=cfg['adjacency'],T=cfg['T'],H=cfg['H'],protocol=cfg['protocol'],
                targets=count_targets,origins=count_origins,seconds=time.monotonic()-started,
                replay_audit=loaded['replay_audit'],identity_replay_audit=identity['replay_audit'],
                evidence_thresholds=thresholds,threshold_scope='Original descriptive test-input median strata',
                donor_distance_mean=float(np.mean(distance_all)) if distance_all else None,
                donor_distance_p95=float(np.quantile(distance_all,.95)) if distance_all else None,
                intervention='Separate full-input world for each target; graph and learned parameters fixed, weight trajectories recomputed.',
                empty_sets='Empty interventions reproduce the paired unperturbed prediction exactly.',
                effect_baseline='Unperturbed full-input worlds evaluated with the same batch shape and arithmetic as each intervention.',
                physical_world_batch_size=8,tail_padding='Duplicate the final world; retain the requested targets only.',
                paired_baseline_max_difference_from_archive=baseline_replay_max,
                matching='Distinct external-group donors, nearest in training-standardized log mean, nonzero frequency and log standard deviation.',
                disconnected_control='Measured comparator; EvolveGCN-H can transmit dependence through globally conditioned weights.',
                checkpoint_sha256=loaded['record']['checkpoint_sha256'],
                source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn_io.py')]})
    if reuse_gate:
        report['computation_reuse']=dict(gate_sha256=sha256(args.world_reuse_gate.read_bytes()).hexdigest(),
            per_origin_group_reference_targets=16,reference_comparison='Bitwise equal target predictions and paired baselines for every origin and group.',
            scope='Identical complete input worlds share one evaluation; all targets, groups and origins retained.')
        for name in ['reused_worlds.py','reference_export_neighbor_effects.py']:
            report['source_sha256'][name]=sha256(source_file(name).read_bytes()).hexdigest()
    (args.out/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for name in ['source','run','identity-run','bundle','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-targets',type=int);p.add_argument('--max-origins',type=int);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--world-reuse-gate',type=Path)
    main(p.parse_args())
