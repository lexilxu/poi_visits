"""Own-feature sensitivity versus full-model responses at fixed sampled events."""
from exp.common.provenance import source_hash
from exp.common.provenance import source_file
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from exp.common.evolvegcn_io import load_checkpoint, origin_batch_size
from exp.exp5.own_gradients import forward_with_support, diagonal_blocks
from exp.exp5.feature_worlds import GROUPS, perturb_group, encoded_forecast
from exp.exp5.own_gradients import PHYSICAL_BATCH_SIZE


def ranking_agreement(score,response):
    """Spearman correlation over the five groups within informative events."""
    x=score.reshape(5,-1);y=np.abs(response).reshape(5,-1)
    informative=(np.ptp(x,axis=0)>1e-10)&(np.ptp(y,axis=0)>1e-4)
    if not informative.any():return dict(events=0,mean_spearman=None)
    a=rankdata(x[:,informative],axis=0);b=rankdata(y[:,informative],axis=0)
    a-=a.mean(0);b-=b.mean(0)
    corr=(a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return dict(events=int(informative.sum()),mean_spearman=float(corr.mean()))


def summarize(origins,groups):
    rows=[];rankings=[]
    scores=np.concatenate([o['raw_scores'] for o in origins],axis=1)
    baseline=np.concatenate([o['baseline'] for o in origins],axis=0)
    truth=np.concatenate([o['truth'] for o in origins],axis=0)
    for ref_index,reference in enumerate(['encoded_zero','training_activity_matched_POI']):
        directions=np.concatenate([o['directions'][ref_index] for o in origins],axis=1)
        for step_index,step in enumerate([.01,1.]):
            after=np.concatenate([o['after'][ref_index,step_index] for o in origins],axis=1)
            drop=baseline[None]-after
            rankings.append(dict(reference=reference,step=step,
                raw_gradient_input=ranking_agreement(scores,drop),
                directional_gradient=ranking_agreement(np.abs(directions),drop),
                zero_all_group_score_events=int((scores.sum(0)<=1e-10).sum()),
                zero_all_group_score_with_response=int(((scores.sum(0)<=1e-10)&(np.abs(drop).max(0)>1e-4)).sum())))
            for g,name in enumerate(groups):
                actual=drop[g];linear=step*directions[g]
                informative=(np.abs(actual)>1e-4)&(np.abs(linear)>1e-4)
                scale=float(np.abs(actual).mean());deviation=float(np.abs(actual-linear).mean())
                rows.append(dict(reference=reference,step=step,group=name,events=int(actual.size),
                    mean_raw_gradient_input=float(scores[g].mean()),mean_absolute_response=scale,
                    mean_forecast_reduction=float(actual.mean()),directional_error_MAE=deviation,
                    directional_relative_error=deviation/scale if scale>1e-6 else None,
                    directional_sign_agreement=float((np.sign(actual[informative])==np.sign(linear[informative])).mean()) if informative.any() else None,
                    sign_informative_events=int(informative.sum()),
                    zero_score_with_response=int(((scores[g]<=1e-10)&(np.abs(actual)>1e-4)).sum()),
                    mean_MAE_change=float((np.abs(after[g]-truth)-np.abs(baseline-truth)).mean())))
    return rows,rankings


def main(args):
    if args.out.exists():raise FileExistsError('Existing analysis is immutable')
    loaded=load_checkpoint(args.source,args.run,args.bundle)
    model=loaded['model'];cfg=loaded['config'];panel=loaded['panel']
    assert cfg['static'] and cfg['adjacency']=='Identity' and cfg['H']==7
    gate=json.loads(args.world_gate.read_text());gradient_gate=json.loads(args.gradient_gate.read_text())
    for name,record in [('feature_worlds.py',gate),('own_gradients.py',gradient_gate)]:
        assert record['status']=='passed'
        assert record['source_sha256'][name]==sha256(source_file(name).read_bytes()).hexdigest()
    assert gate['source_sha256'].get('evolvegcn.py', gate['source_sha256'].get('model.py'))==source_hash(loaded['record'], 'exp/common/evolvegcn.py')
    targets=np.sort(np.random.default_rng(421).choice(len(panel.attrs),256,replace=False))
    test=np.flatnonzero(loaded['split']=='test')
    origins=test[np.linspace(0,len(test)-1,3,dtype=int)]
    if args.max_targets is not None:targets=targets[:args.max_targets]
    if args.max_origins is not None:origins=origins[:args.max_origins]
    training=panel.raw[:loaded['train_end']];observed=np.isfinite(training)
    count=observed.sum(0)
    mean=np.divide(np.nansum(training,0),count,out=np.zeros(len(count)),where=count>0)
    frequency=np.divide((training>0).sum(0),count,out=np.zeros(len(count)),where=count>0)
    activity=np.column_stack([np.log1p(mean),frequency,np.log1p(np.nanstd(training,axis=0))])
    activity=(activity-activity.mean(0))/np.maximum(activity.std(0),1e-8)
    distances,nearest=cKDTree(activity).query(activity[targets],k=2)
    donors=np.asarray([row[np.flatnonzero(row!=target)[0]] for row,target in zip(nearest,targets)])
    assert (donors!=targets).all()
    static=model.encode_static().detach();groups=list(GROUPS);results=[]
    args.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    for number,index in enumerate(origins):
        own=torch.as_tensor((loaded['own'][index:index+1]-loaded['mean'])/loaded['std'],device='cuda',dtype=torch.float32).requires_grad_(True)
        attributes=static.clone().requires_grad_(True)
        physical=origin_batch_size(loaded,index)
        logits,support=forward_with_support(model,own,attributes,physical)
        forecast=torch.clamp(logits*loaded['std']+loaded['mean'],min=0.)
        np.testing.assert_allclose(forecast.detach().cpu().numpy()[0],loaded['prediction'][index],atol=1e-5,rtol=1e-5)
        with torch.no_grad():
            paired_base=encoded_forecast(model,own.detach(),static,loaded['mean'],loaded['std'],physical)
        np.testing.assert_allclose(paired_base.cpu().numpy(),forecast.detach().cpu().numpy(),atol=1e-5,rtol=1e-5)
        baseline=paired_base.cpu().numpy()[0][:,targets]
        gradient_baseline_difference=float(np.abs(baseline-forecast.detach().cpu().numpy()[0][:,targets]).max())
        history_base=own[0,:,targets].T.detach()
        feature_base=static[targets]
        history_delta=[history_base,history_base-own[0,:,donors].T.detach()]
        static_delta=[feature_base,feature_base-static[donors]]
        raw_scores=np.empty((5,cfg['H'],len(targets)),dtype=np.float64)
        directions=np.empty((2,5,cfg['H'],len(targets)),dtype=np.float64)
        for lead in range(cfg['H']):
            gh,gs=diagonal_blocks(forecast,own,attributes,support,lead,args.vjp_chunk)
            dh=gh[0,:,targets].T;ds=gs[targets]
            for g,name in enumerate(groups):
                columns=GROUPS[name]
                if columns is None:
                    raw_scores[g,lead]=(dh*history_base).abs().sum(1).cpu().numpy()
                    for ref in range(2):directions[ref,g,lead]=(dh*history_delta[ref]).sum(1).cpu().numpy()
                else:
                    raw_scores[g,lead]=(ds[:,columns]*feature_base[:,columns]).abs().sum(1).cpu().numpy()
                    for ref in range(2):directions[ref,g,lead]=(ds[:,columns]*static_delta[ref][:,columns]).sum(1).cpu().numpy()
        del forecast,logits,attributes,gh,gs,dh,ds
        after=np.empty((2,2,5,cfg['H'],len(targets)),dtype=np.float32)
        own=own.detach()
        for ref in range(2):
            for step_index,step in enumerate([.01,1.]):
                for g,name in enumerate(groups):
                    values=perturb_group(model,own,static,targets,history_delta[ref],static_delta[ref],name,step,
                        loaded['mean'],loaded['std'],args.batch_size,physical)
                    columns=GROUPS[name]
                    delta=history_delta[ref] if columns is None else static_delta[ref][:,columns]
                    unchanged=(delta.abs().sum(1)==0).cpu().numpy()
                    np.testing.assert_array_equal(values[:,unchanged],baseline[:,unchanged])
                    after[ref,step_index,g]=values
        result=dict(raw_scores=raw_scores,directions=directions,after=after,baseline=baseline,
            truth=panel.raw[loaded['target'][index]][:,targets],target_indices=targets,donor_indices=donors,
            forecast_origin=np.asarray(panel.dates)[loaded['origins'][index]],support_nodes=len(support),
            gradient_baseline_difference=gradient_baseline_difference)
        assert np.isfinite(after).all() and np.isfinite(raw_scores).all() and np.isfinite(directions).all()
        np.savez_compressed(args.out/f'origin_{number:02d}.npz',**result)
        results.append(result)
        progress=dict(completed_origins=number+1,total_origins=len(origins),targets=len(targets),seconds=time.monotonic()-started)
        (args.out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    rows,rankings=summarize(results,groups)
    pd.DataFrame(rows).to_csv(args.out/'group_responses.csv',index=False)
    report=dict(status='complete' if args.max_targets is None and args.max_origins is None else 'pilot',
        protocol=cfg['protocol'],seed=cfg['seed'],sample_targets=len(targets),origins=len(origins),
        forecast_origins=[str(r['forecast_origin']) for r in results],rankings=rankings,
        sample_rule='Uniform fixed sample of 256 POIs using seed421; first, middle and last December test origin, selected before reading response values.',
        references=['Zero in fitted encoded coordinates','Distinct POI nearest in training log-mean, nonzero frequency and log-standard-deviation'],
        reference_scope='Donor blocks are observed feature groups; holding the remaining target features fixed does not establish joint input plausibility.',
        responses='Each target-group world changes one POI, holds fitted parameters and identity graph fixed, and recomputes both weight trajectories.',
        gradient_comparison='Compare raw absolute Gradient x Input rankings with absolute response. Compare signed gradient dot input change with forecast reduction at 1% and full steps.',
        zero_convention='Zero scores remain zero; correlations require variation in both the scores and responses. Unchanged feature blocks must reproduce the paired unperturbed forecast exactly.',
        physical_world_batch_sizes=[origin_batch_size(loaded,index) for index in origins],
        gradient_vs_response_baseline_max_error=max(r['gradient_baseline_difference'] for r in results),
        matched_distance_mean=float(np.linalg.norm(activity[targets]-activity[donors],axis=1).mean()),
        sampled_recorded_parent_pois=int(panel.attrs.parent_placekey.iloc[targets].fillna('').astype(str).str.strip().ne('').sum()),
        checkpoint_sha256=loaded['record']['checkpoint_sha256'],replay_audit=loaded['replay_audit'],
        seconds=time.monotonic()-started,
        source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*[source_file(n) for n in ['feature_worlds.py','own_gradients.py','checkpoint_io.py']]]})
    (args.out/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for field in ['source','run','bundle','world-gate','gradient-gate','out']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--max-targets',type=int);p.add_argument('--max-origins',type=int)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--vjp-chunk',type=int,default=4)
    main(p.parse_args())
