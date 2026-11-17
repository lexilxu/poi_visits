"""Replay a corrected static checkpoint and export exact own-POI attribution.

Static attributes are single shared input leaves across the historical steps.
Their time derivatives are summed before taking absolute Gradient x Input.
The five shares normalize own-POI inputs; global weight-conditioning effects
from other POIs are outside these shares. Execute only on AutoDL4.
"""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,socket,sys,time
import numpy as np
import pandas as pd
import torch
from exp.exp5.own_gradients import forward_with_support, diagonal_blocks
from exp.common.evolvegcn_io import load_checkpoint, origin_batch_size

GROUPS=['History','Location','Identity','Parent','Others']


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def own_scores(history,static,gh,gs):
    hs=(history*gh).abs().sum(1)[0]
    ss=(static*gs).abs()
    groups=torch.stack([hs,ss[:,:2].sum(1),ss[:,4:36].sum(1)+ss[:,44:52].sum(1),
                        ss[:,36:44].sum(1),ss[:,2:4].sum(1)],dim=1)
    semantic=torch.stack([ss[:,4:12].sum(1),ss[:,44:52].sum(1),ss[:,12:20].sum(1),ss[:,20:28].sum(1)],dim=1)
    return groups.detach().cpu().numpy().astype(np.float64),semantic.detach().cpu().numpy().astype(np.float64)


def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    loaded=load_checkpoint(args.source,args.run,args.bundle)
    record=loaded['record'];cfg=loaded['config']
    assert record['status']=='complete' and cfg['static'] and cfg['adjacency']=='Identity'
    gate=json.loads(args.gradient_gate.read_text())
    assert gate['status']=='passed' and gate['source_sha256']['own_gradients.py']==sha256((Path(__file__).resolve().parents[2] / 'exp/exp5/own_gradients.py').read_bytes()).hexdigest()
    device=torch.device('cuda:0');model=loaded['model'];panel=loaded['panel']
    test=loaded['split']=='test';origins=loaded['origins'][test];test_indices=np.flatnonzero(test)
    expected=loaded['prediction'][test];labels=panel.raw[loaded['target'][test]]
    mean,std=loaded['mean'],loaded['std']
    histories=(loaded['own'][test]-mean)/std
    args.out.mkdir(parents=True,exist_ok=False)
    n=len(panel.attrs);take=len(origins) if args.max_origins is None else min(args.max_origins,len(origins))
    raw=np.zeros((n,5));translated_raw=np.zeros((n,5));semantic_share=np.zeros(4);semantic_wins=np.zeros(4,dtype=np.int64);semantic_error=np.zeros(4)
    semantic_events=0;replay_max=0.;support_sizes=[];started=time.monotonic()
    static_base=model.encode_static().detach()
    static_reference=static_base.mean(0,keepdim=True)
    translation_replay_max=0.
    for i in range(take):
        begin=time.monotonic()
        history=torch.as_tensor(histories[i:i+1],device=device,dtype=torch.float32).requires_grad_(True)
        static_leaf=static_base.clone().requires_grad_(True)
        physical=origin_batch_size(loaded,test_indices[i])
        logits,support=forward_with_support(model,history,static_leaf,physical)
        prediction=torch.clamp(logits*std+mean,min=0.)
        if i==0:
            with torch.no_grad():
                translated_logits,_=forward_with_support(model,history.detach(),(static_base-static_reference)+static_reference,physical)
                translated_prediction=torch.clamp(translated_logits*std+mean,min=0.)
                np.testing.assert_allclose(translated_prediction.cpu().numpy(),prediction.detach().cpu().numpy(),atol=1e-5,rtol=1e-5)
                translation_replay_max=float((translated_prediction-prediction.detach()).abs().max())
        replay=prediction.detach().cpu().numpy()[0]
        np.testing.assert_allclose(replay,expected[i],atol=1e-5,rtol=1e-5)
        replay_max=max(replay_max,float(np.max(np.abs(replay-expected[i]))));support_sizes.append(len(support))
        origin_raw=np.zeros_like(raw)
        for lead in range(cfg['H']):
            gh,gs=diagonal_blocks(prediction,history,static_leaf,support,lead,args.vjp_chunk)
            groups,semantics=own_scores(history,static_leaf,gh,gs)
            translated_groups,_=own_scores(history,static_leaf-static_reference,gh,gs)
            translated_raw+=translated_groups
            assert np.isfinite(groups).all() and (groups>=0).all()
            origin_raw+=groups
            total=groups.sum(1);eligible=(semantics.sum(1)>0)&(total>0)
            semantic_share+=(semantics[eligible]/total[eligible,None]).sum(0)
            winners=semantics[eligible].argmax(1)
            semantic_wins+=np.bincount(winners,minlength=4)
            semantic_error+=np.bincount(winners,weights=np.abs(expected[i,lead]-labels[i,lead])[eligible],minlength=4)
            semantic_events+=int(eligible.sum())
        raw+=origin_raw
        np.savez_compressed(args.out/f'origin_{i:02d}.npz',raw_group_scores=origin_raw,forecast_origin=np.asarray(panel.dates[origins[i]]),global_support=support.cpu().numpy())
        progress=dict(completed_origins=i+1,requested_origins=take,total_origins=len(origins),
                      seconds=time.monotonic()-started,last_origin_seconds=time.monotonic()-begin,
                      support_nodes=len(support),replay_max_absolute_error=replay_max)
        atomic_json(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        del prediction,logits,history,static_leaf
    totals=raw.sum(1);eligible=totals>0
    shares=np.divide(raw,totals[:,None],out=np.zeros_like(raw),where=totals[:,None]>0)
    z=np.full_like(shares,np.nan)
    if eligible.any():
        mu=shares[eligible].mean(0);scale=shares[eligible].std(0)
        z[eligible]=(shares[eligible]-mu)/np.where(scale>0,scale,1.)
    assigned=np.full(n,-1,dtype=int);assigned[eligible]=z[eligible].argmax(1)
    errors=np.abs(expected[:take]-labels[:take]).mean((0,1))
    table=pd.DataFrame(dict(placekey=panel.attrs.placekey,regime=[GROUPS[j]+'-driven' if j>=0 else 'Unassigned' for j in assigned],mae=errors,total_own_attribution=totals))
    for g,name in enumerate(GROUPS):table[name+'_share']=shares[:,g];table[name+'_z']=z[:,g]
    table.to_csv(args.out/'poi_profiles.csv',index=False)
    translated_totals=translated_raw.sum(1);translated_eligible=translated_totals>0
    translated_shares=np.divide(translated_raw,translated_totals[:,None],out=np.zeros_like(translated_raw),where=translated_totals[:,None]>0)
    translated_assignment=np.full(n,-1,dtype=int)
    if translated_eligible.any():
        shifted=translated_shares[translated_eligible]
        centered=(shifted-shifted.mean(0))/np.where(shifted.std(0)>0,shifted.std(0),1.)
        translated_assignment[translated_eligible]=centered.argmax(1)
    sensitivity=pd.DataFrame(dict(placekey=panel.attrs.placekey,
        original_regime=assigned,translated_regime=translated_assignment,
        original_total=totals,translated_total=translated_totals))
    for g,name in enumerate(GROUPS):sensitivity[name+'_translated_share']=translated_shares[:,g]
    sensitivity.to_csv(args.out/'coordinate_translation.csv',index=False)
    jointly_assigned=eligible&translated_eligible
    rows=[]
    for g,name in enumerate(GROUPS):
        selected=assigned==g
        rows.append(dict(regime=name+'-driven',n_pois=int(selected.sum()),mae=float(errors[selected].mean()) if selected.any() else np.nan,
                         **{key:float(shares[selected,k].mean()) if selected.any() else np.nan for k,key in enumerate(GROUPS)}))
    pd.DataFrame(rows).to_csv(args.out/'regime_summary.csv',index=False)
    pd.DataFrame(dict(descriptor=['Top category','Sub-category','NAICS-2','NAICS-4'],
        mean_share=semantic_share/max(semantic_events,1),dominance_rate=semantic_wins/max(semantic_events,1),
        dominant_mae=np.divide(semantic_error,semantic_wins,out=np.full(4,np.nan),where=semantic_wins>0),
        n_dominant=semantic_wins)).to_csv(args.out/'semantic_summary.csv',index=False)
    parent=panel.attrs.parent_placekey.fillna('').astype(str).str.strip().ne('').to_numpy()
    parent_valid=parent&eligible
    report=dict(status='complete' if take==len(origins) else 'pilot',architecture='EvolveGCN-H',protocol=cfg['protocol'],seed=cfg['seed'],
                forecast_origins=take,total_forecast_origins=len(origins),pois=n,unassigned_pois=int((~eligible).sum()),
                raw_location_largest=int((raw.argmax(1)[eligible]==1).sum()),
                recorded_parent_pois=int(parent.sum()),recorded_parent_assigned=int(parent_valid.sum()),
                recorded_parent_mean_share=float(shares[parent_valid,3].mean()) if parent_valid.any() else None,
                replay_max_absolute_error=replay_max,global_support_nodes=support_sizes,seconds=time.monotonic()-started,
                checkpoint_sha256=record['checkpoint_sha256'],gradient_gate=gate,
                coordinate_translation=dict(reference='Mean fitted encoded static vector across the fixed POI cohort; standardized history origin unchanged.',
                    function='Evaluate the same forecast function at translated coordinates by adding the reference back before all model layers.',
                    forward_max_absolute_error=translation_replay_max,
                    jointly_assigned_pois=int(jointly_assigned.sum()),
                    regime_agreement=float((assigned[jointly_assigned]==translated_assignment[jointly_assigned]).mean()) if jointly_assigned.any() else None,
                    translated_location_largest=int((translated_raw.argmax(1)[translated_eligible]==1).sum()),
                    mean_absolute_share_change=float(np.abs(shares[jointly_assigned]-translated_shares[jointly_assigned]).mean()) if jointly_assigned.any() else None,
                    interpretation='Coordinate-origin sensitivity of Gradient x Input under an unchanged prediction function.'),
                estimand='Absolute Gradient x Input for each target POI own history and static attributes, then normalized across five own-input groups.',
                static_handling='One static leaf per POI; temporal derivatives summed before absolute value.',
                physical_batch_sizes=[origin_batch_size(loaded,index) for index in test_indices[:take]],
                prediction_target='Clipped visit forecasts in original units; history inputs remain training-standardized and static inputs use the fitted encoding.',
                source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/exp5/own_gradients.py'),(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn_io.py')]})
    atomic_json(args.out/'complete.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for field in ['source','run','bundle','gradient-gate','out']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--max-origins',type=int);p.add_argument('--vjp-chunk',type=int,default=4)
    main(p.parse_args())
