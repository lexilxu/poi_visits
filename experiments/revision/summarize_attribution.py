"""Cross-seed attribution stability, feature-removal utility, and parent diagnostics."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from audit_protocol import load_panel
from attribution_audit import GROUPS,SEMANTICS,group_indices,semantic_indices
from checkpoint_tools import selected_directories
from evaluate_saved import load_events,aligned
from summarize_queue import write_json


def profile(absolute,t):
    total=np.stack([absolute[:,i].sum(1) for i in group_indices(t)],axis=1)
    valid=total.sum(1)>0
    shares=np.divide(total,total.sum(1,keepdims=True),out=np.zeros_like(total),where=valid[:,None])
    standardized=np.zeros_like(shares)
    if valid.any():
        standardized[valid]=(shares[valid]-shares[valid].mean(0))/np.maximum(shares[valid].std(0),1e-12)
    regime=np.where(valid,standardized.argmax(1),-1)
    return shares,regime,standardized,valid


def agreement(a,b,t):
    sa,ra,_,va=profile(a,t);sb,rb,_,vb=profile(b,t);valid=va & vb
    result=dict(eligible_POIs=int(valid.sum()),unassigned_A=int((~va).sum()),unassigned_B=int((~vb).sum()),
                regime_agreement=float((ra[valid]==rb[valid]).mean()) if valid.any() else None)
    for k,name in enumerate(GROUPS):
        rho=spearmanr(sa[valid,k],sb[valid,k]).statistic if valid.sum()>2 else np.nan
        result[name+'_rank_spearman']=float(rho) if np.isfinite(rho) else None
        result[name+'_share_mean_A']=float(sa[valid,k].mean()) if valid.any() else None
        result[name+'_share_mean_B']=float(sb[valid,k].mean()) if valid.any() else None
    return result


def per_poi_errors(paths,january=None):
    data_by_seed={};errors={}
    for path in paths:
        seed=json.loads((path/'fit.json').read_text())['seed']
        if january is None:prediction=path/'predictions.npz'
        else:prediction=january/'private_predictions'/(path.parent.name+f'_seed{seed}.npz')
        data=load_events(prediction);test=data['dataset_split']=='test'
        valid=data['valid'][test];error=np.abs(data['observed_visits'][test]-data['predicted_visits'][test])
        errors[seed]=np.where(valid,error,0.).sum((0,1))/valid.sum((0,1))
        data_by_seed[seed]=data
    return errors,data_by_seed


def run(args):
    raw,dates,attrs=load_panel(args.bundle);n=len(attrs);t=14
    originals=selected_directories(args.queue,'GNN-static','Identity','full',14)
    if len(originals)!=5:raise ValueError('All five GNN-static fits must be complete')
    full_error,full_events=per_poi_errors(originals,args.january)
    profiles={};protocols=[];rows=[];comparisons=[]
    for seed in range(42,47):
        folder=args.attribution/f'seed{seed}'
        protocol=json.loads((folder/'attribution_protocol.json').read_text());protocols.append(protocol)
        methods={}
        for name in ('Gradient_x_Input','IG_reference_1','IG_reference_2'):
            path=folder/'private_attribution'/f'{name}.npz'
            with np.load(path,allow_pickle=False) as z:
                indices=z['poi_index'];absolute=z['absolute']
                np.testing.assert_array_equal(z['poi_id'],attrs.placekey.to_numpy(dtype=str)[indices])
            shares,regime,standardized,valid=profile(absolute,t)
            methods[name]=dict(absolute=absolute,indices=indices,shares=shares,regime=regime,valid=valid)
            for j,group in enumerate(GROUPS):
                rows.append(dict(seed=seed,method=name,group=group,POIs=len(indices),eligible_POIs=int(valid.sum()),
                                 unassigned=int((~valid).sum()),mean_share=float(shares[valid,j].mean()),
                                 regime_fraction=float((regime[valid]==j).mean())))
            total=absolute.sum(1)
            semantic=np.stack([absolute[:,i].sum(1) for i in semantic_indices(t)],axis=1)
            sem_valid=semantic.sum(1)>0
            for j,name_sem in enumerate(SEMANTICS):
                rows.append(dict(seed=seed,method=name,group=name_sem,POIs=len(indices),
                    eligible_POIs=int(valid.sum()),unassigned=int((~valid).sum()),
                    mean_share=float((semantic[valid,j]/total[valid]).mean()),
                    within_semantic_share=float((semantic[sem_valid,j]/semantic[sem_valid].sum(1)).mean()),
                    dominant_POI_fraction=float((semantic[sem_valid].argmax(1)==j).mean())))
        sample=methods['IG_reference_1']['indices']
        for name in ('IG_reference_1','IG_reference_2'):
            comparisons.append(dict(seed=seed,method_A='Gradient_x_Input',method_B=name,
                     **agreement(methods['Gradient_x_Input']['absolute'][sample],methods[name]['absolute'],t)))
        comparisons.append(dict(seed=seed,method_A='IG_reference_1',method_B='IG_reference_2',
                     **agreement(methods['IG_reference_1']['absolute'],methods['IG_reference_2']['absolute'],t)))
        profiles[seed]=methods
    stability=[]
    for a,b in combinations(range(42,47),2):
        stability.append(dict(seed_A=a,seed_B=b,**agreement(profiles[a]['Gradient_x_Input']['absolute'],
                                                          profiles[b]['Gradient_x_Input']['absolute'],t)))
    all_shares=np.stack([profiles[s]['Gradient_x_Input']['shares'] for s in range(42,47)])
    all_valid=np.stack([profiles[s]['Gradient_x_Input']['valid'] for s in range(42,47)])
    consensus=np.divide(all_shares.sum(0),all_valid.sum(0)[:,None],out=np.zeros((n,5)),where=all_valid.sum(0)[:,None]>0)
    eligible=consensus.sum(1)>0;standardized=np.zeros_like(consensus)
    standardized[eligible]=(consensus[eligible]-consensus[eligible].mean(0))/np.maximum(consensus[eligible].std(0),1e-12)
    regime=np.where(eligible,standardized.argmax(1),-1)
    mean_error=np.mean([full_error[s] for s in range(42,47)],axis=0)
    regime_rows=[];categories=[];fingerprint=[]
    for k,name in enumerate(GROUPS):
        keep=regime==k
        if not keep.any():continue
        centroid=consensus[keep].mean(0)
        indices=np.flatnonzero(keep)
        representatives=indices[np.argsort(np.square(consensus[keep]-centroid).sum(1),kind='stable')[:2]]
        regime_rows.append(dict(regime=name,POIs=int(keep.sum()),MAE=float(mean_error[keep].mean()),
                          recorded_parent_POIs=int((keep & attrs.parent_placekey.fillna('').astype(str).ne('').to_numpy()).sum()),
                          representative_POIs='; '.join(attrs.location_name.iloc[representatives].astype(str))))
        for j,group in enumerate(GROUPS):fingerprint.append(dict(regime=name,source=group,
                          standardized_share=float(standardized[keep,j].mean()),mean_share=float(consensus[keep,j].mean())))
    for category in attrs.category9.unique():
        keep=attrs.category9.to_numpy()==category;assigned=keep & eligible
        categories.append(dict(category=category,POIs=int(keep.sum()),unassigned=int((keep & ~eligible).sum()),
                               MAE=float(mean_error[keep].mean()),
                               **{group:float((regime[assigned]==k).mean()) for k,group in enumerate(GROUPS)}))
    parent=attrs.parent_placekey.fillna('').astype(str).ne('').to_numpy()
    parent_profiles=[]
    parent_groups=[('All embedded POIs',parent)]+[(str(label),parent & (attrs.category9.to_numpy()==label)) for label in attrs.category9.unique()]
    for seed in range(42,47):
        share=profiles[seed]['Gradient_x_Input']['shares'][:,3]
        valid=profiles[seed]['Gradient_x_Input']['valid']
        for label,keep in parent_groups:
            if not keep.any():continue
            parent_profiles.append(dict(seed=seed,category=label,POIs=int(keep.sum()),
                attribution_POIs=int((keep&valid).sum()),parent_share=float(share[keep&valid].mean()),
                full_MAE=float(full_error[seed][keep].mean())))
    if args.profiles_only:
        args.out.mkdir(parents=True,exist_ok=True)
        for name,values in [('profiles',rows),('method_agreement',comparisons),('seed_agreement',stability),
                           ('regimes',regime_rows),('regime_categories',categories),('fingerprint',fingerprint),
                           ('parent_profiles_by_seed',parent_profiles)]:
            pd.DataFrame(values).to_csv(args.out/f'{name}.csv',index=False)
        np.savez_compressed(args.out/'private_consensus.npz',poi_id=attrs.placekey.to_numpy(dtype=str),
                            shares=consensus,regime=regime,MAE=mean_error,standardized=standardized)
        write_json(args.out/'summary_protocol.json',dict(status='profiles_complete_parent_pending',seeds=list(range(42,47)),
              period='2021-01' if args.january else '2020-12',unassigned_POIs=int((~eligible).sum()),
              primary_regime='Standardize mean nonzero-attribution shares across all five fits; take largest standardized source',
              method_comparison='Identical IG-sampled POIs and forecast events; zero-total-attribution profiles excluded from agreement',
              completeness=[dict(seed=p['seed'],checks=p['completeness']) for p in protocols],
              summary_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
        print(json.dumps(dict(status='profiles_complete_parent_pending')));return
    parent_rows=[];parent_correlations=[]
    removed=selected_directories(args.queue,'GNN-static','Identity','without_parent',14)
    if len(removed)!=5:raise ValueError('All parent-removal fits must be complete')
    reduced_error,reduced_events=per_poi_errors(removed,args.january)
    for seed in range(42,47):
        aligned(full_events[seed],reduced_events[seed])
        share=profiles[seed]['Gradient_x_Input']['shares'][:,3]
        valid=profiles[seed]['Gradient_x_Input']['valid']
        for label,keep in [('All embedded POIs',parent)]+[(str(label),parent & (attrs.category9.to_numpy()==label)) for label in attrs.category9.unique()]:
            if not keep.any():continue
            parent_rows.append(dict(seed=seed,category=label,POIs=int(keep.sum()),
                      attribution_POIs=int((keep&valid).sum()),parent_share=float(share[keep&valid].mean()),
                      full_MAE=float(full_error[seed][keep].mean()),
                      without_parent_MAE=float(reduced_error[seed][keep].mean()),
                      removal_MAE_increase=float((reduced_error[seed]-full_error[seed])[keep].mean())))
        keep=parent & valid;rho=spearmanr(share[keep],(reduced_error[seed]-full_error[seed])[keep]).statistic
        parent_correlations.append(dict(seed=seed,POIs=int(keep.sum()),Spearman_share_vs_removal_MAE_increase=float(rho)))
    args.out.mkdir(parents=True,exist_ok=True)
    for name,values in [('profiles',rows),('method_agreement',comparisons),('seed_agreement',stability),
                       ('regimes',regime_rows),('regime_categories',categories),('fingerprint',fingerprint),
                       ('parent_by_seed',parent_rows),('parent_association',parent_correlations)]:
        pd.DataFrame(values).to_csv(args.out/f'{name}.csv',index=False)
    np.savez_compressed(args.out/'private_consensus.npz',poi_id=attrs.placekey.to_numpy(dtype=str),
                        shares=consensus,regime=regime,MAE=mean_error,standardized=standardized)
    write_json(args.out/'summary_protocol.json',dict(status='complete',seeds=list(range(42,47)),
                period='2021-01' if args.january else '2020-12',unassigned_POIs=int((~eligible).sum()),
                primary_regime='Standardize mean nonzero-attribution shares across all five fits; take largest standardized source',
                method_comparison='Identical IG-sampled POIs and forecast events; zero-total-attribution profiles excluded from agreement',
                parent_association='Descriptive Spearman correlation within recorded-parent POIs; no independent-event significance test',
                completeness=[dict(seed=p['seed'],checks=p['completeness']) for p in protocols],
                summary_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status='complete',period='January' if args.january else 'December')))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--attribution',type=Path,required=True);p.add_argument('--january',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profiles-only',action='store_true',help='Summarize completed attribution profiles without claiming parent ablation is complete')
    run(p.parse_args())
