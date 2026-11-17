"""NLinear, TSB, validation calibration, and observed-activity prioritization.

NLinear uses the shared affine architecture in cure-lab/LTSF-Linear, fitted by
regularized least squares instead of stochastic optimization. This distinction
is recorded; this is not claimed as a reproduction of their training recipe.
TSB follows Teunter et al. (2011), doi:10.1016/j.ejor.2011.05.018.
All outputs use the archived masks for comparability, not a repaired protocol.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from exp.common.audit import load_panel, low_history, SETTINGS, TRAIN_END, EXPECTED_BUNDLE_SHA256
from exp.exp1.baselines import predictions, metrics, observed_mean, block_interval
from exp.common.controls import affine_calibration, matched_far_threshold, grid_groups, spatial_coverage


def nlinear(raw, history, target, split, valid):
    start = time.monotonic()
    hx = np.nan_to_num(raw[history], nan=0.).transpose(0,2,1).astype(float)
    last = hx[..., -1]
    train = split == 'train'
    scale = max(float(np.nanstd(raw[history[train]])), 1e-8)
    residual = (hx[..., :-1]-last[..., None])/scale
    design = np.concatenate((residual, np.ones((*last.shape, 1))), axis=-1)
    response = (np.nan_to_num(raw[target])-last[:, None, :])/scale
    h = target.shape[1]
    sufficient = []
    for lead in range(h):
        mask = valid[train, lead]
        x, y = design[train][mask], response[train, lead][mask]
        if not len(y):
            raise ValueError('No training labels for a forecast lead')
        sufficient.append((x.T@x/len(y), x.T@y/len(y)))
    penalty = np.eye(design.shape[-1]); penalty[-1,-1] = 0
    scores, weights = {}, {}
    val = valid & (split == 'validation')[:,None,None]
    best = None
    for alpha in (0., 1e-4, .01, 1.):
        coefficient = np.column_stack([np.linalg.lstsq(gram+alpha*penalty, cross, rcond=None)[0]
                                       for gram, cross in sufficient])
        pred = np.maximum(np.einsum('bnt,th->bhn', design, coefficient)*scale+last[:,None,:], 0.)
        loss = float(np.abs(pred-raw[target])[val].mean())
        scores[str(alpha)] = loss
        weights[str(alpha)] = coefficient
        if best is None or loss < best[0]:
            best = (loss, alpha, pred)
    return best[2], {'ridge_penalty':best[1], 'validation_MAE_by_penalty':scores,
                      'training_loss':'masked MSE plus coefficient penalty',
                      'selection_loss':'masked validation MAE',
                      'solver':'deterministic sufficient-statistic least squares',
                      'shared_across_pois':True, 'train_scale':scale,
                      'parameters':int(design.shape[-1]*h), 'seconds':time.monotonic()-start}, weights[str(best[1])]


def tsb_states(raw, alpha, beta):
    n = raw.shape[1]
    probability, size = np.zeros(n), np.zeros(n)
    seen, positive_seen = np.zeros(n, bool), np.zeros(n, bool)
    result = np.zeros((len(raw)+1, n))
    for day, values in enumerate(raw):
        observed, positive = np.isfinite(values), values > 0
        first = observed & ~seen
        probability[first] = positive[first]
        update = observed & seen
        probability[update] += beta*(positive[update]-probability[update])
        first_positive = positive & ~positive_seen
        size[first_positive] = values[first_positive]
        update_positive = positive & positive_seen
        size[update_positive] += alpha*(values[update_positive]-size[update_positive])
        seen |= observed; positive_seen |= positive
        result[day+1] = probability*size
    return result


def run(bundle, out, complete_horizon=False):
    from exp.common.protocol import Protocol
    raw, dates, attrs = load_panel(bundle)
    out.mkdir(parents=True, exist_ok=True)
    private = out/'private_predictions'; private.mkdir(exist_ok=True)
    rows, fits, paired, coverage_rows, far_rows, lead_rows = [], [], [], [], [], []
    calendar_cohort, _ = low_history(raw[:50 if complete_horizon else TRAIN_END])
    grid = grid_groups(attrs.latitude, attrs.longitude)
    tsb = {(a,b):tsb_states(raw,a,b) for a in (.05,.1,.2) for b in (.05,.1,.2)}
    settings = ((14,7),(28,7)) if complete_horizon else SETTINGS
    for t,h in settings:
        protocol = Protocol(T=t,H=h) if complete_horizon else None
        basic, origins, history, target, split, valid, recent_selection = predictions(raw,dates,t,h,protocol)
        train, val, test = (split == name for name in ('train','validation','test'))
        legacy, _ = low_history(raw[history[train]].reshape(-1,raw.shape[1]))
        y = raw[target]
        linear, fit, weights = nlinear(raw,history,target,split,valid)
        shape = linear.shape
        validation = valid & val[:,None,None]
        tsb_scores = {f'{a},{b}':float(np.abs(np.broadcast_to(s[origins,None,:],shape)-y)[validation].mean())
                      for (a,b),s in tsb.items()}
        key = min(tsb, key=lambda v:tsb_scores[f'{v[0]},{v[1]}'])
        models = {'NLinear ridge':linear,
                  'TSB':np.broadcast_to(tsb[key][origins,None,:],shape),
                  'Recent mean':basic['Recent mean'], 'Archived HA':basic['Archived HA']}
        fits.append({'T':t,'H':h,'model':'NLinear ridge',**fit})
        fits.append({'T':t,'H':h,'model':'TSB','alpha':key[0],'beta':key[1],
                     'validation_MAE_by_parameters':tsb_scores,
                     'initialization':'First observed indicator; first positive size. Missing days do not update.'})
        np.savez_compressed(private/f'nlinear_T{t}_H{h}_weights.npz',weights=weights,scale=fit['train_scale'])
        reference = np.where(valid[test],np.abs(basic['Archived HA'][test]-y[test]),0).sum((1,2))/valid[test].sum((1,2))
        for name, original in models.items():
            mask = validation & legacy[None,None,:]
            calibration = affine_calibration(original[mask],y[mask])
            fits.append({'T':t,'H':h,'model':name,'cohort':'Archived low-history',
                         'calibration':calibration})
            variants = [(name,original), (name+' calibrated',np.maximum(0.,calibration['a']*original+calibration['c']))]
            for label, pred in variants:
                for cohort_name, cohort in [('All',np.ones(raw.shape[1],bool)),
                                            ('Archived low-history',legacy),('Calendar low-history',calendar_cohort)]:
                    rows.append({'T':t,'H':h,'model':label,'cohort':cohort_name,
                                 **metrics(y[test],pred[test],valid[test],cohort)})
                errors = np.abs(pred[test]-y[test])
                delta = np.where(valid[test],errors,0).sum((1,2))/valid[test].sum((1,2))-reference
                paired.append({'T':t,'H':h,'model':label,'delta_MAE_vs_archived_HA':float(delta.mean()),
                               'CI95_7origin_block':block_interval(delta) if h == 7 else None})
                for lead in range(h):
                    lead_rows.append({'T':t,'H':h,'model':label,'horizon_step':lead+1,
                                      'MAE':float(errors[:,lead][valid[test][:,lead]].mean())})
                zeros = validation & legacy[None,None,:] & (y == 0)
                eval_mask = valid[test] & legacy[None,None,:]
                for level in (.05,.1,.2,.3):
                    threshold = matched_far_threshold(pred[zeros],level)
                    labels, active = y[test][eval_mask] > 0, pred[test][eval_mask] >= threshold
                    far_rows.append({'T':t,'H':h,'model':label,'target_validation_FAR':level,
                                     'threshold':threshold,'test_FAR':float(active[~labels].mean()),
                                     'test_recall':float(active[labels].mean())})
                if 'calibrated' not in label:
                    for fraction in (.05,.1,.2):
                        achieved, ideal, k, n = spatial_coverage(y[test],pred[test],grid,fraction)
                        coverage_rows.append({'T':t,'H':h,'model':label,'cell_km':1.,
                                              'selected_fraction':fraction,'selected_cells':k,'total_cells':n,
                                              'coverage':float(np.nanmean(achieved)),
                                              'oracle_coverage':float(np.nanmean(ideal)),
                                              'regret':float(np.nanmean(ideal-achieved))})
                    subset = val | test
                    np.savez_compressed(private/f'{name.replace(" ","_")}_T{t}_H{h}.npz',
                                        model=name,adjacency='None',seed=-1,
                                        forecast_origin=np.asarray(dates[origins[subset]],dtype=str),horizon_step=np.arange(1,h+1),
                                        poi_id=attrs.placekey.to_numpy(dtype=str),
                                        observed_visits=y[subset],predicted_visits=pred[subset].astype('float32'),
                                        valid=valid[subset],dataset_split=split[subset],
                                        train_end=50 if complete_horizon else TRAIN_END,
                                        T=t, H=h, protocol='complete_horizon' if complete_horizon else 'archived',
                                        bundle_sha256=EXPECTED_BUNDLE_SHA256)
                print(f'T={t} H={h} {label}: MAE {errors.mean():.6f}',flush=True)
    for filename, records in [('extended_baseline_metrics',rows),('matched_far',far_rows),
                              ('horizon_step_metrics',lead_rows),('activity_prioritization',coverage_rows)]:
        pd.DataFrame(records).to_csv(out/f'{filename}.csv',index=False)
    (out/'extended_baseline_protocol.json').write_text(json.dumps({
        'bundle_sha256':EXPECTED_BUNDLE_SHA256,'fits':fits,'paired_comparisons':paired,
        'scope':('Complete-horizon windows, train_end=50, validation_end=64. No new neural fits.' if complete_horizon
                 else 'Archived date masks. No new neural fits. H>7 validation remains incomplete.'),
        'calibration_scope':'Affine calibration fitted on low-history validation events only.',
        'spatial_scope':'Observed activity coverage; no claim about transit use or realized operational benefits.',
        'seed_note':'Deterministic models use seed=-1, not five artificial seed replicates.'
    },indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--complete-horizon',action='store_true')
    args = p.parse_args(); run(args.bundle,args.out,args.complete_horizon)
