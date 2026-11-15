"""Reproduce simple baselines on the archived forecast windows.

All selection uses validation labels only. Predictions stay local. Aggregate
results distinguish the archived, window-weighted cohort from a fixed cohort.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_protocol import load_panel, window_arrays, low_history, SETTINGS, TRAIN_END


def observed_mean(x, axis=0):
    count = np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x, axis=axis), count,
                     out=np.zeros(count.shape, dtype=float), where=count > 0)


def average_precision(y, score):
    """Non-interpolated AP, grouping tied scores into one threshold."""
    order = np.argsort(-score, kind='stable')
    labels, scores = y[order], score[order]
    if labels.sum() == 0:
        return None
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    true_positive = np.cumsum(labels)[ends]
    recall = true_positive / labels.sum()
    return float(np.sum(np.diff(np.r_[0., recall]) * true_positive / (ends+1)))


def metrics(y, pred, mask, cohort):
    keep = mask & cohort[None, None, :]
    target, forecast = y[keep], pred[keep]
    positive = target > 0
    zero = ~positive
    row = {'events': int(keep.sum()), 'pois': int(cohort.sum()),
           'positive_prevalence': float(positive.mean()),
           'MAE': float(np.abs(target-forecast).mean()),
           'RMSE': float(np.sqrt(np.square(target-forecast).mean())),
           'WAPE': float(np.abs(target-forecast).sum()/target.sum()),
           'DMR': float(forecast[positive].sum()/target[positive].sum()),
           'mean_prediction': float(forecast.mean()),
           'average_precision': average_precision(positive, forecast)}
    for threshold in (.1, .25, .5, 1., 2.):
        activated = forecast >= threshold
        row[f'FAR_{threshold:g}'] = float(activated[zero].mean())
        row[f'recall_{threshold:g}'] = float(activated[positive].mean())
        row[f'precision_{threshold:g}'] = (float(positive[activated].mean())
                                            if activated.any() else None)
    return row


def predictions(raw, dates, t, h, protocol=None):
    origins, history, target, split, valid = (window_arrays(raw, t, h) if protocol is None
                                            else protocol.windows(raw))
    train_end = TRAIN_END if protocol is None else protocol.train_end
    calendar_mean = observed_mean(raw[:train_end])
    weighted_mean = observed_mean(raw[history[split == 'train']].reshape(-1, raw.shape[1]))
    shape = (len(origins), h, raw.shape[1])
    expand = lambda mean: np.broadcast_to(mean, shape)
    candidates = {'Zero': np.zeros(shape), 'Calendar HA': expand(calendar_mean),
                  'Archived HA': expand(weighted_mean)}
    candidates['Last value'] = np.broadcast_to(
        np.where(np.isfinite(raw[origins-1]), raw[origins-1], calendar_mean)[:,None,:], shape)
    seasonal_dates = origins[:,None] - 7 + (np.arange(h) % 7)
    candidates['Seasonal naive'] = np.where(np.isfinite(raw[seasonal_dates]),
                                           raw[seasonal_dates], calendar_mean)
    weekdays = pd.to_datetime(dates).dayofweek.to_numpy()
    weekday_means = np.stack([observed_mean(raw[:train_end][weekdays[:train_end] == d])
                             for d in range(7)])
    candidates['Weekday HA'] = weekday_means[weekdays[target]]
    recent = {}
    for length in (7, 14, 28):
        if length <= t:
            means = observed_mean(raw[origins[:,None]+np.arange(-length,0)], axis=1)
            recent[length] = np.broadcast_to(means[:,None,:], shape)
    validation = valid & (split == 'validation')[:,None,None]
    scores = {length: float(np.abs(pred-raw[target])[validation].mean())
              for length, pred in recent.items()}
    selected = min(scores, key=lambda length: (scores[length], length))
    candidates['Recent mean'] = recent[selected]
    selection = {'T':t, 'H':h, 'selected_recent_days':selected,
                 'validation_MAE_by_recent_days':scores,
                 'selection_objective':'masked validation MAE; ties favor shorter history'}
    return candidates, origins, history, target, split, valid, selection


def block_interval(loss_difference, seed=421, repetitions=2000):
    """Circular seven-origin block bootstrap, conditional on this POI panel.

    Origins are resampled jointly across all POIs and leads. This does not
    estimate uncertainty across cities or across independent POI samples.
    """
    n = len(loss_difference)
    if n < 14:
        return None
    rng = np.random.default_rng(seed)
    start = rng.integers(n, size=(repetitions, int(np.ceil(n/7))))
    indices = ((start[...,None]+np.arange(7)) % n).reshape(repetitions,-1)[:,:n]
    delta = loss_difference[indices].mean(axis=1)
    return [float(v) for v in np.quantile(delta, [.025,.975])]


def run(bundle, out, complete_horizon=False):
    from training_protocol import Protocol
    raw, dates, _ = load_panel(bundle)
    out.mkdir(parents=True, exist_ok=True)
    rows, selections, intervals = [], [], []
    fixed, _ = low_history(raw[:50 if complete_horizon else TRAIN_END])
    settings = ((14,7),(28,7)) if complete_horizon else SETTINGS
    for t, h in settings:
        protocol = Protocol(T=t,H=h) if complete_horizon else None
        candidates, origins, history, target, split, valid, selected = predictions(raw, dates, t, h, protocol)
        selections.append(selected)
        legacy, _ = low_history(raw[history[split == 'train']].reshape(-1, raw.shape[1]))
        test = split == 'test'
        y, mask = raw[target[test]], valid[test]
        cohorts = {'All':np.ones(raw.shape[1], dtype=bool), 'Archived low-history':legacy,
                   'Calendar low-history':fixed}
        reference = np.abs(candidates['Archived HA'][test]-y)
        reference_loss = np.where(mask, reference, 0).sum((1,2))/mask.sum((1,2))
        for model, pred in candidates.items():
            for cohort_name, cohort in cohorts.items():
                rows.append({'T':t,'H':h,'model':model,'cohort':cohort_name,
                             **metrics(y, pred[test], mask, cohort)})
            error = np.abs(pred[test]-y)
            by_origin = np.where(mask,error,0).sum((1,2))/mask.sum((1,2))
            delta = by_origin-reference_loss
            interval = block_interval(delta) if h == 7 else None
            intervals.append({'T':t,'H':h,'model':model,'origins':int(test.sum()),
                              'MAE_difference_from_archived_HA':float(delta.mean()),
                              'circular_7_origin_bootstrap_95CI':interval})
    results = pd.DataFrame(rows)
    archived_path = Path(__file__).resolve().parents[2]/'artifacts/archived_2020q4/exp1/multi_horizon_results.csv'
    reproduced = None
    if archived_path.exists() and not complete_horizon:
        archived = pd.read_csv(archived_path)
        expected = archived[archived.model == 'Historical average'].set_index(['T','H']).mae
        actual = results[(results.model == 'Archived HA') & (results.cohort == 'All')].set_index(['T','H']).MAE
        np.testing.assert_allclose(actual.reindex(expected.index),expected,atol=1e-6,rtol=0)
        reproduced = True
    results.to_csv(out/'baseline_metrics.csv',index=False)
    (out/'baseline_protocol.json').write_text(json.dumps({
        'selection':selections,'paired_diagnostics':intervals,
        'archived_HA_MAE_reproduced_to_1e_6':reproduced,
        'uncertainty_scope':'Conditional on the observed Seattle panel. Seven-origin blocks for H=7 only; short series limits precision.',
        'comparison_scope':('Complete-horizon windows, train_end=50, validation_end=64. No new neural results.'
                            if complete_horizon else 'Same target masks as archived experiments; no new neural results.')
    },indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(pd.DataFrame(rows).query("cohort == 'All'")[['T','H','model','MAE']].to_string(index=False))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--complete-horizon',action='store_true')
    args = parser.parse_args()
    run(args.bundle,args.out,args.complete_horizon)
