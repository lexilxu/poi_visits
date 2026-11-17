"""Export the observed-day statistics underlying the descriptive paper figures.

Reductions follow paper_figures/generate_figures.py. Missing records remain NaN;
POI means and activity ratios use each POI's observed days.
"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
from exp.common.paths import DATA_ROOT, RESULTS_ROOT
from exp.exp0.categories import CATEGORY_MAP


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=DATA_ROOT/'prepared')
    parser.add_argument('--attributes', type=Path, default=DATA_ROOT/'Seattle_all_poi.csv')
    parser.add_argument('--out', type=Path, default=RESULTS_ROOT/'exp0/data')
    args = parser.parse_args()
    sources = [args.prepared/name for name in
               ['approved_period_visits_with_missing.npy', 'identified_9884_pois.csv', 'approved_dates.csv']]
    visits = np.load(sources[0], allow_pickle=False).astype(np.float64)
    ids = pd.read_csv(sources[1]).placekey
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(sources[2]).date))
    attrs = pd.read_csv(args.attributes, low_memory=False).drop_duplicates('placekey').set_index('placekey').reindex(ids)
    categories = attrs.top_category.map(CATEGORY_MAP).to_numpy()
    observed = np.isfinite(visits)
    flat = visits[observed]
    days = observed.sum(0)
    assert visits.shape == (92, 9884)
    assert dates.equals(pd.date_range('2020-10-01', '2020-12-31'))
    assert ids.is_unique and pd.notna(categories).all()
    assert np.all(flat >= 0) and np.all(flat == np.floor(flat))
    assert int((~observed).sum()) == 3410 and int((~observed).any(0).sum()) == 110
    assert np.all(days > 0)
    active = (visits > 0).sum(0)/days
    means = np.nansum(visits, axis=0)/days
    zero = 1-active
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(placekey=ids, category9=categories, observed_days=days,
                      active_day_fraction=active, mean_observed_visits=means,
                      zero_day_fraction=zero)).to_csv(args.out/'poi_summary.csv', index=False)
    daily = pd.DataFrame(dict(date=dates, mean_observed_visits=np.nanmean(visits, axis=1)))
    daily['centered_7_day_mean'] = daily.mean_observed_visits.rolling(7, center=True, min_periods=7).mean()
    daily.to_csv(args.out/'daily_mean_visits.csv', index=False)
    from exp.exp0.categories import ROW_ORDER
    order = ROW_ORDER
    rows = []
    for group in order:
        selected = categories == group
        for measure, values in [('mean_observed_visits', means[selected]), ('zero_day_percentage', zero[selected]*100)]:
            quantiles = np.quantile(values, [.05, .25, .5, .75, .95])
            rows.append(dict(category=group, pois=int(selected.sum()), measure=measure,
                             **{f'q{p}': float(x) for p, x in zip([5, 25, 50, 75, 95], quantiles)}))
    pd.DataFrame(rows).to_csv(args.out/'category_quantiles.csv', index=False)
    bins = [flat == k for k in range(6)]+[(flat >= 6) & (flat <= 10), (flat >= 11) & (flat <= 20), flat >= 21]
    labels = ['0', '1', '2', '3', '4', '5', '6–10', '11–20', '21+']
    pd.DataFrame(dict(visit_count_bin=labels, count=[int(b.sum()) for b in bins],
                      percentage=[float(b.mean()*100) for b in bins])).to_csv(args.out/'visit_count_distribution.csv', index=False)
    x, counts = np.unique(active*100, return_counts=True)
    pd.DataFrame(dict(active_days_percentage=x, cumulative_pois_percentage=counts.cumsum()/len(ids)*100)).to_csv(args.out/'active_day_ecdf.csv', index=False)
    record = dict(pois=len(ids), calendar_days=len(dates), observed_poi_days=int(observed.sum()),
                  missing_poi_days=int((~observed).sum()), zero_visit_percentage=float((flat == 0).mean()*100),
                  input_sha256={str(p): sha256(p.read_bytes()).hexdigest() for p in [*sources, args.attributes]},
                  denominator='Observed dates for each POI; observed POIs for each date',
                  scope='Full-period description; training-defined low-history cohorts are separate')
    (args.out/'statistics_and_provenance.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
