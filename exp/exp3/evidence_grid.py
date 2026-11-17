"""Aggregate stored predictions into validation-defined evidence cells on AutoDL."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import sys
import numpy as np
import pandas as pd


def main(args):
    sys.path.insert(0, str(args.source / 'exp/common'))
    from exp.common.audit import load_panel, low_history
    raw, dates, attrs = load_panel(args.bundle)
    with np.load(args.extended, allow_pickle=False) as z:
        extended = z['visits']; all_dates = z['dates'].astype(str)
        np.testing.assert_array_equal(extended[:92], raw)
        np.testing.assert_array_equal(z['poi_id'], attrs.placekey.to_numpy(dtype=str))
    fine = pd.factorize(attrs.naics_code.astype('int64').astype(str).str[:4])[0]
    low, _ = low_history(raw[:50])
    targets = np.flatnonzero(low & (np.bincount(fine)[fine] > 1))
    assert len(targets) == 1935
    filled = np.nan_to_num(extended, nan=0.)
    validation = np.arange(50, 58)
    rows = []; reports = []; provenance = []
    for graph, digits in [('NAICS-2', 2), ('NAICS-4', 4)]:
        labels = pd.factorize(attrs.naics_code.astype('int64').astype(str).str[:digits])[0]
        size = np.bincount(labels)[labels[targets]]
        total = np.zeros((len(filled), labels.max()+1), dtype=np.float64)
        np.add.at(total.T, labels, filled.T)
        group = (total[:, labels] / np.bincount(labels)[labels]).astype(np.float32)
        propagated = (filled + group) * .5

        def evidence(origins):
            history = origins[:, None] + np.arange(-14, 0)
            own = filled[history][:, :, targets]
            target = own.mean(1)
            grouped = (2*propagated[history][:, :, targets]-own).mean(1)
            neighbor = (grouped*size-target)/(size-1)
            assert target.min() >= 0 and neighbor.min() >= -1e-7
            return target, neighbor

        vt, vn = evidence(validation)
        cuts = [np.unique(np.quantile(a, np.arange(1, 6)/6)) for a in (vt, vn)]
        medians = [float(np.median(a)) for a in (vt, vn)]
        for period in ['december', 'january']:
            arrays = []
            for seed in range(42, 47):
                directory = args.inputs / period / f'{graph}_seed{seed}'
                path = directory / 'stored_predictions.npz'
                with np.load(path, allow_pickle=False) as z: array = dict(z)
                np.testing.assert_array_equal(array['target_index'], targets)
                protocol = json.loads((directory / 'neighbor_protocol.json').read_text())
                np.testing.assert_allclose(protocol['evidence_medians_validation_only'], medians, rtol=0, atol=1e-12)
                arrays.append(array)
                provenance.append(dict(period=period, adjacency=graph, seed=seed,
                                       prediction_sha256=sha256(path.read_bytes()).hexdigest()))
            first = arrays[0]
            for array in arrays[1:]:
                for key in ['observed_visits','valid','target_index','evidence_strata','forecast_origin']:
                    np.testing.assert_array_equal(first[key], array[key])
            origins = np.searchsorted(all_dates, first['forecast_origin'])
            np.testing.assert_array_equal(all_dates[origins], first['forecast_origin'])
            target, neighbor = evidence(origins)
            y = first['observed_visits']; valid = first['valid']
            np.testing.assert_array_equal(y, extended[origins[:,None]+np.arange(7)][:,:,targets])
            strata = np.broadcast_to(((target > medians[0]).astype('int8')*2 +
                                      (neighbor > medians[1]).astype('int8'))[:,None,:], y.shape)
            np.testing.assert_array_equal(strata, first['evidence_strata'])
            xb, yb = [np.searchsorted(c, a, side='left') for c, a in zip(cuts, (target, neighbor))]
            gains = np.stack([np.abs(a['identity_prediction']-y)-np.abs(a['graph_prediction']-y) for a in arrays])
            gain_mean = np.mean(gains.astype(np.float64), axis=0)
            nx, ny = len(cuts[0])+1, len(cuts[1])+1
            weighted_sum = 0.; count = 0
            for iy in range(ny):
                for ix in range(nx):
                    mask = valid & ((xb == ix) & (yb == iy))[:,None,:]
                    n = int(mask.sum()); count += n
                    seed_gains = gains[:,mask].mean(1, dtype=np.float64) if n else np.full(5,np.nan)
                    mean = float(seed_gains.mean())
                    if n: weighted_sum += n*mean
                    row = dict(period=period,adjacency=graph,x_bin=ix,y_bin=iy,
                               events=n,target_POIs=int(mask.any((0,1)).sum()),
                               MAE_gain_mean=mean,MAE_gain_std=float(seed_gains.std(ddof=1)),
                               display_cell=bool(n>=30))
                    for seed, value in zip(range(42,47),seed_gains):row[f'seed{seed}']=float(value)
                    rows.append(row)
            overall = float(gain_mean[valid].mean())
            assert count == int(valid.sum())
            assert abs(weighted_sum/count-overall) < 1e-12
            reports.append(dict(period=period,adjacency=graph,events=count,target_POIs=len(targets),
                                overall_gain=overall,x_cutpoints=cuts[0].tolist(),y_cutpoints=cuts[1].tolist(),
                                grid_shape=[ny,nx],median_strata_match=True,weighted_reconstruction_error=abs(weighted_sum/count-overall)))
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(args.out/'cells.csv',index=False)
    report=dict(status='complete',panels=reports,provenance=provenance,
                cutpoints='Distinct validation sextiles; ties enter lower bin; open upper tail',
                seeds=list(range(42,47)),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='complete',panels=reports)),flush=True)


if __name__ == '__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--extended',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
