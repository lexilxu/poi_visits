"""Controls shared by the revision experiments; no fitted neural results assumed."""
import numpy as np
from scipy.optimize import minimize


def matched_neighbors(coarse, fine, k, seed):
    """Equal row degree across semantic and random graphs, excluding self.

    A row uses min(k, number of fine-class peers). Sampling is uniform without
    replacement. Singleton fine classes stay isolated in every control.
    """
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    n = len(coarse)
    rng = np.random.default_rng(seed)
    pools = [{v: np.flatnonzero(labels == v) for v in np.unique(labels)}
             for labels in (coarse, fine)]
    result = {name: np.full((n, k), -1, dtype=np.int32)
              for name in ('NAICS-2', 'NAICS-4', 'Random')}
    for j in range(n):
        a, b = pools[0][coarse[j]], pools[1][fine[j]]
        a, b = a[a != j], b[b != j]
        if not np.isin(b, a).all():
            raise ValueError('Fine groups must be nested in coarse groups')
        d = min(k, len(b))
        if d:
            result['NAICS-2'][j, :d] = rng.choice(a, d, replace=False)
            result['NAICS-4'][j, :d] = rng.choice(b, d, replace=False)
            random = rng.choice(n-1, d, replace=False)
            result['Random'][j, :d] = random + (random >= j)
    return result


def propagate_neighbors(history, neighbors):
    """Half own count plus half mean neighbor count; identity for isolates."""
    out = np.array(history, dtype=float, copy=True)
    for j, peers in enumerate(neighbors):
        peers = peers[peers >= 0]
        if len(peers):
            out[..., j] = .5 * (history[..., j] + history[..., peers].mean(-1))
    return out


def affine_calibration(pred, observed):
    """Fit max(0, a*p+c), a>=0, to validation MAE only.

    Bounds are fixed before test inspection; identity is an explicit candidate.
    The intercept bound uses validation outcomes. Return fitted coefficients and
    optimization diagnostics so an unsuccessful optimization is never hidden.
    """
    x, y = np.asarray(pred, float).ravel(), np.asarray(observed, float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if not len(x):
        raise ValueError('Calibration requires observed validation events')
    bound = max(1., float(np.quantile(y, .99)))
    objective = lambda z: float(np.abs(np.maximum(0., z[0]*x+z[1])-y).mean())
    cov = float(np.mean((x-x.mean())*(y-y.mean())))
    slope = max(0., cov/max(float(x.var()), 1e-12))
    start = np.array([min(slope, 20.), np.clip(y.mean()-slope*x.mean(), -bound, bound)])
    candidates = [(objective([1., 0.]), np.array([1., 0.]), True, 'identity')]
    for initial in (np.array([1., 0.]), start):
        fit = minimize(objective, initial, method='Powell', bounds=((0., 20.), (-bound, bound)),
                       options={'maxiter':100, 'xtol':1e-7, 'ftol':1e-7})
        if fit.success and np.isfinite(fit.fun):
            candidates.append((float(fit.fun), fit.x, True, 'Powell'))
    loss, coef, success, method = min(candidates, key=lambda item: item[0])
    return {'a':float(coef[0]), 'c':float(coef[1]), 'validation_MAE':loss,
            'method':method, 'success':success, 'a_bounds':[0.,20.],
            'c_bounds':[-bound,bound], 'events':len(x)}


def matched_far_threshold(validation_zero_scores, far):
    """Conservative threshold calibrated on validation zeros, grouping ties."""
    x = np.asarray(validation_zero_scores, float)
    if not len(x) or not 0 <= far < 1:
        raise ValueError('Need validation zero events and FAR in [0,1)')
    score = np.sort(x)[::-1]
    allowed = int(np.floor(far * len(score)))
    return float(np.nextafter(score[min(allowed, len(score)-1)], np.inf))


def grid_groups(latitude, longitude, cell_km=1.):
    """Fixed equirectangular grid anchored at 0 degrees, reference latitude 47.6.

    Used solely for observed-activity prioritization, not transport ridership.
    """
    lat, lon = np.asarray(latitude), np.asarray(longitude)
    xy = np.column_stack((lon*111.320*np.cos(np.deg2rad(47.6)), lat*110.574))
    _, groups = np.unique(np.floor(xy/cell_km).astype(np.int64), axis=0, return_inverse=True)
    return groups


def spatial_coverage(y, pred, groups, fraction):
    """Coverage per origin/lead at a fixed fraction of occupied grid cells."""
    n = int(groups.max())+1
    observed = np.zeros((*y.shape[:-1], n))
    forecast = np.zeros_like(observed)
    for j in range(n):
        members = groups == j
        observed[..., j] = np.nansum(y[..., members], axis=-1)
        forecast[..., j] = np.sum(pred[..., members], axis=-1)
    k = max(1, int(np.ceil(fraction*n)))
    selected = np.argsort(-forecast, axis=-1, kind='stable')[..., :k]
    oracle = np.argsort(-observed, axis=-1, kind='stable')[..., :k]
    total = observed.sum(-1)
    coverage = np.divide(np.take_along_axis(observed, selected, -1).sum(-1), total,
                         out=np.full(total.shape, np.nan), where=total > 0)
    ideal = np.divide(np.take_along_axis(observed, oracle, -1).sum(-1), total,
                      out=np.full(total.shape, np.nan), where=total > 0)
    return coverage, ideal, k, n
