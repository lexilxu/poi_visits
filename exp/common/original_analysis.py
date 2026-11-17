"""Reduce the approved-panel runs to the experiment tables and figure data."""
from argparse import ArgumentParser
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from exp.common.original_protocol import GraphGRU, SEED, SETTINGS, TRAIN_END, group_average, group_codes, load_panel, make_windows, model_key, save_json, static_features, tensor, train_model
from exp.exp1.train_mtgnn import CONFIGS


GRAPH_OPTIONS = ("Identity", "NAICS-2", "Top category", "Distance RBF",
                 "Visit correlation", "Distance KNN")
BENCHMARKS = ("Historical average", "AR-ridge", "LSTM")
ATTR_GROUPS = ("History", "Location", "Identity", "Parent", "Others")
PERTURB_GROUPS = ("All", "Recent-active", "High-mean", "Top-10 active", "Random-10",
                  "Top-10 correlated", "Aligned", "Low-mean")


def forecast(run, model, adjacency="None"):
    return np.load(run/f"{model_key(model, adjacency)}.npy", allow_pickle=False)


def fit_summary(run, model, adjacency):
    return json.loads((run/f"{model_key(model, adjacency)}.json").read_text(encoding="utf-8"))


def accuracy(y, prediction):
    error = prediction-y
    centered = y-y.mean()
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "rse": float(np.linalg.norm(error.ravel()) / np.linalg.norm(centered.ravel())),
            "corr": correlation(y, prediction)}


def correlation(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b / norm) if norm else np.nan


def low_history(w):
    eligible = w.history_mean > 0
    cutoff = float(np.quantile(w.history_mean[eligible], .25))
    return eligible & (w.history_mean <= cutoff) & (w.zero_ratio >= .70), cutoff


def diagnostic(y, pred, ha, mask):
    low_y, low_pred, low_ha = y[:, :, mask], pred[:, :, mask], ha[:, :, mask]
    pos = low_y > 0
    mass = low_y[pos].sum()
    dmr = float(low_pred[pos].sum()/mass)
    dmr_ha = float(low_ha[pos].sum()/mass)
    return {
        "lift": float((low_pred-low_ha).mean()), "dmr_model": dmr, "dmr_ha": dmr_ha,
        "delta_dmr": dmr-dmr_ha, "far_0_5": float((low_pred[~pos] >= .5).mean()),
        "mean_prediction": float(low_pred.mean()),
    }


def forecasting(panel, root, out):
    metrics, horizon, recovery, context = [], [], [], {}
    training_records = []
    for t, h in SETTINGS:
        run = root/f"T{t}_H{h}"
        w = make_windows(panel.raw, t, h)
        y = w.y[w.test_start:]
        ha = forecast(run, "Historical average")
        selected, q25 = low_history(w)
        splits = json.loads((run/"split.json").read_text(encoding="utf-8"))
        context[f"T{t}_H{h}"] = {
            **splits, "low_history_pois": int(selected.sum()),
            "positive_low_history_events": int((y[:, :, selected] > 0).sum()),
            "low_history_events": int(y[:, :, selected].size), "historical_mean_quartile": q25,
        }
        best_mtg = min(CONFIGS, key=lambda adj: fit_summary(run, "MTGNN", adj)["validation_loss"])
        other_graphs = [adj for adj in GRAPH_OPTIONS if adj != "Identity"] + ["NAICS-4"]
        best_evolve = min(other_graphs, key=lambda adj: fit_summary(run, "EvolveGCN", adj)["validation_loss"])
        models = [(m, "None", m) for m in BENCHMARKS]
        models.extend([("MTGNN", best_mtg, "MTGNN (best)"),
                       ("EvolveGCN", "Identity", "EvolveGCN (identity)"),
                       ("EvolveGCN", best_evolve, "EvolveGCN (best non-identity)")])
        ha_mae = accuracy(y, ha)["mae"]
        for model, adj, series in models:
            pred = forecast(run, model, adj)
            scores = accuracy(y, pred)
            skill = 100*(1-scores["mae"]/ha_mae)
            metrics.append({"T": t, "H": h, "model": model, "adjacency": adj, "series": series,
                            **scores, "skill_percent": skill,
                            "anomaly_corr": correlation(y-ha, pred-ha), **diagnostic(y, pred, ha, selected)})
            if t == 28:
                horizon.append({"T": t, "H": h, "series": series, "model": model,
                                "adjacency": adj, "mae": scores["mae"], "skill_percent": skill})
        if (t, h) == (14, 7):
            rows = [(m, "None") for m in BENCHMARKS]
            rows += [("MTGNN", adj) for adj in CONFIGS] + [("EvolveGCN", adj) for adj in GRAPH_OPTIONS]
            for model, adj in rows:
                pred = forecast(run, model, adj)
                recovery.append({"model": model, "adjacency": adj, **accuracy(y, pred),
                                 **diagnostic(y, pred, ha, selected)})
        for report in sorted(run.glob("*.json")):
            record = json.loads(report.read_text(encoding="utf-8"))
            if "model" in record:
                training_records.append(record)
    pd.DataFrame(metrics).to_csv(out/"multi_horizon_results.csv", index=False)
    pd.DataFrame(horizon).to_csv(out/"07_horizon_skill.csv", index=False)
    pd.DataFrame(recovery).to_csv(out/"01_recovery_activation.csv", index=False)
    save_json(out/"forecast_protocol.json", context)
    save_json(out/"model_fit_summaries.json", training_records)
    return context


def transfer(panel, root, out):
    w = make_windows(panel.raw, 28, 14)
    run = root/"T28_H14"
    mask, q25 = low_history(w)
    hist = panel.filled[w.history_idx[w.test_start:]]
    node_history = hist.mean(axis=1)
    y = w.y[w.test_start:, :, mask]
    identity = forecast(run, "EvolveGCN", "Identity")[:, :, mask]
    groups = group_codes(panel.attrs)
    stats, metadata = {}, {}
    rows, cells = [], []
    for adjacency in ("NAICS-2", "NAICS-4", "Top category"):
        pred = forecast(run, "EvolveGCN", adjacency)[:, :, mask]
        labels = groups[adjacency]
        size = np.bincount(labels)[labels]
        neighbors = (group_average(node_history, labels)*size-node_history) / np.maximum(size-1, 1)
        target = np.broadcast_to(node_history[:, None, mask], y.shape)
        neighbor = np.broadcast_to(neighbors[:, None, mask], y.shape)
        lift = pred-identity
        gain = np.abs(y-identity)-np.abs(y-pred)
        low_t, high_n = target < np.median(target), neighbor >= np.median(neighbor)
        positive = y > 0
        false = ~positive & (pred >= .5)
        attempted = low_t & high_n & (lift >= .1)
        regimes = {
            "Spurious transfer": attempted & false,
            "Substitutive transfer": attempted,
            "Confirmatory transfer": ~low_t & high_n & (lift >= .1),
            "Redundant self-history": ~low_t & (lift < .1),
            "Conservative zero": ~positive & (pred < .5),
            "Other": np.ones_like(positive),
        }
        remaining = np.ones_like(positive)
        for label, subset in regimes.items():
            keep = subset & remaining
            remaining &= ~keep
            cols = [("graph_lift", lift), ("target_evidence", target), ("neighbor_evidence", neighbor),
                    ("mean_mae_gain", gain), ("positive_event_share", positive), ("false_activation_share", false)]
            rows.append({"adjacency": adjacency, "regime": label, "n_events": int(keep.sum()),
                         **{name: float(values[keep].mean()) if keep.any() else None for name, values in cols}})
        assert not remaining.any()
        metadata[adjacency] = {"low_history_pois": int(mask.sum()), "n_events": int(y.size),
                               "positive_events": int(positive.sum()), "zero_events": int((~positive).sum()),
                               "mean_gain_positive": float(gain[positive].mean()),
                               "mean_gain_zero": float(gain[~positive].mean()),
                               "target_median": float(np.median(target)),
                               "neighbor_median": float(np.median(neighbor)),
                               "history_mean_quartile": q25}
        if adjacency == "NAICS-2":
            edges = [np.unique(np.quantile(v, np.linspace(0, 1, 7))) for v in (target, neighbor)]
            bx, by = [np.searchsorted(e[1:-1], v, side="right") for v, e in zip((target, neighbor), edges)]
            for outcome, status in [("Positive demand", positive), ("Zero demand", ~positive)]:
                for j, (loy, hiy) in enumerate(zip(edges[1][:-1], edges[1][1:])):
                    for i, (lox, hix) in enumerate(zip(edges[0][:-1], edges[0][1:])):
                        selected = status & (bx == i) & (by == j)
                        n = int(selected.sum())
                        cells.append({"outcome": outcome, "x_bin": i, "y_bin": j, "target_low": lox,
                                      "target_high": hix, "neighbor_low": loy, "neighbor_high": hiy,
                                      "n_events": n, "display_cell": n >= 30,
                                      "mean_mae_gain": float(gain[selected].mean()) if n else None})
            assert sum(c["n_events"] for c in cells) == y.size
        if adjacency == "NAICS-4":
            stats = {"mask": mask, "w": w, "hist": hist, "labels": labels, "pred": pred,
                     "y": y, "attempted": attempted, "false": false,
                     "beneficial": positive & (gain > 0),
                     "recovered": positive & (identity < .5) & (pred >= .5) & (gain > 0)}
    pd.DataFrame(rows).to_csv(out/"02_graph_transfer_regimes.csv", index=False)
    pd.DataFrame(cells).to_csv(out/"02_graph_transfer_cells.csv", index=False)
    save_json(out/"graph_transfer_summary.json", metadata)
    return stats


def perturbation(stats, root, out, device):
    w, hist, labels, pred, truth = [stats[k] for k in ("w", "hist", "labels", "pred", "y")]
    low_ids = np.flatnonzero(stats["mask"])
    harmful = stats["attempted"] & stats["false"]
    helpful = stats["attempted"] & stats["beneficial"]
    recovered = stats["attempted"] & stats["recovered"]
    assert harmful.any() and helpful.any() and recovered.any()
    selected_size = (len(hist), len(low_ids))
    original = np.empty((*selected_size, hist.shape[1]), dtype=np.float32)
    changes = np.zeros((len(PERTURB_GROUPS), *original.shape), dtype=np.float32)
    counts = np.zeros((len(PERTURB_GROUPS), *selected_size), dtype=np.int32)
    rng = np.random.default_rng(SEED)
    members = {g: np.flatnonzero(labels == g) for g in np.unique(labels)}
    for i, lag in enumerate(hist):
        means = lag.mean(0)
        centered = lag-lag.mean(0)
        lengths = np.linalg.norm(centered, axis=0)
        for j, node in enumerate(low_ids):
            group = members[labels[node]]
            neighbors = group[group != node]
            original[i, j] = (lag[:, node]+lag[:, group].mean(1))*.5
            if not len(neighbors):
                continue
            history = lag[:, neighbors]
            strength = means[neighbors]
            mid = np.median(strength)
            corr = (centered[:, node] @ centered[:, neighbors]) / np.maximum(lengths[node]*lengths[neighbors], 1e-7)
            active = np.flatnonzero(history[-7:].max(0) > 0)
            high = np.flatnonzero(strength > mid)
            if len(high) == 0:
                high = np.flatnonzero(strength >= mid)
            aligned = np.flatnonzero(((history > 0) & (lag[:, node, None] > 0)).any(0))
            selections = (np.arange(len(neighbors)), active, high, np.argsort(-strength, kind="stable")[:10],
                          rng.choice(len(neighbors), min(10, len(neighbors)), replace=False),
                          np.argsort(-corr, kind="stable")[:10], aligned, np.flatnonzero(strength < mid))
            for k, pick in enumerate(selections):
                counts[k, i, j] = len(pick)
                changes[k, i, j] = history[:, pick].sum(1) / (2*len(group))

    model = GraphGRU(14).to(device)
    checkpoint = root/"T28_H14"/f"{model_key('EvolveGCN','NAICS-4')}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    own = (hist[:, :, low_ids]-w.mean)/w.std

    def predict(adjusted):
        x = ((adjusted-w.mean)/w.std).transpose(0, 2, 1)
        with torch.no_grad():
            results = [model(tensor(x[i:i+4], device), tensor(own[i:i+4], device)).cpu().numpy()
                       for i in range(0, len(x), 4)]
        return np.maximum(np.concatenate(results)*w.std+w.mean, 0)

    np.testing.assert_allclose(predict(original), pred, atol=2e-3, rtol=1e-3)
    rows = []
    for name, delta, count in zip(PERTURB_GROUPS, changes, counts):
        other = predict(original-delta)
        drop = pred-other
        per_node = drop / np.maximum(count[:, None, :], 1)
        err = np.abs(other-truth)-np.abs(pred-truth)
        rows.append({"neighbor_group": name, "beneficial_lift": float(drop[helpful].mean()),
                     "harmful_lift": float(drop[harmful].mean()), "beneficial_per_node": float(per_node[helpful].mean()),
                     "harmful_per_node": float(per_node[harmful].mean()),
                     "beneficial_mae_effect": float(err[helpful].mean()), "harmful_mae_effect": float(err[harmful].mean()),
                     "false_activations_removed": float((other[harmful] < .5).mean()),
                     "recoveries_lost": float((other[recovered] < .5).mean())})
    pd.DataFrame(rows).to_csv(out/"03_neighbor_perturbation.csv", index=False)
    save_json(out/"perturbation_summary.json", {"attempted": int(stats["attempted"].sum()),
               "beneficial": int(helpful.sum()), "harmful_false_activations": int(harmful.sum()),
               "recovered_positives": int(recovered.sum())})


from exp.exp5.regimes import export_attribution_regimes


def attribution(panel, root, out, device, repo):
    w = make_windows(panel.raw, 14, 7)
    run = root/"T14_H7"
    static, _ = static_features(panel.attrs)
    model = GraphGRU(7, static).to(device)
    key = model_key("GNN-static", "Identity")
    model.load_state_dict(torch.load(run/f"{key}.pt", map_location=device, weights_only=True))
    model.requires_grad_(False)
    model.eval()
    model.gru.train()
    x = (panel.filled[w.history_idx[w.test_start:]]-w.mean)/w.std
    n = len(panel.attrs)
    y, pred = w.y[w.test_start:], forecast(run, "GNN-static", "Identity")
    errors = np.abs(y-pred)
    raw = np.zeros((n, len(ATTR_GROUPS)), dtype=np.float64)
    semantics = ("Top category", "Sub-category", "NAICS-2", "NAICS-4")
    descriptor_slices = (slice(5, 13), slice(45, 53), slice(13, 21), slice(21, 29))
    attr_sums, dominates, dominant_error = [np.zeros(4, dtype=np.float64) for _ in range(3)]
    events = 0
    static_tensor = model.encode_static().detach()
    weights = model.project.weight
    static_projection = torch.nn.functional.linear(static_tensor, weights[:, 1:])
    inference_batch = fit_summary(run, "GNN-static", "Identity")["batch_size"]
    for i in range(0, len(x), inference_batch):
        hx = tensor(x[i:i+inference_batch], device)
        batch = len(hx)
        stop = i+batch
        # Use the exact factored forward pass from training. Apply the chain
        # rule through the linear projection to retain individual lag scores.
        projected = (torch.nn.functional.linear(hx[..., None], weights[:, :1], model.project.bias)
                     + static_projection).detach().requires_grad_(True)
        last = hx[:, -1, :].detach().requires_grad_(True)
        encoded = torch.cat((hx[..., None], static_tensor[None, None].expand(batch, 14, -1, -1)), dim=-1)
        logits = model.forecast_projected(projected, last)
        np.testing.assert_allclose(np.maximum(logits.detach().cpu().numpy()*w.std+w.mean, 0),
                                   pred[i:stop], atol=2e-3, rtol=1e-3)
        for h in range(7):
            dprojected, dlast = torch.autograd.grad(logits[:, h].sum(), (projected, last), retain_graph=h < 6)
            gradient = dprojected @ weights
            gradient[:, -1, :, 0] += dlast
            scores = (encoded*gradient).abs().sum(dim=1).cpu().numpy()
            group = np.stack((scores[..., 0], scores[..., 1:3].sum(-1),
                              scores[..., 5:37].sum(-1)+scores[..., 45:53].sum(-1),
                              scores[..., 37:45].sum(-1), scores[..., 3:5].sum(-1)), axis=-1)
            raw += group.sum(0)
            descriptors = np.stack([scores[..., s].sum(-1) for s in descriptor_slices], axis=-1).reshape(-1, 4)
            eligible = descriptors.sum(1) > 0
            desc = descriptors[eligible]
            attr_sums += (desc / group.reshape(-1, 5)[eligible].sum(1)[:, None]).sum(0)
            winners = desc.argmax(1)
            dominates += np.bincount(winners, minlength=4)
            dominant_error += np.bincount(winners, weights=errors[i:stop, h].ravel()[eligible], minlength=4)
            events += int(eligible.sum())
        print(f"Attribution windows: {stop}/{len(x)}", flush=True)
    shares = raw / raw.sum(axis=1, keepdims=True)
    assert np.isfinite(shares).all() and np.allclose(shares.sum(1), 1)
    mae = errors.mean((0, 1))
    export_attribution_regimes(panel.attrs, shares, mae, out)
    pd.DataFrame({"descriptor": semantics, "mean_share": attr_sums/events, "dominance_rate": dominates/events,
                  "dominant_mae": np.divide(dominant_error, dominates, out=np.full(4, np.nan), where=dominates>0),
                  "n_dominant": dominates.astype(int)}).to_csv(out/"semantic_attribution.csv", index=False)

    from exp.exp6.categories import EMBEDDED_CATEGORY_GROUPS
    parent_groups = EMBEDDED_CATEGORY_GROUPS
    embedded = panel.attrs.parent_placekey.notna().to_numpy()
    parent_rows = []
    for label, select in [(name, embedded & panel.attrs.top_category.isin(cats).to_numpy())
                          for name, cats in parent_groups.items()] + [("All embedded POIs", embedded)]:
        if select.any():
            parent = float(shares[select, 3].mean())
            parent_rows.append({"group": label, "n_pois": int(select.sum()), "parent_share": parent,
                                "own_share": 1-parent, "gap": 2*parent-1, "mae": float(mae[select].mean())})
    pd.DataFrame(parent_rows).to_csv(out/"parent_attribution.csv", index=False)
    save_json(out/"attribution_summary.json", {"T": 14, "H": 7, "nodes": n,
               "regimes": [f"{group}-driven" for group in ATTR_GROUPS],
               "regime_assignment": "Maximum feature-group z-score across POIs",
               "test_windows": len(x), "eligible_semantic_events": events,
               "model": "Full-feature GNN-static with encoded sub-category and identity adjacency"})


def density(panel, root, out, device, epochs):
    train = panel.raw[:TRAIN_END]
    mask = np.isfinite(train)
    ratio = (train > 0).sum(0) / mask.sum(0)
    recent = (train[-14:] > 0).sum(0) / mask[-14:].sum(0)
    eligible = (train[:-1] > 0) & mask[1:]
    persist = (eligible & (train[1:] > 0)).sum(0) / np.maximum(eligible.sum(0), 1)
    ranks = {"Dense": ratio, "Recent-dense": recent, "Persist-dense": ratio*persist}
    jobs = [("Full", np.arange(train.shape[1]))]
    for label, score in ranks.items():
        for pct in (10, 25, 50, 75):
            retained = int(np.ceil((100-pct)*len(score)/100))
            jobs.append((f"{label}{pct}", np.sort(np.argsort(-score, kind="stable")[:retained])))
    records = []
    for label, ids in jobs:
        w = make_windows(panel.raw[:, ids], 28, 14)
        lag = panel.filled[:, ids][w.history_idx]
        y = w.y[w.test_start:]
        print(f"Density {label}: {len(ids):,} POIs", flush=True)
        if label == "Full":
            pred = forecast(root/"T28_H14", "EvolveGCN", "Identity")
        else:
            _, pred, _ = train_model(lambda: GraphGRU(14), [lag, lag], w, device, .001, 6, epochs)
        a = accuracy(y, pred)
        ha_mae = float(np.abs(y - w.history_mean).mean())
        records.append({"subset": label, "n_pois": len(ids), "nonzero_ratio": float(ratio[ids].mean()),
                        **a, "ha_mae": ha_mae, "skill_percent": 100*(1-a["mae"]/ha_mae)})
        pd.DataFrame(records).to_csv(out/"density_sensitivity.csv", index=False)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--density", action="store_true")
    p.add_argument("--attribution-only", action="store_true",
                   help="Recompute attribution from the saved GNN-static fit.")
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    panel = load_panel(args.bundle, args.sha256)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out/"dataset_manifest.json", panel.manifest)
    if args.attribution_only:
        attribution(panel, args.run, args.out, device, args.repo)
        return
    context = forecasting(panel, args.run, args.out)
    print(f"Forecasting aggregates complete: {list(context)}", flush=True)
    grouped = transfer(panel, args.run, args.out)
    print("Graph-transfer tables and cells complete", flush=True)
    perturbation(grouped, args.run, args.out, device)
    print("Perturbation table complete", flush=True)
    attribution(panel, args.run, args.out, device, args.repo)
    print("Attribution aggregates complete", flush=True)
    if args.density:
        density(panel, args.run, args.out, device, 50)
        print("Density sensitivity complete", flush=True)


if __name__ == "__main__":
    main()
