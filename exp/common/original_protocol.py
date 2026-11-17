"""Forecast the approved 9,884-POI panel with date-disjoint target sets."""
from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn


SETTINGS = ((14, 7), (28, 14), (28, 7), (28, 28))
TRAIN_END, VAL_END = 57, 64
SEED = 42
GRAPH_NAMES = ("Identity", "NAICS-2", "NAICS-4", "Top category",
               "Distance KNN", "Distance RBF", "Visit correlation")
GRAPH_LR = {"NAICS-2": .0005}


@dataclass
class Panel:
    raw: np.ndarray
    dates: list
    attrs: pd.DataFrame
    manifest: dict

    @property
    def filled(self):
        return np.nan_to_num(self.raw, nan=0.)


@dataclass
class Windows:
    origins: np.ndarray
    history_idx: np.ndarray
    target_idx: np.ndarray
    valid: np.ndarray
    y: np.ndarray
    n_train: int
    n_val: int
    history_mean: np.ndarray
    zero_ratio: np.ndarray
    mean: float
    std: float

    @property
    def test_start(self):
        return self.n_train + self.n_val


def load_panel(bundle, expected_sha):
    digest = sha256(bundle.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise ValueError(f"Dataset SHA-256 mismatch: {digest}")
    with ZipFile(bundle) as archive:
        raw = np.load(BytesIO(archive.read("visits.npy")), allow_pickle=False)
        dates = pd.read_csv(BytesIO(archive.read("dates.csv"))).date.tolist()
        attrs = pd.read_csv(BytesIO(archive.read("poi_attributes.csv")))
        manifest = json.loads(archive.read("dataset_manifest.json"))
    assert dates == pd.date_range("2020-10-01", "2020-12-31").strftime("%Y-%m-%d").tolist()
    assert raw.shape == (len(dates), len(attrs)) == (92, 9884) and attrs.placekey.is_unique
    observed = np.isfinite(raw)
    assert observed.sum() == manifest["observed_poi_days"] == 905918
    assert (~observed).sum() == manifest["missing_poi_days"] == 3410
    manifest["bundle_sha256"] = digest
    manifest["target_dates"] = {
        "train": [dates[0], dates[TRAIN_END-1]],
        "validation": [dates[TRAIN_END], dates[VAL_END-1]],
        "test": [dates[VAL_END], dates[-1]],
    }
    manifest["missing_policy"] = "Missing lag values enter as zero placeholders; missing labels are excluded."
    manifest["target_policy"] = "Origin follows chronological split; scored target dates are disjoint."
    return Panel(raw, dates, attrs, manifest)


def make_windows(raw, window, horizon):
    origins = np.arange(window, len(raw)-horizon+1)
    histories = origins[:, None] + np.arange(-window, 0)
    targets = origins[:, None] + np.arange(horizon)
    n_train = int(np.searchsorted(origins, TRAIN_END))
    stop_val = int(np.searchsorted(origins, VAL_END))
    y = raw[targets]
    valid = np.isfinite(y)
    valid[:n_train] &= (targets[:n_train, :, None] < TRAIN_END)
    valid[n_train:stop_val] &= (targets[n_train:stop_val, :, None] < VAL_END)
    hx = raw[histories[:n_train]]
    count = np.isfinite(hx).sum(axis=(0, 1))
    assert (count > 0).all() and valid[:n_train].any(axis=(0, 2)).all()
    historical_mean = np.nansum(hx, axis=(0, 1)) / count
    zero_ratio = (hx == 0).sum(axis=(0, 1)) / count
    assert valid[stop_val:].all() and stop_val < len(origins)
    return Windows(origins, histories, targets, valid, np.nan_to_num(y, nan=0.),
                   n_train, stop_val-n_train, historical_mean.astype(np.float32),
                   zero_ratio, float(np.nanmean(hx)), float(np.nanstd(hx)))


def group_codes(attrs):
    from exp.common.adjacency import clean_naics_code
    naics = attrs.naics_code.map(clean_naics_code)
    assert naics.str.len().ge(4).all()
    return {
        "NAICS-2": pd.factorize(naics.str[:2])[0],
        "NAICS-4": pd.factorize(naics.str[:4])[0],
        "Top category": pd.factorize(attrs.top_category)[0],
    }


def group_average(data, groups):
    total = np.zeros((data.shape[0], int(groups.max()+1)), dtype=np.float64)
    np.add.at(total.T, groups, data.T)
    return (total[:, groups] / np.bincount(groups)[groups]).astype(np.float32)


def spatial_matrix(attrs, name):
    from exp.common.adjacency import build_distance_knn_adj, build_distance_rbf_adj
    build = build_distance_knn_adj if name == "Distance KNN" else build_distance_rbf_adj
    return sparse.csr_matrix(build(attrs.latitude.to_numpy(), attrs.longitude.to_numpy()))


def correlation_matrix(training):
    def standardized(data):
        x = data - data.mean(0)
        return x / np.maximum(np.linalg.norm(x, axis=0), 1e-7)
    incomplete = np.isnan(training).any(axis=0)
    matrix = standardized(np.nan_to_num(training, nan=0.))
    adj = np.abs(matrix.T @ matrix)
    if incomplete.any():
        both_observed = training[np.isfinite(training).all(axis=1)]
        matrix = standardized(both_observed)
        cross = np.abs(matrix.T @ matrix[:, incomplete])
        adj[:, incomplete] = cross
        adj[incomplete, :] = cross.T
    adj[adj < .3] = 0
    np.fill_diagonal(adj, 1)
    adj /= adj.sum(1, keepdims=True)
    return sparse.csr_matrix(adj)


def propagate(data, adj):
    with_self = adj + sparse.eye(adj.shape[0], dtype=np.float32, format="csr")
    scale = np.asarray(with_self.sum(1)).ravel() ** -.5
    return ((with_self @ (data * scale).T).T * scale).astype(np.float32)


def graph_series(panel, groups):
    raw = panel.filled
    series = {"Identity": raw}
    for name, labels in groups.items():
        series[name] = (raw + group_average(raw, labels)) * .5
    for name in ("Distance KNN", "Distance RBF"):
        series[name] = propagate(raw, spatial_matrix(panel.attrs, name))
    series["Visit correlation"] = propagate(raw, correlation_matrix(panel.raw[:TRAIN_END]))
    return series


def static_features(attrs):
    from exp.common.static_features import build_static_features, frequency_group
    numeric, info, *ids = build_static_features(attrs, None)
    subcategory, vocab = frequency_group(attrs.sub_category, 1)
    features = np.column_stack([numeric, *ids, subcategory])
    assert features.shape == (len(attrs), 10)
    return features.astype(np.float32), {**info, "subcategory_vocab": vocab}


class GraphGRU(nn.Module):
    """The pinned GraphGRU recurrence with batched nodes and pre-applied adjacency."""

    def __init__(self, horizon, static=None):
        super().__init__()
        self.register_buffer("static", None if static is None else torch.as_tensor(static))
        sizes = [] if static is None else [int(static[:, i].max())+1 for i in range(4, static.shape[1])]
        self.embeddings = nn.ModuleList(nn.Embedding(n, 8) for n in sizes)
        self.project = nn.Linear(1 + (4 + 8*len(sizes) if sizes else 0), 32)
        self.gru = nn.GRU(32, 32)
        self.last = nn.Linear(1, 32)
        self.decode = nn.Linear(32, horizon)

    def encode_static(self):
        return torch.cat([self.static[:, :4]] +
                         [layer(self.static[:, 4+i].long())
                          for i, layer in enumerate(self.embeddings)], dim=1)

    def forecast_projected(self, encoded, own_last):
        b, t, n, _ = encoded.shape
        seq = encoded.transpose(0, 1).reshape(t, b*n, 32)
        _, state = self.gru(seq)
        logits = self.decode(state[0].reshape(b, n, 32) + self.last(own_last[..., None]))
        return logits.transpose(1, 2)

    def forward(self, visits, own):
        encoded = nn.functional.linear(visits[..., None], self.project.weight[:, :1], self.project.bias)
        if self.static is not None:
            encoded = encoded + nn.functional.linear(self.encode_static(), self.project.weight[:, 1:])
        return self.forecast_projected(encoded, own[:, -1])


def tensor(data, device):
    return torch.as_tensor(data, device=device)


def save_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def train_model(make_model, features, windows, device, lr, batch_size, epochs,
                normalized=True, objective="MSE", weight_decay=0., clip=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mean, std = (windows.mean, windows.std) if normalized else (0., 1.)
    inputs = [tensor((arr-mean)/std, device) for arr in features]
    target = tensor((windows.y-mean)/std, device)
    mask = tensor(windows.valid, device)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_epoch = float("inf"), 0
    history = []

    def batch_loss(indices):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(*(x[indices] for x in inputs))
        delta = pred.float() - target[indices]
        error = delta.square() if objective == "MSE" else delta.abs()
        return error.masked_fill(~mask[indices], 0).sum()

    for epoch in range(1, epochs+1):
        start = time.monotonic()
        model.train()
        costs = []
        order = rng.permutation(windows.n_train)
        for sub in np.array_split(order, int(np.ceil(len(order)/batch_size))):
            idx = tensor(sub, device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(idx) / mask[idx].sum()
            scaler.scale(loss).backward()
            if clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            costs.append(loss.detach())
        model.eval()
        with torch.no_grad():
            ids = torch.arange(windows.n_train, windows.test_start, device=device)
            score = float(sum(batch_loss(i) for i in ids.split(batch_size)) / mask[ids].sum())
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite validation loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": float(torch.stack(costs).mean()),
                        "validation_loss": score, "seconds": round(time.monotonic()-start, 3)})
        if score < best:
            best, best_epoch = score, epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch}: validation {score:.5f}, {history[-1]['seconds']:.2f}s", flush=True)
        if epoch-best_epoch >= 10:
            break
    model.load_state_dict(best_weights)
    model.eval()
    preds = []
    with torch.no_grad():
        for idx in torch.arange(windows.test_start, len(windows.origins), device=device).split(batch_size):
            preds.append(model(*(x[idx] for x in inputs)).cpu().float().numpy())
    return model, np.maximum(np.concatenate(preds)*std + mean, 0), {
        "best_epoch": best_epoch, "validation_loss": best, "epochs": history,
        "learning_rate": lr, "batch_size": batch_size, "seed": SEED, "loss": f"masked {objective}",
    }


def ridge(w, hx):
    x_train = hx[:w.n_train].transpose(2, 0, 1).astype(np.float64)
    x_test = hx[w.test_start:].transpose(2, 0, 1)
    pred = np.empty_like(w.y[w.test_start:])
    ident = np.eye(x_train.shape[-1]) * 100.
    for h in range(w.y.shape[1]):
        masks, groups = np.unique(w.valid[:w.n_train, h, :].T, axis=0, return_inverse=True)
        for i, keep in enumerate(masks):
            select = groups == i
            x = x_train[select][:, keep, :]
            y = w.y[:w.n_train, h, :][keep][:, select].T.astype(np.float64)
            x_center = x.mean(1, keepdims=True)
            x -= x_center
            y_center = y.mean(1, keepdims=True)
            y -= y_center
            transpose = x.transpose(0, 2, 1)
            weights = np.linalg.solve(transpose @ x + ident, transpose @ y[..., None])
            pred[:, h, select] = ((x_test[select]-x_center) @ weights).squeeze(-1).T + y_center.T
    return np.maximum(pred, 0)


def model_key(model, adj):
    return model.lower().replace(" ", "_") + "__" + adj.lower().replace(" ", "_")


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("graph", "static"), required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=6)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    from exp.common.lstm import LSTMForecast
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    panel = load_panel(args.bundle, args.sha256)
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out/"dataset_manifest.json", panel.manifest)
    print(f"Verified panel: {panel.raw.shape}, {panel.dates[0]} through {panel.dates[-1]}, {args.sha256}", flush=True)
    groups = group_codes(panel.attrs)
    series = graph_series(panel, groups) if args.stage == "graph" else None
    static = static_features(panel.attrs)[0] if args.stage == "static" else None
    for t, h in SETTINGS if args.stage == "graph" else [(14, 7)]:
        w = make_windows(panel.raw, t, h)
        run = args.out / f"T{t}_H{h}"
        run.mkdir(exist_ok=True)
        save_json(run/"split.json", {
            "T": t, "H": h, "train_origins": w.n_train, "validation_origins": w.n_val,
            "test_origins": len(w.origins)-w.test_start,
            "training_events": int(w.valid[:w.n_train].sum()),
            "validation_events": int(w.valid[w.n_train:w.test_start].sum()),
            "test_events": int(w.valid[w.test_start:].sum()),
            "test_origin_dates": [panel.dates[j] for j in w.origins[w.test_start:]],
            "last_lead_training_origins": int(w.valid[:w.n_train, -1, :].any(axis=1).sum()),
            "input_mean": w.mean, "input_std": w.std,
        })
        own = panel.filled[w.history_idx]
        if args.stage == "static":
            models = [("GNN-static", "Identity")]
        else:
            models = [("Historical average", "None"), ("AR-ridge", "None"), ("LSTM", "None")]
            models.extend(("EvolveGCN", name) for name in GRAPH_NAMES)
        for model, adj in models:
            key = model_key(model, adj)
            report = run / f"{key}.json"
            if report.exists():
                assert (run/f"{key}.npy").is_file()
                print(f"T{t} H{h} {key}: existing verified-run output", flush=True)
                continue
            print(f"T{t} H{h} {model} {adj}: fitting", flush=True)
            fit = {}
            if model == "Historical average":
                pred = np.broadcast_to(w.history_mean, w.y[w.test_start:].shape).copy()
            elif model == "AR-ridge":
                pred = ridge(w, own)
            elif model == "LSTM":
                _, pred, fit = train_model(
                    lambda: LSTMForecast(len(panel.attrs), 128, h), [own], w,
                    device, .001, 32, args.epochs, normalized=False)
            else:
                model_static = static if model == "GNN-static" else None
                graph_input = own if series is None else series[adj][w.history_idx]
                fitted, pred, fit = train_model(
                    lambda: GraphGRU(h, model_static), [graph_input, own], w,
                    device, GRAPH_LR.get(adj, .001), args.batch_size, args.epochs)
                if model == "GNN-static" or (t, h, adj) == (28, 14, "NAICS-4"):
                    torch.save(fitted.state_dict(), run/f"{key}.pt")
            assert pred.shape == w.y[w.test_start:].shape and np.isfinite(pred).all()
            np.save(run/f"{key}.npy", pred.astype(np.float32), allow_pickle=False)
            mae = float(np.abs(pred - w.y[w.test_start:]).mean())
            save_json(report, {"model": model, "adjacency": adj, "T": t, "H": h, "test_mae": mae, **fit})
            print(f"T{t} H{h} {model} {adj}: MAE {mae:.5f}", flush=True)


if __name__ == "__main__":
    main()
