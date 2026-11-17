"""Run the pinned MTGNN temporal model with exact sparse fixed-graph propagation."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import torch
from torch import nn

from exp.common.original_protocol import SETTINGS, group_codes, load_panel, make_windows, model_key, save_json, spatial_matrix, train_model


CONFIGS = {
    "Identity": (.005, .3),
    "NAICS-2": (.001, .1),
    "Top category": (.001, .1),
    "Distance KNN": (.001, .1),
}


class FixedMixProp(nn.Module):
    def __init__(self, original, multiply):
        super().__init__()
        self.alpha = original.alpha
        self.depth = original.gdep
        self.mlp = original.mlp
        self.multiply = multiply

    def forward(self, x, _):
        h = x
        layers = [h]
        for _ in range(self.depth):
            h = self.alpha*x + (1-self.alpha)*self.multiply(h)
            layers.append(h)
        return self.mlp(torch.cat(layers, dim=1))


class WindowMTGNN(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.model = architecture

    def forward(self, x):
        return self.model(x.transpose(1, 2).unsqueeze(1)).squeeze(-1)


def clique_operator(group, device):
    codes = torch.as_tensor(group, device=device)
    n_groups = int(group.max())+1
    sizes = torch.as_tensor(np.bincount(group), device=device, dtype=torch.float32)

    def multiply(x):
        totals = torch.zeros((*x.shape[:2], n_groups, x.shape[-1]),
                             device=device, dtype=torch.float32)
        totals.index_add_(2, codes, x.float())
        means = (totals / sizes[None, None, :, None])[:, :, codes, :]
        return ((x.float()+means)*.5).to(x.dtype)

    return multiply


def sparse_operator(adjacency, device):
    matrix = (adjacency + sparse.eye(adjacency.shape[0], dtype=np.float32, format="csr")).tocsr()
    matrix = (sparse.diags(1 / np.asarray(matrix.sum(1)).ravel()) @ matrix).tocsr()
    matrix.sort_indices()
    kernel = torch.sparse_csr_tensor(matrix.indptr, matrix.indices, matrix.data,
                                     size=matrix.shape, dtype=torch.float32, device=device)

    def multiply(x):
        b, c, n, l = x.shape
        flat = x.float().permute(2, 0, 1, 3).reshape(n, -1)
        with torch.autocast(device.type, enabled=False):
            value = torch.sparse.mm(kernel, flat)
        return value.reshape(n, b, c, l).permute(1, 2, 0, 3).contiguous().to(x.dtype)

    return multiply


def replace_graph_operators(model, operators):
    for modules, multiply in zip((model.gconv1, model.gconv2), operators):
        for i, original in enumerate(modules):
            modules[i] = FixedMixProp(original, multiply)
    return model


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    sys.path[:0] = [str(args.repo/"external/MTGNN"), str(args.repo)]
    from net import gtnet
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    panel = load_panel(args.bundle, args.sha256)
    groups = group_codes(panel.attrs)
    knn = spatial_matrix(panel.attrs, "Distance KNN")
    forward, reverse = sparse_operator(knn, device), sparse_operator(knn.T, device)
    graphs = {"Identity": (lambda x: x, lambda x: x), "Distance KNN": (forward, reverse)}
    for name in ("NAICS-2", "Top category"):
        multiply = clique_operator(groups[name], device)
        graphs[name] = (multiply, multiply)
    print(f"Verified MTGNN panel {panel.raw.shape}, SHA-256 {args.sha256}", flush=True)

    for t, h in SETTINGS:
        run = args.out / f"T{t}_H{h}"
        w = make_windows(panel.raw, t, h)
        x = panel.filled[w.history_idx]
        for name, (lr, dropout) in CONFIGS.items():
            key = model_key("MTGNN", name)
            report = run/f"{key}.json"
            if report.exists():
                assert (run/f"{key}.npy").is_file()
                continue
            print(f"T{t} H{h} MTGNN {name}: fitting", flush=True)

            def make_model():
                network = gtnet(True, False, 2, len(panel.attrs), device,
                                predefined_A=torch.empty((0, 0), device=device),
                                dropout=dropout, seq_length=t, in_dim=1, out_dim=h)
                return WindowMTGNN(replace_graph_operators(network, graphs[name]))

            _, pred, fit = train_model(make_model, [x], w, device, lr, args.batch_size,
                                       args.epochs, objective="MAE", weight_decay=.0001, clip=5.)
            assert pred.shape == w.y[w.test_start:].shape and np.isfinite(pred).all()
            np.save(run/f"{key}.npy", pred.astype(np.float32), allow_pickle=False)
            mae = float(np.abs(pred-w.y[w.test_start:]).mean())
            save_json(report, {"model": "MTGNN", "adjacency": name, "T": t, "H": h,
                               "test_mae": mae, "dropout": dropout, **fit})
            print(f"T{t} H{h} MTGNN {name}: MAE {mae:.5f}", flush=True)


if __name__ == "__main__":
    main()
