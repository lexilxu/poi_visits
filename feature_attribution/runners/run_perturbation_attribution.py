import os
import argparse
import numpy as np
import pandas as pd
import torch

from feature_attribution.src.io_utils import (
    load_yaml,
    ensure_output_dirs,
    load_benchmark_split,
    load_poi_meta,
    get_device,
    add_project_root_to_path,
    save_dataframe,
)
from feature_attribution.src.metrics import rmse
from feature_attribution.src.attribution_methods import (
    lag_occlusion,
    group_occlusion,
    lag_permutation,
)


def load_model_forward(cfg, model_key, device):
    model_cfg = cfg["models"][model_key]
    model_type = model_cfg["model_type"]

    if model_type == "lstm":
        from runners.run_lstm import LSTMForecast

        split = load_benchmark_split(cfg["data"]["benchmark_split_path"])
        x_train = split["x_train"]
        y_train = split["y_train"]

        _, P, N = x_train.shape
        H = y_train.shape[1]

        model = LSTMForecast(
            num_nodes=N,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            horizon=H,
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.0),
        ).to(device)

        state = torch.load(model_cfg["checkpoint_path"], map_location=device)
        model.load_state_dict(state)
        model.eval()

        return lambda x: model(x)

    if model_type == "evolvegcn_style_static":
        from runners.run_evolvegcn import GraphGRUForecast, normalize_adj_numpy

        data = np.load(model_cfg["data_path"], allow_pickle=True)

        x_train = data["x_train"]
        y_train = data["y_train"]

        _, P, N = x_train.shape
        H = y_train.shape[1]

        model = GraphGRUForecast(
            num_nodes=N,
            horizon=H,
            hidden_dim=model_cfg.get("hidden_dim", 16),
        ).to(device)

        state = torch.load(model_cfg["checkpoint_path"], map_location=device)
        model.load_state_dict(state)
        model.eval()

        adj = data["adj"]
        static_adj = torch.tensor(
            normalize_adj_numpy(adj),
            dtype=torch.float32,
            device=device,
        )

        return lambda x: model(
            x=x,
            adjacency_mode="static",
            static_adj=static_adj,
            sample_positions=None,
            edge_data=None,
        )

    raise NotImplementedError(
        f"Perturbation loader for model_type={model_type} is not implemented yet."
    )


def build_category_groups(poi_meta, category_col, max_categories=20):
    if category_col not in poi_meta.columns:
        raise ValueError(f"Column {category_col} not found")

    counts = poi_meta[category_col].value_counts(dropna=False)
    selected = counts.head(max_categories).index.tolist()

    groups = {}

    for cat in selected:
        idx = poi_meta.index[poi_meta[category_col] == cat].tolist()
        groups[str(cat)] = idx

    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="feature_attribution/configs/attribution_config.yaml")
    parser.add_argument("--model_key", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ensure_output_dirs(cfg)
    add_project_root_to_path(cfg["project"].get("root_dir", "."))

    device = get_device(cfg["attribution"].get("device", "cuda:0"))

    split = load_benchmark_split(cfg["data"]["benchmark_split_path"])
    poi_meta = load_poi_meta(
        cfg["data"]["poi_meta_path"],
        cfg["data"].get("location_table_path", None),
    )

    x_test = split["x_test"]
    y_test = split["y_test"]

    batch_size = cfg["attribution"].get("batch_size", 8)
    max_batches = cfg["perturbation"].get("max_batches", 5)
    n = min(len(x_test), batch_size * max_batches)

    x = torch.tensor(x_test[:n], dtype=torch.float32, device=device)
    y_np = y_test[:n]

    model_forward = load_model_forward(cfg, args.model_key, device)

    with torch.no_grad():
        pred = model_forward(x)
        pred_np = pred.detach().cpu().numpy()

    model_name = cfg["models"][args.model_key]["model_name"]
    summary_dir = cfg["outputs"]["summary_dir"]

    rows = []

    base_rmse = rmse(y_np, pred_np)
    rows.append({
        "model": model_name,
        "perturbation_type": "none",
        "group": "baseline",
        "effect": 0.0,
        "rmse": base_rmse,
        "delta_rmse": 0.0,
    })

    mask_kind = cfg["perturbation"].get("mask_value", "mean")
    if mask_kind == "mean":
        mask_values = x.mean(dim=(0, 1))
        scalar_mask = x.mean()
    elif mask_kind == "zero":
        mask_values = torch.zeros(x.shape[2], device=device)
        scalar_mask = torch.tensor(0.0, device=device)
    else:
        raise ValueError(f"Unknown mask value: {mask_kind}")

    if cfg["perturbation"]["lag_occlusion"]["enabled"]:
        lag_effects = lag_occlusion(
            model_forward=model_forward,
            x=x,
            target_pred=pred,
            mask_value=scalar_mask,
        )

        for lag_idx, effect in enumerate(lag_effects, start=1):
            rows.append({
                "model": model_name,
                "perturbation_type": "lag_occlusion",
                "group": f"lag_{lag_idx}",
                "effect": effect,
                "rmse": np.nan,
                "delta_rmse": np.nan,
            })

    if cfg["perturbation"]["category_occlusion"]["enabled"]:
        category_col = cfg["perturbation"]["category_occlusion"].get("category_col", "top_category")
        max_categories = cfg["perturbation"]["category_occlusion"].get("max_categories", 20)

        groups = build_category_groups(
            poi_meta=poi_meta,
            category_col=category_col,
            max_categories=max_categories,
        )

        group_rows = group_occlusion(
            model_forward=model_forward,
            x=x,
            target_pred=pred,
            groups=groups,
            mask_values=mask_values,
        )

        for r in group_rows:
            rows.append({
                "model": model_name,
                "perturbation_type": "category_occlusion",
                "group": r["group"],
                "num_pois": r["num_pois"],
                "effect": r["effect"],
                "rmse": np.nan,
                "delta_rmse": np.nan,
            })

    perm_rows = lag_permutation(
        model_forward=model_forward,
        x=x,
        y_true=y_np,
        metric_fn=rmse,
        random_state=0,
    )

    for r in perm_rows:
        rows.append({
            "model": model_name,
            "perturbation_type": "lag_permutation",
            "group": f"lag_{r['lag']}",
            "effect": np.nan,
            "rmse": r["permuted_score"],
            "delta_rmse": r["delta"],
        })

    out = pd.DataFrame(rows)

    save_dataframe(
        os.path.join(summary_dir, f"{model_name}_perturbation_summary.csv"),
        out,
    )

    print("Saved perturbation summary for", model_name)


if __name__ == "__main__":
    main()