import os
import sys
import argparse
import numpy as np
import torch

from feature_attribution.src.io_utils import (
    load_yaml,
    ensure_output_dirs,
    load_benchmark_split,
    compute_baseline,
    get_device,
    add_project_root_to_path,
    save_numpy,
)
from feature_attribution.src.targets import build_target_fn
from feature_attribution.src.attribution_methods import (
    saliency,
    input_x_gradient,
    integrated_gradients,
)


def load_lstm_model(model_cfg, x_train_shape, y_train_shape, device):
    from runners.run_lstm import LSTMForecast

    _, P, N = x_train_shape
    H = y_train_shape[1]

    model = LSTMForecast(
        num_nodes=N,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        horizon=H,
        num_layers=model_cfg.get("num_layers", 1),
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)

    ckpt = model_cfg["checkpoint_path"]
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


def load_evolvegcn_style_model(model_cfg, device):
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

    ckpt = model_cfg["checkpoint_path"]
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    adjacency_mode = str(data["adjacency_mode"])

    if adjacency_mode != "static":
        raise NotImplementedError(
            "Gradient attribution currently supports static EvolveGCN-style models only."
        )

    adj = data["adj"]
    adj_norm = normalize_adj_numpy(adj)
    static_adj = torch.tensor(adj_norm, dtype=torch.float32, device=device)

    def model_forward(x):
        return model(
            x=x,
            adjacency_mode="static",
            static_adj=static_adj,
            sample_positions=None,
            edge_data=None,
        )

    return model, model_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="feature_attribution/configs/attribution_config.yaml")
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--target_type", type=str, default="global", choices=["global", "node"])
    parser.add_argument("--node_idx", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None, help="One-based horizon. Use 1, 3, 7, etc.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ensure_output_dirs(cfg)
    add_project_root_to_path(cfg["project"].get("root_dir", "."))

    model_cfg = cfg["models"][args.model_key]
    model_type = model_cfg["model_type"]
    model_name = model_cfg["model_name"]

    device = get_device(cfg["attribution"].get("device", "cuda:0"))

    split = load_benchmark_split(cfg["data"]["benchmark_split_path"])

    x_test = split["x_test"]
    y_test = split["y_test"]
    x_train = split["x_train"]
    y_train = split["y_train"]

    max_batches = cfg["attribution"].get("max_batches", 5)
    batch_size = cfg["attribution"].get("batch_size", 8)

    x_batch = torch.tensor(x_test[: max_batches * batch_size], dtype=torch.float32, device=device)

    if model_type == "lstm":
        model = load_lstm_model(
            model_cfg=model_cfg,
            x_train_shape=x_train.shape,
            y_train_shape=y_train.shape,
            device=device,
        )

        def model_forward(x):
            return model(x)

    elif model_type == "evolvegcn_style_static":
        model, model_forward = load_evolvegcn_style_model(model_cfg, device=device)

    elif model_type == "mtgnn":
        raise NotImplementedError(
            "MTGNN gradient attribution needs a dedicated MTGNN forward wrapper. "
            "Use perturbation attribution first."
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if args.target_type == "global":
        target_name = "global"
        target_fn = build_target_fn("global", horizons="all")

    else:
        if args.node_idx is None:
            raise ValueError("--node_idx is required for node target")

        horizon_idx = None
        if args.horizon is not None:
            horizon_idx = args.horizon - 1

        target_name = f"node{args.node_idx}_h{args.horizon if args.horizon else 'all'}"

        target_fn = build_target_fn(
            "node",
            node_idx=args.node_idx,
            horizon_idx=horizon_idx,
        )

    raw_dir = cfg["outputs"]["raw_attribution_dir"]

    methods = cfg["attribution"]["methods"]

    if "saliency" in methods:
        attr = saliency(model_forward, x_batch, target_fn)
        save_numpy(
            os.path.join(raw_dir, f"{model_name}_{target_name}_saliency.npy"),
            attr.detach().cpu().numpy(),
        )

    if "input_x_gradient" in methods:
        attr = input_x_gradient(model_forward, x_batch, target_fn)
        save_numpy(
            os.path.join(raw_dir, f"{model_name}_{target_name}_input_x_gradient.npy"),
            attr.detach().cpu().numpy(),
        )

    if "integrated_gradients" in methods:
        ig_cfg = cfg["attribution"]["integrated_gradients"]
        baseline = compute_baseline(x_batch, kind=ig_cfg.get("baseline", "mean"))
        attr = integrated_gradients(
            model_forward=model_forward,
            x=x_batch,
            target_fn=target_fn,
            baseline=baseline,
            n_steps=ig_cfg.get("n_steps", 20),
        )
        save_numpy(
            os.path.join(raw_dir, f"{model_name}_{target_name}_integrated_gradients.npy"),
            attr.detach().cpu().numpy(),
        )

    print("Finished input attribution for", model_name, target_name)


if __name__ == "__main__":
    main()