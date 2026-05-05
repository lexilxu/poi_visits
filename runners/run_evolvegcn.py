# runners/run_evolvegcn.py

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class POIGraphDataset(Dataset):
    def __init__(self, x, y):
        """
        x: (B, P, N)
        y: (B, H, N)
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], torch.tensor(idx, dtype=torch.long)


def normalize_adj_numpy(adj):
    """
    Symmetric adjacency normalization:
        D^{-1/2} (A + I) D^{-1/2}
    """
    adj = adj.astype(np.float32)
    n = adj.shape[0]

    adj = adj + np.eye(n, dtype=np.float32)

    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0

    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def normalize_adj_torch(adj):
    """
    adj: torch tensor with shape (N, N)
    """
    n = adj.shape[0]
    adj = adj + torch.eye(n, dtype=adj.dtype, device=adj.device)

    deg = adj.sum(dim=1)
    deg = torch.where(deg == 0, torch.ones_like(deg), deg)

    deg_inv_sqrt = torch.pow(deg, -0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def edge_graph_to_dense(edge_src, edge_dst, edge_weight, edge_ptr, graph_id, num_nodes, device):
    """
    Converts one sparse dynamic graph into a dense adjacency matrix.

    edge_src, edge_dst, edge_weight, edge_ptr are numpy arrays.
    graph_id identifies one graph in the dynamic graph sequence.
    """
    start = int(edge_ptr[graph_id])
    end = int(edge_ptr[graph_id + 1])

    src = edge_src[start:end]
    dst = edge_dst[start:end]
    weight = edge_weight[start:end]

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    w_t = torch.tensor(weight, dtype=torch.float32, device=device)

    A[src_t, dst_t] = w_t

    return A


class GraphGRUForecast(nn.Module):
    """
    EvolveGCN-style graph recurrent forecaster.

    At each historical time step:
        1. graph propagation is applied to node features,
        2. node-level hidden states are updated by a GRUCell,
        3. final node representations are decoded into H-step forecasts.

    Input:
        x: (B, P, N)

    Output:
        pred: (B, H, N)
    """

    def __init__(self, num_nodes, horizon, hidden_dim=32):
        super().__init__()

        self.num_nodes = num_nodes
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(1, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, horizon)

    def forward_static(self, x, static_adj):
        """
        x: (B, P, N)
        static_adj: (N, N)
        """
        B, P, N = x.shape

        h = torch.zeros(B * N, self.hidden_dim, device=x.device)

        for t in range(P):
            xt = x[:, t, :]  # (B, N)

            # Graph propagation.
            xt_g = torch.matmul(xt, static_adj.T)  # (B, N)

            feat = xt_g.reshape(B * N, 1)
            feat = self.input_proj(feat)

            h = self.gru_cell(feat, h)

        h_nodes = h.reshape(B, N, self.hidden_dim)
        pred_nh = self.decoder(h_nodes)      # (B, N, H)
        pred = pred_nh.permute(0, 2, 1)      # (B, H, N)

        return pred

    def forward_dynamic(self, x, sample_positions, edge_data):
        """
        x: (B, P, N)

        Dynamic graph convention:
            graph_id = sample_position * P + t
        """
        B, P, N = x.shape

        h = torch.zeros(B * N, self.hidden_dim, device=x.device)

        for t in range(P):
            xt_all = []

            for b in range(B):
                sample_pos = int(sample_positions[b].item())
                graph_id = sample_pos * P + t

                A = edge_graph_to_dense(
                    edge_src=edge_data["edge_src"],
                    edge_dst=edge_data["edge_dst"],
                    edge_weight=edge_data["edge_weight"],
                    edge_ptr=edge_data["edge_ptr"],
                    graph_id=graph_id,
                    num_nodes=N,
                    device=x.device,
                )

                A = normalize_adj_torch(A)

                xt_b = x[b, t, :]            # (N,)
                xt_g_b = torch.matmul(A, xt_b)  # (N,)
                xt_all.append(xt_g_b)

            xt_g = torch.stack(xt_all, dim=0)  # (B, N)

            feat = xt_g.reshape(B * N, 1)
            feat = self.input_proj(feat)

            h = self.gru_cell(feat, h)

        h_nodes = h.reshape(B, N, self.hidden_dim)
        pred_nh = self.decoder(h_nodes)      # (B, N, H)
        pred = pred_nh.permute(0, 2, 1)      # (B, H, N)

        return pred

    def forward(self, x, adjacency_mode, static_adj=None, sample_positions=None, edge_data=None):
        if adjacency_mode == "static":
            return self.forward_static(x, static_adj)

        if adjacency_mode == "dynamic":
            return self.forward_dynamic(x, sample_positions, edge_data)

        raise ValueError(f"Unknown adjacency_mode: {adjacency_mode}")


def load_edge_data(data, split_name):
    return {
        "edge_src": data[f"{split_name}_edge_src"],
        "edge_dst": data[f"{split_name}_edge_dst"],
        "edge_weight": data[f"{split_name}_edge_weight"],
        "edge_ptr": data[f"{split_name}_edge_ptr"],
    }


def make_loss(loss_name):
    if loss_name == "mse":
        return nn.MSELoss()

    if loss_name == "mae":
        return nn.L1Loss()

    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=1.0)

    raise ValueError(f"Unknown loss: {loss_name}")


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rse(y_true, y_pred):
    numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def rae(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def corr(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metrics_by_horizon(y_true, y_pred, model_name, split_name):
    """
    y_true, y_pred: (B, H, N)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    B, H, N = y_true.shape

    rows = []

    for h in range(H):
        yt = y_true[:, h, :]
        yp = y_pred[:, h, :]

        rows.append({
            "model": model_name,
            "split": split_name,
            "horizon": h + 1,
            "num_samples": B,
            "num_nodes": N,
            "MAE": mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "RSE": rse(yt, yp),
            "RAE": rae(yt, yp),
            "CORR": corr(yt, yp),
        })

    rows.append({
        "model": model_name,
        "split": split_name,
        "horizon": "avg",
        "num_samples": B,
        "num_nodes": N,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "RSE": rse(y_true, y_pred),
        "RAE": rae(y_true, y_pred),
        "CORR": corr(y_true, y_pred),
    })

    return pd.DataFrame(rows)


def save_metrics_csv(y_true, y_pred, model_name, output_path, split_name):
    df = metrics_by_horizon(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
        split_name=split_name,
    )
    df.to_csv(output_path, index=False)
    print(f"Saved {split_name} metrics: {output_path}")
    return df


def train_one_epoch(model, loader, adjacency_mode, static_adj, edge_data, optimizer, loss_fn, device):
    model.train()
    losses = []

    for x, y, sample_pos in loader:
        x = x.to(device)
        y = y.to(device)
        sample_pos = sample_pos.to(device)

        optimizer.zero_grad()

        pred = model(
            x=x,
            adjacency_mode=adjacency_mode,
            static_adj=static_adj,
            sample_positions=sample_pos,
            edge_data=edge_data,
        )

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(model, loader, adjacency_mode, static_adj, edge_data, loss_fn, device):
    model.eval()
    losses = []

    for x, y, sample_pos in loader:
        x = x.to(device)
        y = y.to(device)
        sample_pos = sample_pos.to(device)

        pred = model(
            x=x,
            adjacency_mode=adjacency_mode,
            static_adj=static_adj,
            sample_positions=sample_pos,
            edge_data=edge_data,
        )

        loss = loss_fn(pred, y)
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, adjacency_mode, static_adj, edge_data, device):
    model.eval()

    preds = []
    trues = []

    for x, y, sample_pos in loader:
        x = x.to(device)
        sample_pos = sample_pos.to(device)

        pred = model(
            x=x,
            adjacency_mode=adjacency_mode,
            static_adj=static_adj,
            sample_positions=sample_pos,
            edge_data=edge_data,
        )

        preds.append(pred.cpu())
        trues.append(y)

    pred = torch.cat(preds, dim=0).numpy()
    true = torch.cat(trues, dim=0).numpy()

    return pred, true


def apply_normalization(x_train, y_train, x_val, y_val, x_test, y_test, normalize):
    """
    Normalizes x and y using train-set input statistics.

    This matches the spirit of MTGNN scaling while keeping final saved predictions
    on the original visit-count scale.
    """
    if not normalize:
        stats = {
            "mean": 0.0,
            "std": 1.0,
            "normalized": False,
        }
        return x_train, y_train, x_val, y_val, x_test, y_test, stats

    mean = float(x_train.mean())
    std = float(x_train.std())

    if std < 1e-6:
        std = 1.0

    x_train_n = (x_train - mean) / std
    y_train_n = (y_train - mean) / std
    x_val_n = (x_val - mean) / std
    y_val_n = (y_val - mean) / std
    x_test_n = (x_test - mean) / std
    y_test_n = (y_test - mean) / std

    stats = {
        "mean": mean,
        "std": std,
        "normalized": True,
    }

    return x_train_n, y_train_n, x_val_n, y_val_n, x_test_n, y_test_n, stats


def inverse_transform(pred, stats):
    return pred * stats["std"] + stats["mean"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--evolvegcn_dir", type=str, default="external/EvolveGCN")
    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"])

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize x and y using train-set input mean and std. Predictions are saved back on raw scale.",
    )

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--validation_dir", type=str, default="outputs/validation")
    parser.add_argument("--training_log_dir", type=str, default="outputs/training_logs")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.validation_dir, exist_ok=True)
    os.makedirs(args.training_log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = np.load(args.data_path, allow_pickle=True)

    adjacency_mode = str(data["adjacency_mode"])

    x_train_raw = data["x_train"].astype(np.float32)
    y_train_raw = data["y_train"].astype(np.float32)

    x_val_raw = data["x_val"].astype(np.float32)
    y_val_raw = data["y_val"].astype(np.float32)

    x_test_raw = data["x_test"].astype(np.float32)
    y_test_raw = data["y_test"].astype(np.float32)

    x_train, y_train, x_val, y_val, x_test, y_test, norm_stats = apply_normalization(
        x_train=x_train_raw,
        y_train=y_train_raw,
        x_val=x_val_raw,
        y_val=y_val_raw,
        x_test=x_test_raw,
        y_test=y_test_raw,
        normalize=args.normalize,
    )

    _, P, N = x_train.shape
    H = y_train.shape[1]

    print("Loaded EvolveGCN-style data:")
    print("adjacency_mode:", adjacency_mode)
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_val:  ", x_val.shape)
    print("y_val:  ", y_val.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    print("P:", P)
    print("H:", H)
    print("N:", N)
    print("device:", device)
    print("hidden_dim:", args.hidden_dim)
    print("lr:", args.lr)
    print("loss:", args.loss)
    print("normalize:", args.normalize)
    print("normalization mean:", norm_stats["mean"])
    print("normalization std:", norm_stats["std"])

    static_adj = None
    train_edge_data = None
    val_edge_data = None
    test_edge_data = None

    if adjacency_mode == "static":
        adj = data["adj"].astype(np.float32)

        if adj.shape != (N, N):
            raise ValueError(f"adj shape {adj.shape} does not match N={N}")

        adj_norm = normalize_adj_numpy(adj)
        static_adj = torch.tensor(adj_norm, dtype=torch.float32, device=device)

    elif adjacency_mode == "dynamic":
        train_edge_data = load_edge_data(data, "train")
        val_edge_data = load_edge_data(data, "val")
        test_edge_data = load_edge_data(data, "test")

        print("Dynamic edges:")
        print("train:", train_edge_data["edge_src"].shape)
        print("val:  ", val_edge_data["edge_src"].shape)
        print("test: ", test_edge_data["edge_src"].shape)

    else:
        raise ValueError(f"Unknown adjacency_mode: {adjacency_mode}")

    train_ds = POIGraphDataset(x_train, y_train)
    val_ds = POIGraphDataset(x_val, y_val)
    test_ds = POIGraphDataset(x_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = GraphGRUForecast(
        num_nodes=N,
        horizon=H,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_loss(args.loss)

    best_val = float("inf")
    best_state = None
    best_epoch = None
    bad_epochs = 0

    epoch_rows = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            adjacency_mode=adjacency_mode,
            static_adj=static_adj,
            edge_data=train_edge_data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            adjacency_mode=adjacency_mode,
            static_adj=static_adj,
            edge_data=val_edge_data,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_rows.append({
            "model": args.model_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_name": args.loss,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "normalize": args.normalize,
        })

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train {args.loss}: {train_loss:.6f} | "
            f"val {args.loss}: {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    training_log_path = os.path.join(
        args.training_log_dir,
        f"{args.model_name}_training_log.csv",
    )
    pd.DataFrame(epoch_rows).to_csv(training_log_path, index=False)
    print("Saved training log:", training_log_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{args.model_name}.pt",
    )
    torch.save(model.state_dict(), checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)
    print("Best validation loss:", best_val)
    print("Best epoch:", best_epoch)

    # Validation predictions and validation metrics.
    val_pred_norm, val_true_norm = predict(
        model=model,
        loader=val_loader,
        adjacency_mode=adjacency_mode,
        static_adj=static_adj,
        edge_data=val_edge_data,
        device=device,
    )

    if args.normalize:
        val_pred_raw = inverse_transform(val_pred_norm, norm_stats)
        val_true_raw = y_val_raw.astype(np.float32)
    else:
        val_pred_raw = val_pred_norm.astype(np.float32)
        val_true_raw = val_true_norm.astype(np.float32)

    val_pred_raw = np.clip(val_pred_raw, a_min=0.0, a_max=None)

    val_metrics_path = os.path.join(
        args.validation_dir,
        f"{args.model_name}_val_metrics.csv",
    )

    save_metrics_csv(
        y_true=val_true_raw.astype(np.float32),
        y_pred=val_pred_raw.astype(np.float32),
        model_name=args.model_name,
        output_path=val_metrics_path,
        split_name="val",
    )

    # Test predictions.
    test_pred_norm, test_true_norm = predict(
        model=model,
        loader=test_loader,
        adjacency_mode=adjacency_mode,
        static_adj=static_adj,
        edge_data=test_edge_data,
        device=device,
    )

    if args.normalize:
        test_pred_raw = inverse_transform(test_pred_norm, norm_stats)
        test_true_raw = y_test_raw.astype(np.float32)
    else:
        test_pred_raw = test_pred_norm.astype(np.float32)
        test_true_raw = test_true_norm.astype(np.float32)

    test_pred_raw = np.clip(test_pred_raw, a_min=0.0, a_max=None)
    
    pred_path = os.path.join(args.output_dir, f"{args.model_name}_pred.npy")
    true_path = os.path.join(args.output_dir, f"{args.model_name}_true.npy")

    np.save(pred_path, test_pred_raw.astype(np.float32))
    np.save(true_path, test_true_raw.astype(np.float32))

    print("Saved test outputs:")
    print("pred:", pred_path, test_pred_raw.shape)
    print("true:", true_path, test_true_raw.shape)

    # Save normalization metadata for reproducibility.
    norm_path = os.path.join(
        args.checkpoint_dir,
        f"{args.model_name}_normalization.json",
    )

    try:
        import json
        with open(norm_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print("Saved normalization info:", norm_path)
    except Exception as e:
        print("Warning: failed to save normalization info:", e)


if __name__ == "__main__":
    main()