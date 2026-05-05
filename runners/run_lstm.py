# runners/run_lstm.py

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ForecastDataset(Dataset):
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
        return self.x[idx], self.y[idx]


class LSTMForecast(nn.Module):
    def __init__(self, num_nodes, hidden_dim, horizon, num_layers=1, dropout=0.0):
        super().__init__()

        self.num_nodes = num_nodes
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=num_nodes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, horizon * num_nodes)

    def forward(self, x):
        """
        x: (B, P, N)
        output: (B, H, N)
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        pred = pred.view(x.shape[0], self.horizon, self.num_nodes)
        return pred


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    preds = []
    trues = []

    for x, y in loader:
        x = x.to(device)
        pred = model(x)

        preds.append(pred.cpu())
        trues.append(y)

    pred = torch.cat(preds, dim=0).numpy()
    true = torch.cat(trues, dim=0).numpy()

    return pred, true


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split_path",
        type=str,
        default="data/splits/benchmarks/benchmark_split.npz",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/predictions",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="lstm",
    )

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = np.load(args.split_path)

    x_train = data["x_train"]  # (B_train, P, N)
    y_train = data["y_train"]  # (B_train, H, N)

    x_val = data["x_val"]
    y_val = data["y_val"]

    x_test = data["x_test"]
    y_test = data["y_test"]

    if x_train.ndim != 3:
        raise ValueError(f"Expected x_train shape (B, P, N), got {x_train.shape}")

    if y_train.ndim != 3:
        raise ValueError(f"Expected y_train shape (B, H, N), got {y_train.shape}")

    _, P, N = x_train.shape
    H = y_train.shape[1]

    print("Loaded benchmark split:")
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

    train_ds = ForecastDataset(x_train, y_train)
    val_ds = ForecastDataset(x_val, y_val)
    test_ds = ForecastDataset(x_test, y_test)

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

    model = LSTMForecast(
        num_nodes=N,
        hidden_dim=args.hidden_dim,
        horizon=H,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train MSE {train_loss:.6f} | "
            f"val MSE {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
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

    if best_state is not None:
        model.load_state_dict(best_state)

    pred, true = predict(model, test_loader, device)

    pred_path = os.path.join(args.output_dir, f"{args.model_name}_pred.npy")
    true_path = os.path.join(args.output_dir, f"{args.model_name}_true.npy")

    np.save(pred_path, pred.astype(np.float32))
    np.save(true_path, true.astype(np.float32))

    print("Saved LSTM outputs:")
    print("  pred:", pred_path, pred.shape)
    print("  true:", true_path, true.shape)


if __name__ == "__main__":
    main()