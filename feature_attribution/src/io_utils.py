import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_output_dirs(cfg):
    for key in ["raw_attribution_dir", "summary_dir", "plot_dir", "report_dir"]:
        ensure_dir(cfg["outputs"][key])


def load_benchmark_split(split_path):
    data = np.load(split_path)

    required = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]
    for key in required:
        if key not in data.files:
            raise ValueError(f"Missing key {key} in {split_path}")

    return {
        "x_train": data["x_train"].astype(np.float32),
        "y_train": data["y_train"].astype(np.float32),
        "x_val": data["x_val"].astype(np.float32),
        "y_val": data["y_val"].astype(np.float32),
        "x_test": data["x_test"].astype(np.float32),
        "y_test": data["y_test"].astype(np.float32),
    }


def load_poi_meta(poi_meta_path, location_table_path=None):
    meta = pd.read_csv(poi_meta_path)

    if location_table_path is not None and os.path.exists(location_table_path):
        loc = pd.read_csv(location_table_path)

        if "placekey" in meta.columns and "placekey" in loc.columns:
            loc = loc.drop_duplicates(subset=["placekey"])
            meta["placekey"] = meta["placekey"].astype(str)
            loc["placekey"] = loc["placekey"].astype(str)

            extra_cols = [
                c for c in loc.columns
                if c not in meta.columns or c == "placekey"
            ]

            meta = meta.merge(
                loc[extra_cols],
                on="placekey",
                how="left",
            )

    return meta


def make_test_loader(x_test, y_test, batch_size, max_batches=None):
    x = torch.tensor(x_test, dtype=torch.float32)
    y = torch.tensor(y_test, dtype=torch.float32)

    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    if max_batches is None:
        return loader

    batches = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(batch)

    return batches


def get_device(device_str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def compute_baseline(x, kind="mean"):
    """
    x: torch tensor with shape (B, P, N)
    """
    if kind == "zero":
        return torch.zeros_like(x)

    if kind == "mean":
        mean_value = x.mean(dim=0, keepdim=True)
        return mean_value.repeat(x.shape[0], 1, 1)

    if kind == "sample_mean":
        mean_value = x.mean(dim=1, keepdim=True)
        return mean_value.repeat(1, x.shape[1], 1)

    raise ValueError(f"Unknown baseline kind: {kind}")


def save_numpy(path, arr):
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def save_dataframe(path, df):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def add_project_root_to_path(root_dir="."):
    root_dir = os.path.abspath(root_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)