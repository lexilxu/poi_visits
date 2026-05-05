# scripts/create_shared_split.py

import os
import json
import argparse
import numpy as np


def make_multistep_samples(visits, seq_in_len, seq_out_len):
    """
    visits: (T, N)

    Returns:
        X: (num_samples, P, N)
        Y: (num_samples, H, N)
        sample_start_times: first forecast time index for each sample
    """
    T, N = visits.shape

    X_list = []
    Y_list = []
    sample_start_times = []

    for t in range(seq_in_len, T - seq_out_len + 1):
        x = visits[t - seq_in_len:t, :]
        y = visits[t:t + seq_out_len, :]

        X_list.append(x)
        Y_list.append(y)
        sample_start_times.append(t)

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    sample_start_times = np.array(sample_start_times, dtype=np.int64)

    return X, Y, sample_start_times


def chronological_split(num_samples, train_ratio, val_ratio):
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)

    train_idx = np.arange(0, num_train)
    val_idx = np.arange(num_train, num_train + num_val)
    test_idx = np.arange(num_train + num_val, num_samples)

    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--visits_path", type=str, default="data/processed/visits.npy")
    parser.add_argument("--output_dir", type=str, default="data/splits")

    parser.add_argument("--seq_in_len", type=int, default=14)
    parser.add_argument("--seq_out_len", type=int, default=7)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    visits = np.load(args.visits_path).astype(np.float32)

    if visits.ndim != 2:
        raise ValueError(f"Expected visits shape (T, N), got {visits.shape}")

    if np.isnan(visits).any():
        raise ValueError("visits.npy contains NaN. Drop or impute missing values first.")

    X, Y, sample_start_times = make_multistep_samples(
        visits,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
    )

    train_idx, val_idx, test_idx = chronological_split(
        num_samples=X.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    np.savez_compressed(
        os.path.join(args.output_dir, "all_samples.npz"),
        X=X,
        Y=Y,
        sample_start_times=sample_start_times,
    )

    np.savez_compressed(
        os.path.join(args.output_dir, "split_indices.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    split_info = {
        "seq_in_len": int(args.seq_in_len),
        "seq_out_len": int(args.seq_out_len),
        "num_days": int(visits.shape[0]),
        "num_nodes": int(visits.shape[1]),
        "num_samples": int(X.shape[0]),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "train_idx_range": [int(train_idx[0]), int(train_idx[-1])],
        "val_idx_range": [int(val_idx[0]), int(val_idx[-1])],
        "test_idx_range": [int(test_idx[0]), int(test_idx[-1])],
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
    }

    with open(os.path.join(args.output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print("Saved shared samples and split indices.")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("train:", len(train_idx))
    print("val:", len(val_idx))
    print("test:", len(test_idx))


if __name__ == "__main__":
    main()