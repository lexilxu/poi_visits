# scripts/export_mtgnn_split.py

import os
import argparse
import numpy as np


def save_mtgnn_file(path, X, Y, seq_in_len, seq_out_len):
    x = X[..., None].astype(np.float32)
    y = Y[..., None].astype(np.float32)

    x_offsets = np.arange(-seq_in_len + 1, 1).reshape(-1, 1)
    y_offsets = np.arange(1, seq_out_len + 1).reshape(-1, 1)

    np.savez_compressed(
        path,
        x=x,
        y=y,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_path", type=str, default="data/splits/all_samples.npz")
    parser.add_argument("--indices_path", type=str, default="data/splits/split_indices.npz")
    parser.add_argument("--output_dir", type=str, default="data/splits/mtgnn")

    parser.add_argument("--seq_in_len", type=int, default=14)
    parser.add_argument("--seq_out_len", type=int, default=7)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    samples = np.load(args.samples_path)
    indices = np.load(args.indices_path)

    X = samples["X"]  # (num_samples, P, N)
    Y = samples["Y"]  # (num_samples, H, N)

    train_idx = indices["train_idx"]
    val_idx = indices["val_idx"]
    test_idx = indices["test_idx"]

    save_mtgnn_file(
        os.path.join(args.output_dir, "train.npz"),
        X[train_idx],
        Y[train_idx],
        args.seq_in_len,
        args.seq_out_len,
    )

    save_mtgnn_file(
        os.path.join(args.output_dir, "val.npz"),
        X[val_idx],
        Y[val_idx],
        args.seq_in_len,
        args.seq_out_len,
    )

    save_mtgnn_file(
        os.path.join(args.output_dir, "test.npz"),
        X[test_idx],
        Y[test_idx],
        args.seq_in_len,
        args.seq_out_len,
    )

    print("Saved MTGNN split files to:", args.output_dir)
    print("train:", X[train_idx].shape, Y[train_idx].shape)
    print("val:  ", X[val_idx].shape, Y[val_idx].shape)
    print("test: ", X[test_idx].shape, Y[test_idx].shape)


if __name__ == "__main__":
    main()