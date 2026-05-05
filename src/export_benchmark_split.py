# scripts/export_benchmark_split.py

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_path", type=str, default="data/splits/all_samples.npz")
    parser.add_argument("--indices_path", type=str, default="data/splits/split_indices.npz")
    parser.add_argument("--output_path", type=str, default="data/splits/benchmarks/benchmark_split.npz")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    samples = np.load(args.samples_path)
    indices = np.load(args.indices_path)

    X = samples["X"]
    Y = samples["Y"]

    train_idx = indices["train_idx"]
    val_idx = indices["val_idx"]
    test_idx = indices["test_idx"]

    np.savez_compressed(
        args.output_path,
        x_train=X[train_idx],
        y_train=Y[train_idx],
        x_val=X[val_idx],
        y_val=Y[val_idx],
        x_test=X[test_idx],
        y_test=Y[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    print("Saved benchmark split:", args.output_path)
    print("x_train:", X[train_idx].shape)
    print("y_train:", Y[train_idx].shape)
    print("x_val:  ", X[val_idx].shape)
    print("y_val:  ", Y[val_idx].shape)
    print("x_test: ", X[test_idx].shape)
    print("y_test: ", Y[test_idx].shape)


if __name__ == "__main__":
    main()