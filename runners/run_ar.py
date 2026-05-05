# runners/run_ar.py

import os
import argparse
import numpy as np
from sklearn.linear_model import Ridge


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
        default="ar_ridge",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=100,
        help="Ridge regularization strength.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.split_path)

    x_train = data["x_train"]  # (B_train, P, N)
    y_train = data["y_train"]  # (B_train, H, N)

    x_test = data["x_test"]    # (B_test, P, N)
    y_test = data["y_test"]    # (B_test, H, N)

    if x_train.ndim != 3:
        raise ValueError(f"Expected x_train shape (B, P, N), got {x_train.shape}")

    if y_train.ndim != 3:
        raise ValueError(f"Expected y_train shape (B, H, N), got {y_train.shape}")

    B_test, P, N = x_test.shape
    H = y_test.shape[1]

    print("Loaded benchmark split:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    print("P:", P)
    print("H:", H)
    print("N:", N)

    pred = np.zeros((B_test, H, N), dtype=np.float32)

    for j in range(N):
        # Past trajectory for POI j
        X_j = x_train[:, :, j]  # (B_train, P)

        # Future trajectory for POI j
        Y_j = y_train[:, :, j]  # (B_train, H)

        model = Ridge(alpha=args.alpha)
        model.fit(X_j, Y_j)

        pred[:, :, j] = model.predict(x_test[:, :, j]).astype(np.float32)

        if (j + 1) % 500 == 0:
            print(f"Finished {j + 1}/{N} POIs")

    true = y_test.astype(np.float32)

    pred_path = os.path.join(args.output_dir, f"{args.model_name}_pred.npy")
    true_path = os.path.join(args.output_dir, f"{args.model_name}_true.npy")

    np.save(pred_path, pred)
    np.save(true_path, true)

    print("Saved AR-Ridge outputs:")
    print("  pred:", pred_path, pred.shape)
    print("  true:", true_path, true.shape)


if __name__ == "__main__":
    main()