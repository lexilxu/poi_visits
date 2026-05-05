# runners/run_naive.py

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split_path",
        type=str,
        default="data/splits/benchmarks/benchmark_split.npz",
        help="Path to benchmark split file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save prediction and true arrays.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="naive",
        help="Name used for saved prediction files.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.split_path)

    x_test = data["x_test"]  # (num_test, seq_in_len, num_nodes)
    y_test = data["y_test"]  # (num_test, seq_out_len, num_nodes)

    if x_test.ndim != 3:
        raise ValueError(f"Expected x_test shape (B, P, N), got {x_test.shape}")

    if y_test.ndim != 3:
        raise ValueError(f"Expected y_test shape (B, H, N), got {y_test.shape}")

    H = y_test.shape[1]

    # Last observed day: (B, 1, N)
    last_value = x_test[:, -1:, :]

    # Repeat the last observed day for all future horizons: (B, H, N)
    pred = np.repeat(last_value, repeats=H, axis=1).astype(np.float32)

    true = y_test.astype(np.float32)

    pred_path = os.path.join(args.output_dir, f"{args.model_name}_pred.npy")
    true_path = os.path.join(args.output_dir, f"{args.model_name}_true.npy")

    np.save(pred_path, pred)
    np.save(true_path, true)

    print("Saved naive benchmark outputs:")
    print("  pred:", pred_path, pred.shape)
    print("  true:", true_path, true.shape)


if __name__ == "__main__":
    main()