# src/evaluate_all.py

import os
import glob
import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FAR (False Activation Rate) — I_0 cache (computed once per process)
# ---------------------------------------------------------------------------
_I0_INDICES = None  # POI indices with zero visits across ALL data (train+val+test)


def get_zero_history_poi_indices(data_dir="data/splits/mtgnn"):
    """
    Identify I_0: POI indices whose total visit count is zero across
    ALL historical data (train + val + test splits).

    Returns a numpy array of POI indices (N_i0,).
    """
    global _I0_INDICES
    if _I0_INDICES is not None:
        return _I0_INDICES

    all_zeros = np.zeros(0, dtype=bool)
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.npz")
        d = np.load(path)
        # Channel 0 is the visit data
        ch0 = d["x"][:, 0, :, :]  # (B, N, T_in)
        poi_sum = ch0.sum(axis=(0, 2))  # (N,)
        if all_zeros.size == 0:
            all_zeros = np.zeros_like(poi_sum, dtype=bool)
        all_zeros |= (poi_sum == 0)

    _I0_INDICES = np.where(all_zeros)[0]
    print(f"[FAR] I_0 (zero-history POIs): {len(_I0_INDICES)} / {len(all_zeros)}")
    return _I0_INDICES


def far(y_true, y_pred, i0_indices, delta=0.5):
    if i0_indices is None or len(i0_indices) == 0:
        return np.nan, 0

    # Extract only I_0 POI events
    omega_true = y_true[:, :, i0_indices]   # (B, H, N_i0)
    omega_pred = y_pred[:, :, i0_indices]   # (B, H, N_i0)

    # Omega_0^0: j in I_0 AND Y_{b,h,j} = 0
    omega_mask = (omega_true == 0)           # (B, H, N_i0)

    omega_size = int(omega_mask.sum())
    if omega_size == 0:
        return np.nan, 0

    # Count how many of those zero-truth events have pred >= delta
    false_activations = ((omega_pred >= delta) & omega_mask).sum()
    return float(false_activations / omega_size), omega_size


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def normalize_to_bhn(y, name, file_path):
    orig_shape = y.shape
    y = np.asarray(y)

    # 4D -> 3D (e.g. (B,H,N,1) -> (B,H,N))
    if y.ndim == 4:
        y = y.squeeze(-1)

    # 3D: detect (B, H, N) vs (B, N, H)
    if y.ndim == 3:
        d1, d2 = y.shape[1], y.shape[2]
        # Small dim-1 (2-14) = horizon, large dim-2 (>100) = num_nodes -> correct
        if 2 <= d1 <= 14 and d2 > 100:
            pass  # already (B, H, N)
        else:
            y = y.transpose(0, 2, 1)

    if y.ndim != 3:
        raise ValueError(
            f"Cannot normalize {name} from {file_path} with shape {orig_shape} to (B,H,N): got {y.shape}"
        )

    return y


def check_prediction_shapes(model_name, y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch for {model_name}: "
            f"true shape {y_true.shape}, pred shape {y_pred.shape}"
        )

    if y_true.ndim != 3:
        raise ValueError(
            f"Expected prediction and true arrays for {model_name} to have shape "
            f"(num_samples, horizon, num_nodes), but got {y_true.shape}"
        )


def evaluate_by_horizon(model_name, y_true, y_pred, i0_indices=None):
    """
    y_true, y_pred: arrays with shape (B, H, N)

    B = number of test samples
    H = number of forecast horizons
    N = number of POIs
    """
    check_prediction_shapes(model_name, y_true, y_pred)

    B, H, N = y_true.shape
    rows = []

    for h in range(H):
        yt = y_true[:, h, :]
        yp = y_pred[:, h, :]

        row = {
            "model": model_name,
            "horizon": h + 1,
            "num_samples": B,
            "num_nodes": N,
            "MAE": mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "RSE": rse(yt, yp),
            "RAE": rae(yt, yp),
            "CORR": corr(yt, yp),
        }
        if i0_indices is not None and len(i0_indices) > 0:
            far_h, omega_h = far(yt.reshape(B, 1, N), yp.reshape(B, 1, N), i0_indices)
            row["FAR"] = far_h
        rows.append(row)

    return rows


def evaluate_average(model_name, y_true, y_pred, i0_indices=None):
    """
    Average metrics over all samples, horizons, and POIs.
    Also computes FAR over the full Omega_0^0 set.
    """
    check_prediction_shapes(model_name, y_true, y_pred)

    B, H, N = y_true.shape

    row = {
        "model": model_name,
        "num_samples": B,
        "num_horizons": H,
        "num_nodes": N,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "RSE": rse(y_true, y_pred),
        "RAE": rae(y_true, y_pred),
        "CORR": corr(y_true, y_pred),
    }
    if i0_indices is not None and len(i0_indices) > 0:
        far_val, omega_size = far(y_true, y_pred, i0_indices)
        row["FAR"] = far_val
        row["FAR_omega_size"] = omega_size
    return row


def make_latex_table(avg_df, output_path, caption, label, decimals=4):
    """
    Save a simple LaTeX table of average metrics.
    """
    table_df = avg_df.copy()

    metric_cols = ["MAE", "RMSE", "RSE", "RAE", "CORR", "FAR"]
    for col in metric_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"${x:.{decimals}f}$" if pd.notna(x) else "")

    cols = ["model"] + [c for c in metric_cols if c in table_df.columns]
    table_df = table_df[cols]
    table_df = table_df.set_index("model")

    latex = table_df.to_latex(
        escape=False,
        column_format="l" + "r" * (len(cols) - 1),
        caption=caption,
        label=label,
    )

    with open(output_path, "w") as f:
        f.write(latex)

    print("Saved LaTeX table:", output_path)


def make_horizon_latex_table(horizon_df, output_path, caption, label, decimals=4):
    """
    Save a long-form LaTeX table with one row per model-horizon pair.
    """
    table_df = horizon_df.copy()

    metric_cols = ["MAE", "RMSE", "RSE", "RAE", "CORR", "FAR"]
    for col in metric_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"${x:.{decimals}f}$" if pd.notna(x) else "")

    cols = ["model", "horizon"] + [c for c in metric_cols if c in table_df.columns]
    table_df = table_df[cols]

    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (len(cols) - 1),
        caption=caption,
        label=label,
    )

    with open(output_path, "w") as f:
        f.write(latex)

    print("Saved horizon LaTeX table:", output_path)


def get_model_name_from_pred_path(pred_path):
    return os.path.basename(pred_path).replace("_pred.npy", "")


def load_prediction_pair(prediction_dir, model_name):
    pred_path = os.path.join(prediction_dir, f"{model_name}_pred.npy")
    true_path = os.path.join(prediction_dir, f"{model_name}_true.npy")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Missing true file: {true_path}")

    y_pred = np.load(pred_path)
    y_true = np.load(true_path)

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction_dir",
        type=str,
        default="outputs/predictions",
        help="Directory containing *_pred.npy and *_true.npy files.",
    )

    parser.add_argument(
        "--metric_dir",
        type=str,
        default="outputs/metrics",
        help="Directory to save CSV metric outputs.",
    )

    parser.add_argument(
        "--table_dir",
        type=str,
        default="outputs/tables",
        help="Directory to save LaTeX tables.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional list of exact model names to evaluate. "
            "Each name should match the filename before _pred.npy. "
            "For example, if the file is mtgnn_identity_pred.npy, "
            "the model name is mtgnn_identity."
        ),
    )

    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Number of decimals for LaTeX table output.",
    )

    parser.add_argument(
        "--clip_negative",
        action="store_true",
        help="Clip negative predictions to zero before evaluation.",
    )

    parser.add_argument(
        "--far_delta",
        type=float,
        default=0.5,
        help="Threshold delta for False Activation Rate (FAR). Default: 0.5",
    )

    args = parser.parse_args()

    ensure_dir(args.metric_dir)
    ensure_dir(args.table_dir)

    pred_files = sorted(glob.glob(os.path.join(args.prediction_dir, "*_pred.npy")))

    if len(pred_files) == 0:
        raise RuntimeError(f"No *_pred.npy files found in {args.prediction_dir}")

    selected_models = set(args.models) if args.models is not None else None

    # Load I_0 (zero-history POI indices) once
    data_dir = "data/splits/mtgnn"
    i0_indices = get_zero_history_poi_indices(data_dir)

    horizon_rows = []
    average_rows = []
    evaluated_models = []

    for pred_path in pred_files:
        model_name = get_model_name_from_pred_path(pred_path)

        if selected_models is not None and model_name not in selected_models:
            continue

        true_path = os.path.join(args.prediction_dir, f"{model_name}_true.npy")

        if not os.path.exists(true_path):
            print(f"Warning: missing true file for {model_name}: {true_path}")
            continue

        print(f"Evaluating model: {model_name}")

        y_pred = np.load(pred_path)
        y_true = np.load(true_path)

        y_pred = normalize_to_bhn(y_pred, "pred", pred_path)
        y_true = normalize_to_bhn(y_true, "true", true_path)

        if args.clip_negative:
            y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        check_prediction_shapes(model_name, y_true, y_pred)

        horizon_rows.extend(
            evaluate_by_horizon(
                model_name=model_name,
                y_true=y_true,
                y_pred=y_pred,
                i0_indices=i0_indices,
            )
        )

        average_rows.append(
            evaluate_average(
                model_name=model_name,
                y_true=y_true,
                y_pred=y_pred,
                i0_indices=i0_indices,
            )
        )

        evaluated_models.append(model_name)

    if len(evaluated_models) == 0:
        available = [
            get_model_name_from_pred_path(p)
            for p in pred_files
        ]

        raise RuntimeError(
            "No models were evaluated. "
            "Check that the names passed to --models exactly match files in prediction_dir before _pred.npy.\n"
            f"Requested models: {args.models}\n"
            f"Available models: {available}"
        )

    horizon_df = pd.DataFrame(horizon_rows)
    avg_df = pd.DataFrame(average_rows)

    # Sort by average MAE for easier reading.
    avg_df = avg_df.sort_values("MAE").reset_index(drop=True)

    # Keep horizon table ordered by average performance and horizon.
    model_order = avg_df["model"].tolist()
    horizon_df["model"] = pd.Categorical(
        horizon_df["model"],
        categories=model_order,
        ordered=True,
    )
    horizon_df = horizon_df.sort_values(["model", "horizon"]).reset_index(drop=True)
    horizon_df["model"] = horizon_df["model"].astype(str)

    # Save CSVs.
    horizon_csv = os.path.join(args.metric_dir, "all_results_by_horizon.csv")
    avg_csv = os.path.join(args.metric_dir, "average_results.csv")

    horizon_df.to_csv(horizon_csv, index=False)
    avg_df.to_csv(avg_csv, index=False)

    print("Saved horizon metrics:", horizon_csv)
    print("Saved average metrics:", avg_csv)

    # Save LaTeX tables.
    avg_latex = os.path.join(args.table_dir, "average_results.tex")
    horizon_latex = os.path.join(args.table_dir, "all_results_by_horizon.tex")

    make_latex_table(
        avg_df=avg_df,
        output_path=avg_latex,
        caption="Average POI forecasting performance across horizons.",
        label="tab:poi_forecasting_average",
        decimals=args.round,
    )

    make_horizon_latex_table(
        horizon_df=horizon_df,
        output_path=horizon_latex,
        caption="POI forecasting performance by forecast horizon.",
        label="tab:poi_forecasting_by_horizon",
        decimals=args.round,
    )

    print("\nEvaluated models:")
    for m in evaluated_models:
        print("  ", m)


if __name__ == "__main__":
    main()