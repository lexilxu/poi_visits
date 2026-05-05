# select_best_mtgnn_static.py

import os
import re
import pandas as pd


def parse_model_name(model):
    """
    Expected:
        mtgnn_{adj}_lr{lr}_do{dropout}_bs{batch}
    Example:
        mtgnn_distance_knn_lr001_do3_bs8
    """
    if not model.startswith("mtgnn_"):
        return None

    body = model.replace("mtgnn_", "")

    m = re.search(r"_lr(.+)_do(.+)_bs(\d+)$", body)
    if m is None:
        return None

    adj = body[:m.start()]
    return {
        "adjacency": adj,
        "lr_tag": m.group(1),
        "dropout_tag": m.group(2),
        "batch_size": int(m.group(3)),
    }


def main():
    results_path = "outputs/metrics/all_results_by_horizon.csv"
    out_path = "outputs/metrics/mtgnn_static_tuning_summary.csv"
    best_path = "outputs/metrics/mtgnn_static_best_by_adjacency.csv"

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Missing {results_path}. Run src/evaluate_all.py first."
        )

    df = pd.read_csv(results_path)

    rows = []

    for model, g in df.groupby("model"):
        parsed = parse_model_name(model)
        if parsed is None:
            continue

        avg = {
            "model": model,
            "MAE": g["MAE"].mean(),
            "RMSE": g["RMSE"].mean(),
            "RSE": g["RSE"].mean(),
            "RAE": g["RAE"].mean(),
            "CORR": g["CORR"].mean(),
        }
        avg.update(parsed)
        rows.append(avg)

    if len(rows) == 0:
        raise RuntimeError("No MTGNN tuning models found.")

    out = pd.DataFrame(rows)
    out = out.sort_values(["adjacency", "MAE", "RMSE"])
    out.to_csv(out_path, index=False)

    best_rows = []
    for adj, group in out.groupby("adjacency"):
        best = group.sort_values(["MAE", "RMSE"]).iloc[0]
        best_rows.append(best)

    best = pd.DataFrame(best_rows).sort_values("MAE")
    best.to_csv(best_path, index=False)

    print("\nSaved MTGNN tuning summary:")
    print(out_path)

    print("\nSaved best MTGNN config per adjacency:")
    print(best_path)

    cols = [
        "adjacency",
        "model",
        "lr_tag",
        "dropout_tag",
        "batch_size",
        "MAE",
        "RMSE",
        "RSE",
        "RAE",
        "CORR",
    ]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
