# select_best_evolvegcn_static.py

import os
import glob
import re
import pandas as pd


def parse_model_name(model_name):
    """
    Expected name:
        evolvegcn_static_{adj}_h{hidden}_lr{lr_tag}

    Examples:
        evolvegcn_static_identity_h16_lr001
        evolvegcn_static_distance_knn_h32_lr0005
    """
    if not model_name.startswith("evolvegcn_static_"):
        return None

    body = model_name.replace("evolvegcn_static_", "")

    match = re.search(r"_h(\d+)_lr(.+)$", body)
    if match is None:
        return None

    hidden_dim = int(match.group(1))
    lr_tag = match.group(2)

    adj_name = body[: match.start()]

    return {
        "adjacency": adj_name,
        "hidden_dim": hidden_dim,
        "lr_tag": lr_tag,
    }


def main():
    metric_dir = "outputs/validation"
    out_path = "outputs/validation/evolvegcn_static_tuning_summary.csv"
    best_path = "outputs/validation/evolvegcn_static_best_by_adjacency.csv"

    rows = []

    for path in glob.glob(os.path.join(metric_dir, "evolvegcn_static_*_val_metrics.csv")):
        df = pd.read_csv(path)

        avg = df[df["horizon"].astype(str) == "avg"]
        if len(avg) == 0:
            continue

        row = avg.iloc[0].to_dict()
        model_name = row["model"]

        parsed = parse_model_name(model_name)
        if parsed is None:
            continue

        row.update(parsed)
        row["file"] = path
        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No EvolveGCN static validation metrics found.")

    out = pd.DataFrame(rows)

    # Sort mainly by validation MAE.
    out = out.sort_values(["adjacency", "MAE", "RMSE"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    best_rows = []
    for adj, group in out.groupby("adjacency"):
        best = group.sort_values(["MAE", "RMSE"]).iloc[0]
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values("MAE")
    best_df.to_csv(best_path, index=False)

    print("\nSaved full tuning summary:")
    print(out_path)

    print("\nSaved best configuration per adjacency:")
    print(best_path)

    print("\nBest per adjacency, sorted by validation MAE:")
    cols = [
        "adjacency",
        "model",
        "hidden_dim",
        "lr_tag",
        "MAE",
        "RMSE",
        "RSE",
        "RAE",
        "CORR",
    ]
    print(best_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
