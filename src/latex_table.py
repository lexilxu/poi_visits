# src/latex_table.py

import os
import glob
import numpy as np
import pandas as pd

from src.metrics import evaluate_by_horizon


def collect_results(
    prediction_dir="outputs/predictions",
    y_test_path="outputs/predictions/y_test.npy",
    output_csv="outputs/metrics/all_results.csv",
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    y_test = np.load(y_test_path)

    rows = []

    pred_files = glob.glob(os.path.join(prediction_dir, "*.npy"))

    for path in pred_files:
        name = os.path.basename(path).replace(".npy", "")

        if name == "y_test":
            continue

        pred = np.load(path)

        if pred.shape != y_test.shape:
            print(f"Skipping {name}: shape mismatch {pred.shape} vs {y_test.shape}")
            continue

        # Parse model and adjacency from filename
        # Example: mtgnn_adj_distance
        parts = name.split("_adj_")

        if len(parts) == 2:
            model_name = parts[0]
            adj_name = "adj_" + parts[1]
        else:
            model_name = name
            adj_name = "none"

        df = evaluate_by_horizon(
            y_true=y_test,
            y_pred=pred,
            model_name=model_name,
            adj_name=adj_name,
        )

        rows.append(df)

    results = pd.concat(rows, ignore_index=True)
    results.to_csv(output_csv, index=False)

    print("Saved:", output_csv)

    return results


def make_latex_table(
    results_csv="outputs/metrics/all_results.csv",
    output_tex="outputs/tables/forecasting_table.tex",
    metric="RMSE",
):
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)

    df = pd.read_csv(results_csv)

    df["Model"] = df["model"] + " (" + df["adjacency"] + ")"

    table = df.pivot_table(
        index="Model",
        columns="horizon",
        values=metric,
    )

    table = table.round(3)

    latex = table.to_latex(
        caption=f"Multistep POI forecasting results by horizon using {metric}.",
        label=f"tab:poi_forecasting_{metric.lower()}",
        escape=False,
    )

    with open(output_tex, "w") as f:
        f.write(latex)

    print("Saved:", output_tex)


if __name__ == "__main__":
    results = collect_results()
    make_latex_table(metric="RMSE")
    make_latex_table(metric="MAE", output_tex="outputs/tables/forecasting_table_mae.tex")
    make_latex_table(metric="CORR", output_tex="outputs/tables/forecasting_table_corr.tex")