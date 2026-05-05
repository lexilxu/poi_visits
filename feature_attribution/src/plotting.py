import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from feature_attribution.src.io_utils import load_yaml, ensure_output_dirs


def plot_lag_importance(csv_path, output_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 4))
    plt.plot(df["lag_index"], df["importance"], marker="o")
    plt.xlabel("Historical lag index")
    plt.ylabel("Mean absolute attribution")
    plt.title("Lag importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_categories(csv_path, output_path, top_k=15):
    df = pd.read_csv(csv_path)
    df = df.head(top_k).copy()

    label_col = df.columns[0]

    plt.figure(figsize=(8, 5))
    plt.barh(df[label_col].astype(str), df["mean_importance"])
    plt.xlabel("Mean attribution per POI")
    plt.ylabel(label_col)
    plt.title("Top category importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="feature_attribution/configs/attribution_config.yaml")
    parser.add_argument("--lag_csv", type=str, default=None)
    parser.add_argument("--category_csv", type=str, default=None)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ensure_output_dirs(cfg)

    plot_dir = cfg["outputs"]["plot_dir"]

    if args.lag_csv is not None:
        plot_lag_importance(
            args.lag_csv,
            os.path.join(plot_dir, f"{args.prefix}_lag_importance.png"),
        )

    if args.category_csv is not None:
        plot_top_categories(
            args.category_csv,
            os.path.join(plot_dir, f"{args.prefix}_category_importance.png"),
        )


if __name__ == "__main__":
    main()