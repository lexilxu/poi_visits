import os
import argparse
import numpy as np
import pandas as pd

from feature_attribution.src.io_utils import (
    load_yaml,
    ensure_output_dirs,
    load_poi_meta,
    save_dataframe,
)


def summarize_lag_importance(attr):
    """
    attr: (B, P, N)
    """
    scores = np.mean(np.abs(attr), axis=(0, 2))
    return pd.DataFrame({
        "lag_index": np.arange(1, len(scores) + 1),
        "importance": scores,
    })


def summarize_poi_importance(attr, poi_meta):
    """
    attr: (B, P, N)
    """
    scores = np.mean(np.abs(attr), axis=(0, 1))

    out = poi_meta.copy()
    out["poi_index"] = np.arange(len(out))
    out["importance"] = scores
    out = out.sort_values("importance", ascending=False)

    return out


def summarize_category_importance(attr, poi_meta, category_col="top_category"):
    scores = np.mean(np.abs(attr), axis=(0, 1))

    df = poi_meta.copy()
    df["importance"] = scores

    if category_col not in df.columns:
        raise ValueError(f"Column {category_col} not found in POI metadata")

    out = (
        df.groupby(category_col, dropna=False)
        .agg(
            total_importance=("importance", "sum"),
            mean_importance=("importance", "mean"),
            num_pois=("importance", "size"),
        )
        .reset_index()
        .sort_values("mean_importance", ascending=False)
    )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="feature_attribution/configs/attribution_config.yaml")
    parser.add_argument("--attr_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_name", type=str, required=True)
    parser.add_argument("--category_col", type=str, default="top_category")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ensure_output_dirs(cfg)

    attr = np.load(args.attr_path)
    poi_meta = load_poi_meta(
        cfg["data"]["poi_meta_path"],
        cfg["data"].get("location_table_path", None),
    )

    summary_dir = cfg["outputs"]["summary_dir"]

    lag_df = summarize_lag_importance(attr)
    poi_df = summarize_poi_importance(attr, poi_meta)
    cat_df = summarize_category_importance(attr, poi_meta, category_col=args.category_col)

    prefix = f"{args.model_name}_{args.target_name}"

    save_dataframe(
        os.path.join(summary_dir, f"{prefix}_lag_importance.csv"),
        lag_df,
    )

    save_dataframe(
        os.path.join(summary_dir, f"{prefix}_poi_importance.csv"),
        poi_df,
    )

    save_dataframe(
        os.path.join(summary_dir, f"{prefix}_category_importance.csv"),
        cat_df,
    )

    print("Saved summaries for", prefix)


if __name__ == "__main__":
    main()