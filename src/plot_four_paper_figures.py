import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def clean_model_name(name):
    s = str(name)

    if s.lower() == "naive":
        return "Naive"

    if s.lower() in ["ar_ridge", "ar-ridge", "ar ridge"]:
        return "AR-ridge"

    if s.lower() == "lstm":
        return "LSTM"

    if s.startswith("evolvegcn_static_"):
        tail = s.replace("evolvegcn_static_", "")
        tail = re.sub(r"_h\d+", "", tail)
        tail = re.sub(r"_lr[0-9]+", "", tail)

        mapping = {
            "identity": "EvolveGCN (Identity)",
            "distance_knn": "EvolveGCN (KNN)",
            "distance_rbf": "EvolveGCN (KNN-rbf)",
            "top_category": "EvolveGCN (TOP)",
            "naics2": "EvolveGCN (NAICS2)",
            "visit_correlation": "EvolveGCN (CORR)",
        }
        return mapping.get(tail, f"EvolveGCN ({tail})")

    if s.startswith("evolvegcn_dynamic"):
        return "EvolveGCN (Dynamic)"

    if s.startswith("mtgnn_"):
        tail = s.replace("mtgnn_", "")
        tail = re.sub(r"_lr[0-9]+", "", tail)
        tail = re.sub(r"_do[0-9]+", "", tail)
        tail = re.sub(r"_bs[0-9]+", "", tail)

        mapping = {
            "identity": "MTGNN (Identity)",
            "distance_knn": "MTGNN (KNN)",
            "distance_rbf": "MTGNN (KNN-rbf)",
            "top_category": "MTGNN (TOP)",
            "naics2": "MTGNN (NAICS2)",
            "visit_correlation": "MTGNN (CORR)",
        }
        return mapping.get(tail, f"MTGNN ({tail})")

    return s


def model_family(model_clean):
    if model_clean in ["Naive", "AR-ridge", "LSTM"]:
        return "Baseline"
    if model_clean.startswith("EvolveGCN"):
        return "EvolveGCN"
    if model_clean.startswith("MTGNN"):
        return "MTGNN"
    return "Other"


def adjacency_type(model_clean):
    if model_clean in ["Naive", "AR-ridge", "LSTM"]:
        return "Baseline"

    m = re.search(r"\((.*?)\)", model_clean)
    if m:
        return m.group(1)

    return "Other"


def short_label(model_clean):
    mapping = {
        "Naive": "Naive",
        "AR-ridge": "AR",
        "LSTM": "LSTM",
        "EvolveGCN (Identity)": "Evo-Id",
        "EvolveGCN (KNN)": "Evo-KNN",
        "EvolveGCN (KNN-rbf)": "Evo-RBF",
        "EvolveGCN (NAICS2)": "Evo-NAICS",
        "EvolveGCN (TOP)": "Evo-TOP",
        "EvolveGCN (CORR)": "Evo-CORR",
        "EvolveGCN (Dynamic)": "Evo-Dyn",
        "MTGNN (Identity)": "MT-Id",
        "MTGNN (KNN)": "MT-KNN",
        "MTGNN (KNN-rbf)": "MT-RBF",
        "MTGNN (NAICS2)": "MT-NAICS",
        "MTGNN (TOP)": "MT-TOP",
        "MTGNN (CORR)": "MT-CORR",
    }
    return mapping.get(model_clean, model_clean)


def load_results(path):
    df = pd.read_csv(path)

    if "horizon" not in df.columns:
        if "test|horizon" in df.columns:
            df = df.rename(columns={"test|horizon": "horizon"})
        elif "Horizon" in df.columns:
            df = df.rename(columns={"Horizon": "horizon"})

    required = ["model", "horizon", "MAE", "RMSE", "RSE", "RAE", "CORR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["horizon_num"] = pd.to_numeric(df["horizon"], errors="coerce")
    df = df[df["horizon_num"].notna()].copy()
    df["horizon"] = df["horizon_num"].astype(int)
    df = df.drop(columns=["horizon_num"])

    df["model_clean"] = df["model"].apply(clean_model_name)
    df["family"] = df["model_clean"].apply(model_family)
    df["adjacency"] = df["model_clean"].apply(adjacency_type)

    return df


def average_results(df):
    return (
        df.groupby(["model_clean", "family", "adjacency"], as_index=False)[["MAE", "RMSE", "RSE", "RAE", "CORR"]]
        .mean()
    )


def select_best_graph_models(df):
    """
    Keep AR-ridge and LSTM if available, then add the best EvolveGCN
    and best MTGNN based on average MAE.
    Naive is intentionally excluded.
    """
    avg = average_results(df)

    selected = []
    for x in ["AR-ridge", "LSTM"]:
        if x in avg["model_clean"].values:
            selected.append(x)

    evol = avg[avg["family"] == "EvolveGCN"]
    if len(evol) > 0:
        selected.append(evol.sort_values("MAE").iloc[0]["model_clean"])

    mtg = avg[avg["family"] == "MTGNN"]
    if len(mtg) > 0:
        selected.append(mtg.sort_values("MAE").iloc[0]["model_clean"])

    return selected


def plot_selected_models_mae_grouped_bar(df, output_path):
    selected = select_best_graph_models(df)
    plot_df = df[df["model_clean"].isin(selected)].copy()

    horizons = sorted(plot_df["horizon"].unique())
    models = selected

    pivot = plot_df.pivot_table(index="horizon", columns="model_clean", values="MAE")
    pivot = pivot[models]

    x = np.arange(len(horizons))
    width = 0.18

    plt.figure(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        plt.bar(
            x + (i - (len(models) - 1) / 2) * width,
            pivot[model].values,
            width=width,
            label=model,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5
        )

    plt.xticks(x, horizons)
    plt.xlabel("Forecast horizon")
    plt.ylabel("MAE")
    plt.title("MAE across horizons for selected models")
    plt.legend(frameon=False, ncol=2)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_graph_adjacency_heatmaps(df, output_path):
    graph_df = df[df["family"].isin(["EvolveGCN", "MTGNN"])].copy()

    families = ["EvolveGCN", "MTGNN"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax, fam in zip(axes, families):
        sub = graph_df[graph_df["family"] == fam].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        order = (
            sub.groupby("adjacency")["MAE"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

        pivot = sub.pivot_table(index="adjacency", columns="horizon", values="MAE")
        pivot = pivot.loc[order]

        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Forecast horizon")
        ax.set_title(f"{fam}: MAE by adjacency and horizon")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(
                    j, i, f"{pivot.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8
                )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("MAE")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def pareto_frontier_mae_corr(df_avg):
    pts = df_avg[["MAE", "CORR"]].values
    keep = []

    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]) and \
               (pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]):
                dominated = True
                break
        if not dominated:
            keep.append(i)

    frontier = df_avg.iloc[keep].copy().sort_values("MAE")
    return frontier


def plot_tradeoff_all_models(df, output_path):
    avg = average_results(df)

    family_color = {
        "Baseline": "#4c78a8",
        "EvolveGCN": "#f58518",
        "MTGNN": "#54a24b",
        "Other": "#777777",
    }

    marker_map = {
        "Baseline": "o",
        "Identity": "s",
        "KNN": "^",
        "KNN-rbf": "D",
        "NAICS2": "P",
        "TOP": "X",
        "CORR": "v",
        "Dynamic": "*",
        "Other": "o",
    }

    plt.figure(figsize=(8, 6))

    for _, row in avg.iterrows():
        plt.scatter(
            row["MAE"],
            row["CORR"],
            s=110,
            color=family_color.get(row["family"], "#777777"),
            marker=marker_map.get(row["adjacency"], "o"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9
        )

        plt.annotate(
            short_label(row["model_clean"]),
            (row["MAE"], row["CORR"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    frontier = pareto_frontier_mae_corr(avg)
    if len(frontier) >= 2:
        plt.plot(
            frontier["MAE"],
            frontier["CORR"],
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.8,
            label="Pareto frontier"
        )

    plt.xlabel("Average MAE, lower is better")
    plt.ylabel("Average correlation, higher is better")
    plt.title("Tradeoff between forecasting accuracy and correlation")
    plt.grid(True, alpha=0.25)

    family_handles = [
        Line2D([0], [0], marker="o", color="w", label=fam,
               markerfacecolor=col, markeredgecolor="black", markersize=8)
        for fam, col in family_color.items() if fam in avg["family"].unique()
    ]

    adjacency_handles = []
    seen_adj = [a for a in marker_map.keys() if a in avg["adjacency"].unique()]
    for adj in seen_adj:
        adjacency_handles.append(
            Line2D([0], [0], marker=marker_map[adj], color="black", label=adj,
                   linestyle="None", markersize=8)
        )

    leg1 = plt.legend(handles=family_handles, title="Model family",
                      loc="lower left", frameon=False)
    plt.gca().add_artist(leg1)
    plt.legend(handles=adjacency_handles, title="Adjacency type",
               loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_average_mae_bar(df, output_path):
    """
    Replacement for the old improvement-over-naive figure.
    Shows average MAE for all selected models.
    """
    avg = average_results(df).copy()
    avg = avg.sort_values("MAE")

    plt.figure(figsize=(9, 5))
    colors = avg["family"].map({
        "Baseline": "#4c78a8",
        "EvolveGCN": "#f58518",
        "MTGNN": "#54a24b",
        "Other": "#777777",
    })

    plt.barh(avg["model_clean"], avg["MAE"], color=colors, edgecolor="black", linewidth=0.5)
    plt.xlabel("Average MAE")
    plt.ylabel("Model")
    plt.title("Average MAE across horizons")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/metrics/all_results_by_horizon.csv",
        help="Path to horizon-level results CSV."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/paper_figures_v2",
        help="Directory for output figures."
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_plot_style()

    df = load_results(args.input)

    fig1 = os.path.join(args.output_dir, "figure1_selected_models_mae_grouped_bar.png")
    fig2 = os.path.join(args.output_dir, "figure2_graph_adjacency_heatmaps.png")
    fig3 = os.path.join(args.output_dir, "figure3_tradeoff_all_models.png")
    fig4 = os.path.join(args.output_dir, "figure4_average_mae_bar.png")

    plot_selected_models_mae_grouped_bar(df, fig1)
    plot_graph_adjacency_heatmaps(df, fig2)
    plot_tradeoff_all_models(df, fig3)
    plot_average_mae_bar(df, fig4)

    print("Saved figures:")
    print(fig1)
    print(fig2)
    print(fig3)
    print(fig4)


if __name__ == "__main__":
    main()