"""Render the formal-run PDFs from compact, verified experiment results."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_figures as base
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ticks_for(low, high, count):
    return [t for t in MaxNLocator(nbins=count).tick_values(low, high)
            if low <= t <= high]


def recovery():
    data = pd.read_csv(base.DATA/"01_recovery_activation.csv", keep_default_na=False)
    fig = plt.figure(figsize=(base.WIDTH, 4.02))
    ax = fig.add_axes([.126, .147, .846, .754])
    styles = {
        "EvolveGCN": dict(marker="o", ms=7.2, mec="#355878", mfc="#A2C4F1", mew=1.0),
        "MTGNN": dict(marker="^", ms=7.2, mec="#8D3E50", mfc="#FAA09C", mew=.7),
        "Benchmark": dict(marker="D", ms=5.2, mec="#424A52", mfc="#535E66", mew=.45),
    }
    layout = {
        ("Historical average", "None"): (4.25, 33.9, "left"),
        ("LSTM", "None"): (7.0, 24.5, "left"),
        ("AR-ridge", "None"): (7.0, 19.7, "left"),
        ("MTGNN", "Distance KNN"): (10.0, 14.9, "left"),
        ("MTGNN", "NAICS-2"): (10.0, 11.2, "left"),
        ("MTGNN", "Top category"): (10.0, 7.5, "left"),
        ("MTGNN", "Identity"): (10.0, 3.8, "left"),
        ("EvolveGCN", "Identity"): (15.8, 39.0, "right"),
        ("EvolveGCN", "NAICS-2"): (18.6, 30.6, "right"),
        ("EvolveGCN", "Visit correlation"): (24.0, 31.4, "left"),
        ("EvolveGCN", "Top category"): (23.9, 44.0, "right"),
        ("EvolveGCN", "Distance RBF"): (30.0, 37.6, "left"),
        ("EvolveGCN", "Distance KNN"): (29.0, 46.9, "right"),
    }
    marks = []
    for row in data.itertuples():
        xy = (100*row.far_0_5, 100*row.dmr_model)
        marker_style = styles.get(row.model, styles["Benchmark"])
        ax.plot(*xy, ls="none", zorder=5, **marker_style)
        name = row.model if row.adjacency == "None" else row.adjacency
        marks.append(xy)
        x, y, ha = layout[(row.model, row.adjacency)]
        base.label(ax, xy, (x, y), name, ha, size=11.0, color=marker_style["mec"],
                   marker_pad=marker_style["ms"]/2+.6)
    coordinates = np.array(marks)
    span_x = max(coordinates[:, 0].max(), 10)
    min_y, max_y = coordinates[:, 1].min(), coordinates[:, 1].max()
    span_y = max(max_y-min_y, 12)
    xlim = (-.045*span_x, 1.18*span_x)
    ylim = (min_y-.14*span_y, max_y+.14*span_y)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_xlabel("False activation rate (%)", fontsize=12.8, labelpad=6)
    ax.set_ylabel("Demand mass recovery ratio (%)", fontsize=12.8, labelpad=8)
    ax.set_xticks([x for x in ticks_for(*xlim, 6) if 0 <= x <= 100])
    ax.set_yticks(ticks_for(*ylim, 6))
    base.clean(ax)
    ax.tick_params(labelsize=10.6)
    keys = [Line2D([], [], ls="none", label=key, **(styles[key] | {"ms": 8.2}))
            for key in ("EvolveGCN", "MTGNN")]
    with plt.rc_context({"legend.fontsize": 11.0}):
        base.small_legend(fig, keys, .126, .975)
    base.output(fig, "01", "Recovery-activation tradeoff")


def horizons():
    rows = pd.read_csv(base.DATA/"07_horizon_skill.csv")
    assert rows["T"].eq(28).all()
    fig = plt.figure(figsize=(base.WIDTH, 4.02))
    ax = fig.add_axes([.113, .281, .851, .670])
    legend_ax = fig.add_axes([.075, .025, .900, .111])
    legend_ax.set_axis_off()
    styles = {
        "EvolveGCN (identity)": (base.BLUE, "o", "-", True, 3.7),
        "EvolveGCN (best non-identity)": (base.BLUE, "s", (0, (4, 2)), False, 3.7),
        "MTGNN (best)": (base.TEAL, "^", (0, (5, 2, 1.2, 2)), True, 5.8),
        "Historical average": (base.GRAY, None, (0, (2, 2)), True, 0),
        "AR-ridge": (base.RUST, "v", "-", True, 5.8),
        "LSTM": (base.PLUM, "D", "-", False, 5.8),
    }
    min_value, max_value = rows.skill_percent.min(), rows.skill_percent.max()
    spread = max(max_value-min_value, 6)
    lower, upper = min_value-.085*spread, max_value+.105*spread
    ax.set(xlim=(5.7, 29.3), ylim=(lower, upper))
    ax.set_xlabel(r"Forecast horizon $H$ (days)", fontsize=11.7, labelpad=7)
    ax.set_ylabel("MAE skill (%)", fontsize=11.7, labelpad=8)
    ax.set_xticks([7, 14, 28])
    ax.set_yticks(ticks_for(lower, upper, 7))
    ax.grid(axis="y", color="#E1E6EA", linewidth=.55)
    base.clean(ax)
    ax.tick_params(labelsize=10.4)
    handles = []
    for series, (color, marker, ls, filled, marker_size) in styles.items():
        data = rows.loc[rows.series.eq(series)].sort_values("H")
        assert data.H.tolist() == [7, 14, 28]
        line, = ax.plot(data.H, data.skill_percent, label=series, color=color, ls=ls,
                        marker=marker, ms=marker_size, mew=.85, lw=1.75,
                        mfc=color if filled else "white", zorder=4 if marker else 2)
        handles.append(line)
    legend_ax.legend(handles=handles, loc="center", ncols=3, frameon=False,
                     fontsize=10.4, handlelength=2.5, handletextpad=.60,
                     columnspacing=1.55, labelspacing=.65, borderaxespad=0)
    base.output(fig, "07", "Forecasting performance across horizons")


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", choices=("01", "02", "03", "04", "07"))
    args = p.parse_args()
    manifest = json.loads((args.results/"dataset_manifest.json").read_text(encoding="utf-8"))
    assert (manifest["date_start"], manifest["date_end"], manifest["nodes"]) == ("2020-10-01", "2020-12-31", 9884)
    base.DATA, base.ART = args.results, args.out
    base.ART.mkdir(parents=True, exist_ok=True)
    base.style()
    plots = {"01": recovery, "02": base.outcome_heatmaps, "03": base.perturbation,
             "04": base.fingerprint, "07": horizons}
    for key in args.only or plots:
        plots[key]()


if __name__ == "__main__":
    main()
