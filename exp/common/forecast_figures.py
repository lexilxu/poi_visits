"""Publication layouts for two existing POI experiment figures.

Use python -m exp exp1 plot or python -m exp exp2 plot.
Inputs and plotted estimates are unchanged. Both figures use one linear
coordinate system, consistent model colors, and redundant marker encoding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np


ROOT = Path(__file__).resolve().parent
WIDTH = 9.4
# Hex values printed in one palette in the user's Nature reference article.
# https://zhuanlan.zhihu.com/p/704390338
# Reference image: v2-f9e9846d92a129cd688c6cbb7385f953_r.jpg
BLUE = "#5F97C6"
LIGHT_BLUE = "#A3CDEA"
CORAL = "#F09496"
SAGE = "#AFC778"
TAUPE = "#947959"
GRAY = "#9A9A9A"
INK = "#262626"
LEADER = "#B0B0B0"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    family = "Arial"
    try:
        findfont(FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 12, "axes.labelsize": 14,
        "axes.labelweight": "normal", "xtick.labelsize": 12,
        "ytick.labelsize": 12, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.65,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans", "lines.solid_capstyle": "round",
    })


def clean(ax):
    ax.tick_params(length=3.5, width=0.65, pad=5)
    ax.set_axisbelow(True)


def annotate(ax, xy, label_xy, text, ha="left", size=12, color=LEADER):
    return ax.annotate(
        text, xy=xy, xytext=label_xy, ha=ha, va="center", fontsize=size,
        color=INK, annotation_clip=False, zorder=8,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55,
                    "shrinkA": 3, "shrinkB": 4},
    )


def check_layout(fig):
    """Check text bounds and text-on-text intersections at rendering size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundary = fig.bbox
    records = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = Text.get_window_extent(artist, renderer)
        if box.width > 0 and box.height > 0:
            records.append((artist.get_text(), box))
    outside = [s for s, b in records if b.x0 < -1 or b.y0 < -1
               or b.x1 > boundary.x1 + 1 or b.y1 > boundary.y1 + 1]
    overlaps = []
    for i, (a, ba) in enumerate(records):
        for b, bb in records[i + 1:]:
            intersection = Bbox.intersection(ba, bb)
            if intersection is not None and min(intersection.width, intersection.height) > 2.0:
                overlaps.append([a, b])
    return {"outside": outside, "text_overlaps": overlaps, "text_items": len(records)}


def horizon_figure(rows):
    fig = plt.figure(figsize=(WIDTH, 4.8))
    ax = fig.add_axes([0.105, 0.165, 0.860, 0.655])
    specifications = [
        ("EvolveGCN (identity)", BLUE, "o", "-", True),
        ("EvolveGCN (best non-identity)", LIGHT_BLUE, "s", (0, (3.5, 2.2)), True),
        ("MTGNN (best)", CORAL, "^", "-", True),
        ("Historical average", GRAY, None, (0, (1.6, 2.8)), False),
        ("AR-ridge", SAGE, "s", "-", True),
        ("LSTM", TAUPE, "D", "-", True),
    ]
    plotted = []
    for name, color, marker, linestyle, filled in specifications:
        selected = sorted((r for r in rows if r["series"] == name), key=lambda r: int(r["H"]))
        x = np.array([int(r["H"]) for r in selected])
        y = np.array([float(r["skill_percent"]) for r in selected])
        assert x.tolist() == [7, 14, 28]
        assert all(int(r["T"]) == 28 for r in selected)
        line, = ax.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.55 if marker else 0.85, markersize=5.6,
                        markeredgewidth=0.55, markeredgecolor="white" if filled else color,
                        markerfacecolor=color if filled else "white", zorder=4,
                        label=name)
        assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y)
        plotted.append({"series": name, "horizons": x.tolist(), "skill_percent": y.tolist()})
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.535, 0.997), frameon=False, ncol=3,
               fontsize=12, handlelength=2.6, handletextpad=0.65,
               columnspacing=2.2, labelspacing=0.85)
    ax.set(xlim=(5.8, 29.2), ylim=(-42, 24), xlabel=r"Forecast horizon $H$ (days)",
           ylabel="MAE skill (%)")
    ax.set_xticks([7, 14, 28])
    ax.set_yticks([-40, -30, -20, -10, 0, 10, 20])
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    clean(ax)
    return fig, {"axes": len(fig.axes), "data_points": len(rows), "plotted": plotted,
                 "size_inches": [WIDTH, 4.8]}


def point_style(model):
    if model == "EvolveGCN":
        return dict(marker="o", color=BLUE, markerfacecolor=BLUE, markeredgecolor="white",
                    markeredgewidth=0.6, markersize=6.5)
    if model == "MTGNN":
        return dict(marker="^", color=CORAL, markerfacecolor=CORAL,
                    markeredgecolor="white", markeredgewidth=0.6, markersize=7.0)
    if model == "AR-ridge":
        return dict(marker="s", color=SAGE, markerfacecolor=SAGE, markeredgecolor="white",
                    markeredgewidth=0.6, markersize=5.7)
    if model == "LSTM":
        return dict(marker="D", color=TAUPE, markerfacecolor=TAUPE, markeredgecolor="white",
                    markeredgewidth=0.6, markersize=5.7)
    return dict(marker="+", color=GRAY, markerfacecolor="none", markeredgecolor=GRAY,
                markeredgewidth=1.1, markersize=6.8)


def recovery_figure(rows):
    fig = plt.figure(figsize=(WIDTH, 5.0))
    ax = fig.add_axes([0.105, 0.160, 0.860, 0.715])
    legend = [Line2D([], [], linestyle="none", label=model, **point_style(model))
              for model in ["EvolveGCN", "MTGNN", "Historical average", "AR-ridge", "LSTM"]]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.535, 0.987),
               frameon=False, ncol=5, fontsize=12, handlelength=1.0,
               handletextpad=0.5, columnspacing=1.75)
    main_labels = {
        "Identity": (17.7, 41.2, "center"),
        "NAICS-2": (18.6, 30.6, "right"),
        "Top category": (25.8, 44.3, "center"),
        "Visit correlation": (22.4, 31.7, "left"),
        "Distance RBF": (27.3, 36.7, "left"),
        "Distance KNN": (31.6, 47.9, "center"),
    }
    left_labels = {
        ("Historical average", "None"): (2.4, 24.1, "left"),
        ("LSTM", "None"): (3.0, 20.7, "left"),
        ("AR-ridge", "None"): (3.0, 17.4, "left"),
        ("MTGNN", "Identity"): (3.1, 3.7, "left"),
        ("MTGNN", "Top category"): (3.1, 7.1, "left"),
        ("MTGNN", "NAICS-2"): (3.1, 10.5, "left"),
        ("MTGNN", "Distance KNN"): (3.1, 13.9, "left"),
    }
    plotted = []
    for row in rows:
        xy = (100 * float(row["far_0_5"]), 100 * float(row["dmr_model"]))
        options = point_style(row["model"])
        line, = ax.plot(*xy, linestyle="none", zorder=5, **options)
        assert np.isclose(line.get_xdata()[0], xy[0]) and np.isclose(line.get_ydata()[0], xy[1])
        name = row["model"] if row["adjacency"] == "None" else row["adjacency"]
        if row["model"] == "EvolveGCN":
            x, y, ha = main_labels[row["adjacency"]]
        else:
            x, y, ha = left_labels[(row["model"], row["adjacency"])]
        annotate(ax, xy, (x, y), name, ha=ha)
        plotted.append({"model": row["model"], "adjacency": row["adjacency"],
                        "far_percent": xy[0], "dmr_percent": xy[1]})
    assert len(rows) == 13 and len(fig.axes) == 1
    ax.set(xlim=(-1.5, 36.5), ylim=(0, 51), xlabel="False activation rate (%)",
           ylabel="Demand mass recovery ratio (%)")
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    clean(ax)
    return fig, {"axes": len(fig.axes), "data_points": len(rows),
                 "plotted": plotted, "size_inches": [WIDTH, 5.0]}


def export(fig, dest, qa_folder, dpi):
    result = check_layout(fig)
    # Save a review preview even if a draft still needs label adjustments.
    fig.savefig(qa_folder / f"{dest.stem}.png", dpi=160)
    if result["outside"] or result["text_overlaps"]:
        plt.close(fig)
        raise ValueError(f"{dest.stem}: {result}")
    for extension in ["pdf", "svg", "png"]:
        options = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(dest.with_suffix("." + extension), **options)
    plt.close(fig)
    return result


def render(kind, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Exported aggregate CSV')
    parser.add_argument('--out', type=Path, required=True, help='Figure output directory')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args(argv)
    rows = read_csv(args.data)
    if kind == 'horizons':
        for row in rows:
            reference = next(r for r in rows if r['H'] == row['H'] and r['series'] == 'Historical average')
            assert np.isclose(100 * (1 - float(row['mae']) / float(reference['mae'])), float(row['skill_percent']), atol=1e-9)
        plot, stem = horizon_figure, 'forecasting_performance_across_horizons'
    else:
        for row in rows:
            assert 0 <= float(row['far_0_5']) <= 1
            assert np.isclose(float(row['dmr_model']) - float(row['dmr_ha']), float(row['delta_dmr']), atol=1e-7)
        plot, stem = recovery_figure, 'recovery_activation_tradeoff'
    before = sha256(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    preview = args.out/'preview'; preview.mkdir(exist_ok=True)
    style()
    fig, detail = plot(rows)
    destination = args.out/(stem+'.pdf')
    detail.update(layout=export(fig, destination, preview, args.dpi),
                  input_sha256=before, output_sha256=sha256(destination), raster_dpi=args.dpi)
    assert before == sha256(args.data)
    (args.out/(stem+'_audit.json')).write_text(json.dumps(detail, indent=2)+'\n', encoding='utf-8')
    print(destination)
