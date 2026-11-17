"""Plot observed zero-to-positive visit sequences and their cohort prevalence.

Run: python -m exp exp0 plot-sparsity --help

A completed zero gap contains one or more consecutive observed zero days
between two positive-visit days. Missing days break sequences. Leading and
trailing zero runs are excluded because one positive endpoint is unobserved.
Both examples and cohort summaries use the full descriptive study period.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np


HERE = Path(__file__).resolve().parent
from exp.common.paths import DATA_ROOT, RESULTS_ROOT
STEM = "poi_temporal_sparsity"
FIGURE_WIDTH = 9.4
FIGURE_HEIGHT = 5.75
BLUE = "#3F647C"
BROWN = "#A76C51"
PALE = "#F3E9E2"
INK = "#252B30"
GRAY = "#8C949A"
GRID = "#E4E7E9"


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def completed_gaps(series):
    """Return (zero_days, preceding_positive_day, following_positive_day)."""
    positive = np.flatnonzero(series > 0)
    result = []
    for previous, following in zip(positive[:-1], positive[1:]):
        if following > previous + 1 and np.isfinite(series[previous:following + 1]).all():
            assert np.all(series[previous + 1:following] == 0)
            result.append((int(following - previous - 1), int(previous), int(following)))
    return result


def style():
    family = "Arial"
    try:
        findfont(FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 12, "axes.labelsize": 12.5,
        "xtick.labelsize": 11.2, "ytick.labelsize": 11.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150,
    })


def choose_examples(visits, ratios, longest, keys):
    """Select without manual searching for visually extreme visit sequences."""
    complete = np.isfinite(visits).all(axis=0)
    examples = []
    for percentile, label in zip([10, 25, 50], ["A", "B", "C"]):
        target = float(np.percentile(ratios, percentile))
        distance = np.abs(ratios - target)
        nearest = np.min(distance[complete])
        candidates = np.flatnonzero(complete & np.isclose(distance, nearest, atol=1e-12, rtol=0))
        median_gap = float(np.median(longest[candidates]))
        chosen = min(candidates, key=lambda j: (abs(longest[j] - median_gap), keys[j]))
        examples.append({
            "label": f"POI {label}", "column_index": int(chosen),
            "placekey": keys[chosen], "target_nonzero_ratio_percentile": percentile,
            "nonzero_ratio_percent": float(100 * ratios[chosen]),
            "candidate_pois": int(len(candidates)),
            "candidate_median_longest_completed_zero_gap": median_gap,
        })
    return examples


def make_figure(visits, dates, examples, gaps, longest, counts, stats):
    style()
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.text(0.090, 0.953, "(a)  Daily visit sequences", fontsize=14.5, weight="bold")
    fig.legend(handles=[Patch(facecolor=BLUE, label="Positive visits"),
                        Patch(facecolor=BROWN, label="After highlighted zero gap")],
               loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False,
               ncol=2, fontsize=11, handlelength=1.0, columnspacing=1.2)

    day = np.arange(visits.shape[0])
    ticks = [0, 14, 31, 45, 61, 75, 91]
    tick_labels = [date.fromisoformat(dates[i]).strftime("%b %d").replace(" 0", " ") for i in ticks]
    for i, example in enumerate(examples):
        j = example["column_index"]
        values = visits[:, j]
        ax = fig.add_axes([0.090, 0.790 - 0.160 * i, 0.885, 0.110])
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.45)
        ax.tick_params(length=2.5, width=0.6, pad=3)
        chosen_gap = max(gaps[j], key=lambda item: (item[0], -item[1]))
        length, previous, following = chosen_gap
        example["highlighted_gap_days"] = length
        example["highlighted_zero_start"] = dates[previous + 1]
        example["highlighted_zero_end"] = dates[following - 1]
        example["following_positive_date"] = dates[following]
        example["following_positive_visits"] = int(values[following])
        ax.axvspan(previous + 0.5, following - 0.5, color=PALE, zorder=0)
        colors = np.full(len(day), BLUE, dtype=object)
        colors[following] = BROWN
        positive = values > 0
        ax.bar(day[positive], values[positive], width=0.66, color=colors[positive],
               linewidth=0, zorder=3)
        ax.scatter(day[~positive], np.zeros((~positive).sum()), s=4.0,
                   color=GRAY, linewidths=0, clip_on=False, zorder=4)
        ymax = max(2.0, float(values.max()))
        ax.set(xlim=(-0.8, len(day) - 0.2), ylim=(-0.05 * ymax, 1.70 * ymax))
        ax.set_yticks([0, 1, 2] if ymax <= 2 else [0, 2, 4])
        ax.set_xticks(ticks, tick_labels if i == 2 else [""] * len(ticks))
        ax.text(0, 1.11,
                f"{example['label']}   |   Nonzero ratio: {example['nonzero_ratio_percent']:.1f}%",
                transform=ax.transAxes, fontsize=12, va="bottom")
        bracket_y = 1.20 * ymax
        ax.plot([previous + 0.5, previous + 0.5, following - 0.5, following - 0.5],
                [1.06 * ymax, bracket_y, bracket_y, 1.06 * ymax],
                color=BROWN, linewidth=0.75)
        ax.text((previous + following) / 2, 1.29 * ymax,
                f"{length} zero days", ha="center", va="bottom", fontsize=11.2,
                color=BROWN)
    fig.text(0.024, 0.685, "Daily observed visits", rotation=90,
             ha="center", va="center", fontsize=12.5)

    left = fig.add_axes([0.090, 0.092, 0.390, 0.250])
    right = fig.add_axes([0.605, 0.092, 0.370, 0.250])
    fig.text(0.090, 0.375, "(b)  Sparsity across POIs", fontsize=13.5, weight="bold")
    fig.text(0.605, 0.375, "(c)  Daily visit counts", fontsize=13.5, weight="bold")
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.tick_params(length=3, width=0.65, pad=3)

    ratios = (visits > 0).sum(axis=0) / np.isfinite(visits).sum(axis=0)
    values, frequencies = np.unique(100 * ratios, return_counts=True)
    cumulative = 100 * np.cumsum(frequencies) / len(ratios)
    left.step(np.r_[0, values, 100], np.r_[0, cumulative, 100], where="post",
              color=BLUE, linewidth=1.55, zorder=4)
    left.set(xlim=(0, 100), ylim=(0, 103),
             xlabel="Nonzero ratio (%)", ylabel="Cumulative POIs (%)")
    left.set_xticks([0, 20, 40, 60, 80, 100])
    left.set_yticks([0, 25, 50, 75, 100])
    q25, q50 = np.percentile(100 * ratios, [25, 50])
    for quantile, level, color in [(q25, 25, BROWN), (q50, 50, GRAY)]:
        left.plot([0, quantile, quantile], [level, level, 0],
                  color=color, linewidth=0.75, linestyle=(0, (3, 2.5)), zorder=2)
        left.plot(quantile, level, "o", markersize=3.7, markeredgewidth=0.9,
                  markeredgecolor=color, markerfacecolor="white", zorder=5)
    for label, quantile, level, xytext, color in [
        ("25th percentile", q25, 25, (21, 67), BROWN),
        ("Median", q50, 50, (81, 22), INK),
    ]:
        left.annotate(f"{label}\n{quantile:.1f}%", xy=(quantile, level), xytext=xytext,
                      ha="center", va="center", fontsize=11.2, color=color,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                      arrowprops={"arrowstyle": "-", "lw": 0.6, "color": color,
                                  "shrinkA": 4, "shrinkB": 4})

    share = [row[2] for row in counts]
    right.bar(np.arange(len(counts)), share, width=0.72,
              color=[BROWN] + [BLUE] * (len(counts) - 1), zorder=3)
    right.set_xticks(np.arange(len(counts)), [row[0] for row in counts])
    right.set(xlim=(-0.6, len(counts) - 0.4), ylim=(0, 46),
              xlabel="Daily observed visits", ylabel="Observed POI-days (%)")
    right.set_yticks([0, 10, 20, 30, 40])
    for i, value in enumerate(share):
        right.text(i, value + 1, f"{value:.1f}", ha="center", fontsize=10.8,
                   color=BROWN if i == 0 else INK, weight="bold" if i == 0 else "normal")
    right.text(0.98, 0.93, f"0 or 1 visit: {stats['at_most_one_percent']:.1f}%",
               transform=right.transAxes, ha="right", va="top", fontsize=11.2,
               bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        b = artist.get_window_extent(renderer)
        if b.width > 0 and b.height > 0:
            assert b.x0 >= box.x0 - 1 and b.x1 <= box.x1 + 1, artist.get_text()
            assert b.y0 >= box.y0 - 1 and b.y1 <= box.y1 + 1, artist.get_text()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT/'exp0')
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--preview", type=Path)
    parser.add_argument('--prepared', type=Path, default=DATA_ROOT/'prepared')
    args = parser.parse_args()
    source = args.prepared / "approved_period_visits_with_missing.npy"
    visits = np.load(source, allow_pickle=False)
    with (source.parent / "approved_dates.csv").open(encoding="utf-8-sig", newline="") as stream:
        dates = [row["date"] for row in csv.DictReader(stream)]
    with (source.parent / "identified_9884_pois.csv").open(encoding="utf-8-sig", newline="") as stream:
        keys = [row["placekey"] for row in csv.DictReader(stream)]
    assert visits.shape == (len(dates), len(keys)) == (92, 9884)
    assert dates[0] == "2020-10-01" and dates[-1] == "2020-12-31"
    date_values = np.array(dates, dtype="datetime64[D]")
    assert np.all(np.diff(date_values) == np.timedelta64(1, "D"))
    observed = np.isfinite(visits)
    x = visits[observed]
    assert np.all(x >= 0) and np.all(x == np.floor(x))
    ratios = (visits > 0).sum(axis=0) / observed.sum(axis=0)
    gaps = [completed_gaps(visits[:, j]) for j in range(visits.shape[1])]
    longest = np.array([max((g[0] for g in item), default=0) for item in gaps])
    lengths = np.array([g[0] for item in gaps for g in item], dtype=int)
    examples = choose_examples(visits, ratios, longest, keys)
    bins = [("0", 0, 1), ("1", 1, 2), ("2", 2, 3), ("3", 3, 4),
            ("4-5", 4, 6), ("6-10", 6, 11), ("11+", 11, np.inf)]
    counts = []
    for label, lower, upper in bins:
        number = int(np.count_nonzero((x >= lower) & (x < upper)))
        counts.append((label, number, 100 * number / len(x)))
    assert sum(row[1] for row in counts) == len(x)
    stats = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "period_start": dates[0], "period_end": dates[-1],
        "pois": visits.shape[1], "days": visits.shape[0],
        "observed_poi_days": len(x), "missing_poi_days": int((~observed).sum()),
        "zero_poi_days": int(np.count_nonzero(x == 0)),
        "zero_percent": float(100 * np.mean(x == 0)),
        "at_most_one_percent": float(100 * np.mean(x <= 1)),
        "poi_nonzero_ratio_percentiles": {
            str(p): float(np.percentile(100 * ratios, p)) for p in [10, 25, 50, 75]
        },
        "completed_zero_gaps": len(lengths),
        "pois_with_completed_zero_gap": int(np.count_nonzero(longest)),
        "gap_definition": "Consecutive observed zero days between two positive-visit days; both endpoints and all intervening days must be observed.",
        "boundary_policy": "Leading and trailing zero runs, and runs adjacent to missing values, are excluded from completed zero gaps.",
        "prevalence_denominator": "All 9,884 POIs; each POI counts once per threshold if any completed zero gap reaches that threshold.",
        "period_scope": "Full October-December descriptive period, not the training period used to define low-history POIs.",
        "example_selection": "Choose complete POIs closest to the 10th, 25th and 50th percentiles of full-period nonzero ratios; within each tied group, choose the POI closest to the group's median longest completed zero gap; break remaining ties by placekey.",
        "highlight_selection": "Longest completed zero gap in each selected POI; earliest gap breaks a tie.",
        "examples": examples,
        "prevalence": {
            str(k): {"pois": int(np.sum(longest >= k)), "percent": float(100 * np.mean(longest >= k))}
            for k in [1, 3, 7, 14, 21, 28]
        },
        "completed_gap_length_percentiles": {
            str(p): float(np.percentile(lengths, p)) for p in [25, 50, 75, 90, 95, 99, 100]
        },
        "figure_width_inches": FIGURE_WIDTH, "figure_height_inches": FIGURE_HEIGHT,
        "raster_dpi": args.dpi,
    }
    assert stats["source_sha256"] == "bbcc991007223555a333c6fc123cd6b839fdcf7ff8b50bf0fcf7eb25c456bdb0"
    args.out.mkdir(parents=True, exist_ok=True)
    fig = make_figure(visits, dates, examples, gaps, longest, counts, stats)
    for extension in ["pdf", "svg", "png"]:
        options = {"dpi": args.dpi} if extension == "png" else {}
        fig.savefig(args.out / f"{STEM}.{extension}", **options)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=160)
    plt.close(fig)
    (args.out / f"{STEM}_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out / f"{STEM}_sequences.csv",
              ["example", "placekey", "date", "observed_visits", "highlighted_zero_day", "highlighted_following_positive"],
              ((e["label"], e["placekey"], d, int(visits[t, e["column_index"]]),
                int(e["highlighted_zero_start"] <= d <= e["highlighted_zero_end"]),
                int(d == e["following_positive_date"])) for e in examples for t, d in enumerate(dates)))
    write_csv(args.out / f"{STEM}_zero_gaps.csv",
              ["zero_days", "completed_gaps_exact_length", "gap_share_percent", "pois_with_gap_at_least_length", "poi_prevalence_percent"],
              ((k, int(np.sum(lengths == k)), float(100 * np.mean(lengths == k)),
                int(np.sum(longest >= k)), float(100 * np.mean(longest >= k)))
               for k in range(1, int(lengths.max()) + 1)))
    write_csv(args.out / f"{STEM}_daily_counts.csv",
              ["daily_visits", "observed_poi_days", "share_percent"], counts)
    unique_ratios, frequencies = np.unique(100 * ratios, return_counts=True)
    write_csv(args.out / f"{STEM}_nonzero_cdf.csv",
              ["nonzero_ratio_percent", "pois_at_ratio", "cumulative_pois", "cumulative_percent"],
              zip(unique_ratios, frequencies, np.cumsum(frequencies),
                  100 * np.cumsum(frequencies) / len(ratios)))
    print(json.dumps({"zero_percent": stats["zero_percent"], "completed_zero_gaps": len(lengths),
                      "prevalence": stats["prevalence"], "examples": examples}, indent=2))
    print("Saved temporal sparsity figure, source-data tables and statistics to", args.out)


if __name__ == "__main__":
    main()
