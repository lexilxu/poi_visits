"""Publication figures rendered directly from the verified experiment tables."""
import argparse
import json
import sys
from pathlib import Path

from exp.common.paths import DATA_ROOT, RESULTS_ROOT
DATA = DATA_ROOT/'figure_data'
ART = RESULTS_ROOT/'figures'
SUPPLEMENTARY = ART/'supplementary'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm, TwoSlopeNorm
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

WIDTH = 183 / 25.4
BLUE = "#285A80"
RUST = "#B45D3B"
RUST_INK = "#96492F"
TEAL = "#21847D"
PLUM = "#745483"
INK = "#222A31"
GRAY = "#74808A"
RULE = "#A2ABB2"
PALE_BLUE = "#EAF2F7"
PALE_TEAL = "#E7F3F0"
FILES = {
    "01": "recovery_activation_tradeoff.pdf",
    "02": "outcome_conditioned_graph_transfer.pdf",
    "03": "neighbor_perturbation_tradeoff.pdf",
    "04": "attribution_fingerprint.pdf",
    "05": "exp_logic.pdf",
    "06": "poi_sparsity_distribution.pdf",
    "07": "forecasting_performance_across_horizons.pdf",
    "02a": "graph_transfer_archived_P14_H7.pdf",
    "04a": "attribution_archived_gradient_fingerprint.pdf",
}


def style():
    font = "Arial"
    try:
        findfont(FontProperties(family=font), fallback_to_default=False)
    except ValueError:
        font = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": font, "font.size": 9, "font.weight": "normal",
        "axes.labelsize": 10.4, "axes.labelweight": "bold",
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.unicode_minus": True, "axes.grid": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "figure.dpi": 144,
        "mathtext.fontset": "custom", "mathtext.rm": font,
        "mathtext.it": font + ":italic", "mathtext.bf": font + ":bold",
    })


def clean(ax):
    ax.tick_params(axis="both", length=3.1, width=.6, pad=4)
    ax.set_axisbelow(True)


def panel(fig, ax, letter, title, y=.930):
    pos = ax.get_position()
    fig.text(pos.x0-.041, y, letter, fontsize=13.2, weight="bold", va="center")
    fig.text(pos.x0, y, title, fontsize=10.6, weight="bold", va="center")


def label(ax, xy, at, text, ha="left", size=8.6, color=INK, leader=RULE, marker_pad=3.1):
    return ax.annotate(text, xy, xytext=at, ha=ha, va="center", fontsize=size, color=color,
                       annotation_clip=False, zorder=8,
                       arrowprops={"arrowstyle": "-", "color": leader, "lw": .60,
                                   "shrinkA": 2.7, "shrinkB": marker_pad})


def small_legend(fig, handles, left, top):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, top),
               ncols=len(handles), frameon=False, borderaxespad=0,
               handlelength=1.1, handletextpad=.35, columnspacing=1.45)


def validate_text(fig, key):
    # Measure actual rendered letters, excluding annotation leader lines.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    records = []
    for artist in fig.findobj(Text):
        txt = artist.get_text().strip()
        if txt and artist.get_visible():
            b = Text.get_window_extent(artist, renderer)
            if np.isfinite(b.extents).all() and b.width > 0 and b.height > 0:
                records.append((txt, b))
    outside = [t for t, b in records
               if min(b.x0, b.y0) < -1 or b.x1 > width+1 or b.y1 > height+1]
    overlaps = []
    for i, (a, ba) in enumerate(records):
        for b, bb in records[i+1:]:
            s = Bbox.intersection(ba, bb)
            if s is not None and min(s.width, s.height) > 2.5:
                overlaps.append((a, b))
    if outside or overlaps:
        raise RuntimeError(f"{key} layout: outside={outside}; overlaps={overlaps}")


def output(fig, key, title):
    validate_text(fig, key)
    dest = (SUPPLEMENTARY if key.endswith("a") else ART) / FILES[key]
    fig.savefig(dest, dpi=2000, metadata={"Title": title, "Subject": "POI demand forecasting"})
    plt.close(fig)
    print(f"{key}: {dest.name}", flush=True)


def recovery():
    data = pd.read_csv(DATA / "01_recovery_activation.csv", keep_default_na=False)
    fig = plt.figure(figsize=(WIDTH, 3.92))
    ax = fig.add_axes([.106, .169, .861, .724])
    styles = {
        "EvolveGCN": dict(marker="o", ms=4.2, mec=BLUE, mfc=BLUE, mew=.60),
        "MTGNN": dict(marker="o", ms=2.05, mec=RUST, mfc="white", mew=.62),
        "Benchmark": dict(marker="D", ms=4.1, mec=INK, mfc=INK, mew=.55),
    }
    layout = {
        ("AR-ridge", "None"): ("AR-ridge", (12.0, 84.8), "left"),
        ("Historical average", "None"): ("Historical average", (2.0, 10.8), "left"),
        ("LSTM", "None"): ("LSTM", (3.0, 20.3), "left"),
        ("EvolveGCN", "Identity"): ("Identity", (18.2, 56.3), "left"),
        ("EvolveGCN", "Distance KNN"): ("Distance KNN", (26.2, 67.0), "right"),
        ("EvolveGCN", "Distance RBF"): ("Distance RBF", (26.6, 47.5), "right"),
        ("EvolveGCN", "NAICS-2"): ("NAICS-2", (30.7, 48.4), "left"),
        ("EvolveGCN", "Visit correlation"): ("Visit correlation", (34.0, 57.0), "left"),
        ("EvolveGCN", "Top category"): ("Top category", (36.0, 73.5), "left"),
        ("MTGNN", "Distance KNN"): ("Distance KNN", (16.1, 16.5), "left"),
        ("MTGNN", "Top category"): ("Top category", (32.6, 32.2), "left"),
        ("MTGNN", "Identity"): ("Identity", (32.6, 24.9), "left"),
        ("MTGNN", "NAICS-2"): ("NAICS-2", (32.6, 17.8), "left"),
    }
    assert len(data) == len(layout)
    for row in data.itertuples():
        xy = (100*row.far_0_5, 100*row.dmr_model)
        s = styles.get(row.model, styles["Benchmark"])
        ax.plot(*xy, ls="none", zorder=5, **s)
        name, at, ha = layout[(row.model, row.adjacency)]
        color = {"EvolveGCN": BLUE, "MTGNN": RUST_INK}.get(row.model, INK)
        leader = {"EvolveGCN": "#91A9BA", "MTGNN": "#C1998A"}.get(row.model, RULE)
        label(ax, xy, at, name, ha, color=color, leader=leader, marker_pad=s["ms"]/2+.65)
    ax.set(xlim=(-1.5,44), ylim=(6,92), xlabel="False activation rate (%)",
           ylabel="Demand mass recovery ratio (%)")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([10,30,50,70,90])
    clean(ax)
    keys = []
    for model in ("EvolveGCN", "MTGNN"):
        keys.append(Line2D([], [], ls="none", label=model, **(styles[model] | {"ms": 5.0})))
    small_legend(fig, keys, .108, .972)
    output(fig, "01", "Recovery-activation tradeoff")


def formatted_edges(values):
    for decimals in range(2, 6):
        out = [f"{v:.{decimals}f}".rstrip("0").rstrip(".") for v in values]
        if len(set(out)) == len(values):
            return out
    raise ValueError("Evidence bin edges cannot be distinguished.")


def heatmap_tiles(ax, values, scale, fmt, fontsize, dominant=None, inset=.04):
    # Separate tiles using the inset geometry of Andersson et al., Fig. 3C
    # (https://github.com/almaan/her2st/blob/master/scripts/corrplot.R#L295-L325).
    faces = scale.to_rgba(np.ma.masked_invalid(values))
    for (row, col), value in np.ndenumerate(values):
        face = faces[row, col]
        ax.add_patch(Rectangle((col+inset, row+inset), 1-2*inset, 1-2*inset,
                               facecolor=face, edgecolor="none"))
        rgb = face[:3]
        linear_rgb = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        luminance = linear_rgb @ np.array([.2126, .7152, .0722])
        bold = dominant is not None and col == dominant[row]
        ax.text(col+.5, row+.5, fmt(value) if np.isfinite(value) else "—",
                ha="center", va="center", fontsize=fontsize,
                color="white" if luminance < .179 else "black",
                weight="bold" if bold else "normal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def signed_gain(value):
    return "0.00" if abs(value) < .005 else f"{value:+.2f}".replace("-", "−")


def outcome_heatmaps(archived=False):
    source = "02_archived_graph_transfer_cells.csv" if archived else "02_graph_transfer_cells.csv"
    frame = pd.read_csv(DATA / source)
    fig = plt.figure(figsize=(WIDTH, 3.73 if archived else 3.54))
    axes = [fig.add_axes([x, .168, .349, .683 if archived else .711])
            for x in (.117, .505)]
    max_gain = frame.loc[frame.display_cell, "mean_mae_gain"].abs().max()
    limit = max(.05, np.ceil(max_gain*20)/20)
    # Crameri, Scientific Colour Maps v8. Table and MIT license from
    # cmcrameri at 87197e141a2c451691ec53fadc6da4d017f79a4c.
    vik = np.loadtxt(HERE / "scientific_colormaps" / "vik.txt")
    cmap = ListedColormap(vik, name="vik").reversed().with_extremes(bad="#E9EDF0")
    scale = ScalarMappable(norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                           cmap=cmap)
    if archived:
        fig.text(.5, .985, "Archived setting (P = 14, H = 7 days)",
                 ha="center", va="top", fontsize=8.5)
    for i, (ax, name) in enumerate(zip(axes, ("Positive demand", "Zero demand"))):
        cells = frame.loc[frame.outcome.eq(name)].sort_values(["y_bin", "x_bin"])
        nx, ny = cells.x_bin.nunique(), cells.y_bin.nunique()
        mat = cells.mean_mae_gain.where(cells.display_cell).to_numpy().reshape(ny, nx)
        heatmap_tiles(ax, mat, scale, signed_gain, 8.3)
        xbins = cells.drop_duplicates("x_bin")
        ybins = cells.drop_duplicates("y_bin")
        edges_x = [*xbins.target_low, xbins.target_high.iloc[-1]]
        edges_y = [*ybins.neighbor_low, ybins.neighbor_high.iloc[-1]]
        ax.set_xticks(np.arange(nx+1), formatted_edges(edges_x))
        ax.set_yticks([] if i else np.arange(ny+1), [] if i else formatted_edges(edges_y))
        ax.set(xlim=(0, nx), ylim=(0, ny))
        ax.tick_params(length=2.4, width=.55, pad=3.5, labelsize=8.0)
        supported = cells.loc[cells.n_events.gt(0)]
        gain = np.average(supported.mean_mae_gain, weights=supported.n_events)
        val = f"{gain:+.3f}".replace("-", "−")
        pos = ax.get_position()
        top = .915 if archived else .937
        fig.text(pos.x0, top, "ab"[i], va="center", fontsize=13.2, weight="bold")
        fig.text(pos.x0+.038, top, name, va="center", fontsize=10.8, weight="bold")
        fig.text(pos.x1, top, f"Mean {val}", ha="right", va="center", fontsize=8.6,
                 color=BLUE if gain >= 0 else RUST_INK)
    axes[0].set_ylabel("Neighbor evidence (visits/day)", labelpad=7)
    fig.text(.486, .059, "Target evidence (visits/day)", ha="center", va="center",
             fontsize=10.4, weight="bold")
    cax = fig.add_axes([.891, .184, .017, .679 if not archived else .651])
    cb = fig.colorbar(scale, cax=cax)
    ticks = np.linspace(-limit, limit, 7)
    cb.set_ticks(ticks, labels=[f"{v:+.2f}".replace("-", "−") if abs(v) > 1e-9 else "0"
                               for v in ticks])
    cb.set_label("MAE gain (visits)", fontsize=9.0, labelpad=4.8)
    cb.ax.tick_params(length=2.6, pad=3, width=.55, labelsize=8.1)
    cb.outline.set_linewidth(.45)
    cb.solids.set_rasterized(False)
    output(fig, "02a" if archived else "02", "Outcome-conditioned graph transfer")


def perturbation():
    data = pd.read_csv(DATA / "03_neighbor_perturbation.csv")
    y = np.arange(len(data))
    a,b = 100*data.false_activations_removed,100*data.recoveries_lost
    fig = plt.figure(figsize=(WIDTH,3.36))
    ax = fig.add_axes([.241,.181,.710,.694])
    for j,x1,x2 in zip(y,a,b):
        ax.plot([x1,x2],[j-.10,j+.10],color="#9AA7B1",lw=.85,zorder=1)
    l1, = ax.plot(a,y-.10,ls="none",marker="o",ms=4.8,color=BLUE,
                   label="False activations removed",zorder=3)
    l2, = ax.plot(b,y+.10,ls="none",marker="D",ms=4.7,mfc="white",mec=RUST,mew=1,
                   label="Recoveries lost",zorder=4)
    ax.set_yticks(y,[name.removesuffix(" neighbors") for name in data.neighbor_group])
    endpoint = max(10, np.ceil(max(a.max(), b.max()) / 10)*10)
    step = 20 if endpoint >= 80 else 10
    ax.set(xlim=(-.016*endpoint, 1.04*endpoint),ylim=(len(y)-.52,-.52),
           xlabel="Share of reference events (%)")
    ax.set_xticks(np.arange(0,endpoint+1,step))
    ax.axvline(0,lw=.55,color="#ABB5BD",zorder=0)
    clean(ax)
    ax.tick_params(axis="y",length=0,pad=7,labelsize=9)
    ax.spines["left"].set_visible(False)
    fig.text(.048,.928,"Neighbor group",fontsize=9.2,weight="bold",va="center")
    small_legend(fig,[l1,l2],.244,.955)
    output(fig,"03","Neighbor perturbation tradeoff")


def fingerprint(archived=False):
    source = "04_archived_gradient_fingerprint.csv" if archived else "04_attribution_fingerprint.csv"
    data = pd.read_csv(DATA / source, index_col=0)
    mat = 100*data.to_numpy()
    regime_sources = {f"{column}-driven": i for i, column in enumerate(data.columns)}
    named_regimes = not archived and set(data.index) == regime_sources.keys()
    fig = plt.figure(figsize=(WIDTH, 3.24))
    if named_regimes:
        assignments = pd.read_csv(DATA / "attribution_cluster_assignments.csv")
        score_columns = [f"{column}_z" for column in data.columns]
        mat = assignments.groupby("cluster")[score_columns].mean().reindex(data.index).to_numpy()
        ax = fig.add_axes([.208, .240, .510, .647])
        extent = max(.5, np.ceil(2*np.nanmax(np.abs(mat)))/2)
        ticks = np.linspace(-extent, extent, 5)
        cmap = LinearSegmentedColormap.from_list("attribution_z", ["#255D87", "#F7F7F4", "#B95321"])
        norm = TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent)
    elif archived:
        ax = fig.add_axes([.259, .259, .697, .578])
        cmap = matplotlib.colormaps["magma_r"]
        ticks = [0, 5, 20, 50, 100]
    else:
        ax = fig.add_axes([.132, .240, .586, .647])
        cmap = LinearSegmentedColormap.from_list("attribution_blue", [
            "#F6FAFC", "#E3EFF4", "#C9DEE8", "#9DC2D4",
            "#6898B5", "#487CA1", "#2B5579", "#17354F",
        ])
        ticks = [0, 5, 15, 30, 50, 80]
    if not named_regimes:
        norm = PowerNorm(gamma=.5, vmin=0, vmax=ticks[-1])
    scale = ScalarMappable(norm=norm, cmap=cmap.with_extremes(bad="#E9EDF0"))
    fontsize = 9.6 if archived else 12.2
    if named_regimes:
        defining_columns = np.array([regime_sources[name] for name in data.index])
        available = np.isfinite(mat[np.arange(len(data)), defining_columns])
        bold_columns = np.where(available, defining_columns, -1)
    else:
        bold_columns = mat.argmax(axis=1)
    heatmap_tiles(ax, mat, scale, lambda value: f"{value:.2f}", fontsize,
                  dominant=bold_columns, inset=.04 if archived else .014)
    display = data.index if archived or named_regimes else [f"Cluster {j+1}" for j in range(len(data))]
    centers = np.arange(len(data))+.5
    ax.set_yticks(centers, display)
    ax.set_xticks(np.arange(len(data.columns))+.5,
                  [x.replace("Hours and missingness", "Hours and\nmissingness") for x in data.columns])
    ax.set(xlim=(0, len(data.columns)), ylim=(len(data), 0))
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=6.0, labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_weight("bold")
    if not archived:
        clusters = pd.read_csv(DATA / "04_attribution_clusters.csv").set_index("display_label")
        sizes = clusters.loc[data.index, "n_pois"]
        bx = fig.add_axes([.756, .240, .220, .647])
        bx.axvline(0, color="#CBD6DE", lw=.55, clip_on=False)
        bx.barh(centers, sizes, height=.36, color="#7796AC", lw=0)
        for y, n in zip(centers, sizes):
            bx.text(1, y, f"{n:,}", transform=bx.get_yaxis_transform(), ha="right",
                    va="center", fontsize=11.7)
        bx.set(xlim=(0, max(sizes)*1.56), ylim=(len(data), 0))
        bx.set_axis_off()
        bx.set_title("Regime size (POIs)" if named_regimes else "Cluster size (POIs)",
                     loc="left", fontsize=12.2, weight="bold", pad=8.5)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, .146 if archived else .153, pos.width, .038 if archived else .036])
    cb = fig.colorbar(scale, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=8.1 if archived else 10.2, length=2.4, width=.55, pad=2.5)
    cb.outline.set_linewidth(.45)
    cb.solids.set_rasterized(False)
    if not archived:
        # Overlap adjacent vector swatches to prevent hairline gaps in PDF viewers.
        cb.solids.set_edgecolor("face")
        cb.solids.set_linewidth(.3)
    color_title = "Mean standardized attribution" if named_regimes else (
        "Mean normalized score (%)" if archived else "Mean attribution share (%)")
    fig.text(pos.x0+.5*pos.width, .035, color_title,
             fontsize=9.3 if archived else 11.0, ha="center", va="bottom")
    output(fig, "04a" if archived else "04", "Attribution fingerprint")


def framework():
    fig = plt.figure(figsize=(WIDTH,2.27))
    ax = fig.add_axes([.014,.075,.972,.850])
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")

    def rect(x, y, width, height, face):
        ax.add_patch(Rectangle((x,y),width,height,facecolor=face,edgecolor="none"))

    def arrow(left, right, y, color):
        ax.add_patch(FancyArrowPatch((left,y),(right,y),arrowstyle="-|>",mutation_scale=8.5,
                                    linewidth=.8,color=color,shrinkA=0,shrinkB=0))

    rect(.013,.301,.171,.398,"#F1F3F4")
    ax.text(.0985,.561,"SafeGraph\nSeattle",ha="center",va="center",weight="bold",
            fontsize=11.0,linespacing=1.08)
    ax.text(.0985,.380,"Visits and attributes",ha="center",va="center",fontsize=8.1)
    ax.plot([.184,.215],[.500,.500],color=GRAY,lw=.75)
    ax.plot([.215,.215],[.250,.750],color=GRAY,lw=.75)

    for mid, color, face, model, detail in [
        (.750,BLUE,PALE_BLUE,"GNN-adj","Visit history\nAlternative adjacency"),
        (.250,TEAL,PALE_TEAL,"GNN-static","Visits and attributes\nIdentity adjacency"),
    ]:
        rect(.250,mid-.196,.737,.392,face)
        rect(.250,mid-.196,.005,.392,color)
        arrow(.215,.250,mid,color)
        ax.text(.279,mid+.076,model,va="center",color=color,weight="bold",fontsize=11.3)
        ax.text(.279,mid-.065,detail,va="center",fontsize=8.6,linespacing=1.12)

    ax.plot([.503,.539],[.750,.750],color=BLUE,lw=.8)
    ax.plot([.539,.539],[.666,.834],color=BLUE,lw=.8)
    for y, name in [(.834,"Graph-transfer analysis"),(.666,"Neighbor perturbation")]:
        arrow(.539,.568,y,BLUE)
        ax.text(.580,y,name,va="center",weight="bold",fontsize=10.0)

    arrow(.503,.568,.250,TEAL)
    ax.text(.580,.250,"Gradient attribution",va="center",weight="bold",fontsize=10.0)
    arrow(.780,.817,.250,TEAL)
    ax.text(.831,.250,"Attribution\nregimes",va="center",weight="bold",fontsize=10.0)
    output(fig,"05","Overall forecasting and interpretation framework")


def sparsity(meta):
    visits = pd.read_csv(DATA/"06_daily_count_distribution.csv")
    poi = pd.read_csv(DATA/"06_poi_nonzero_ratio_distribution.csv")
    fig = plt.figure(figsize=(WIDTH,3.12))
    ax = fig.add_axes([.105,.182,.411,.662])
    bx = fig.add_axes([.626,.182,.337,.662])
    x = np.arange(len(visits))
    ax.bar(x,visits.share_percent,color=[RUST]+[BLUE]*(len(x)-1),width=.77,
           edgecolor="white",linewidth=.45)
    ax.set_xticks(x,visits.count_bin,fontsize=7.75)
    ax.set(xlim=(-.64,len(x)-.37),ylim=(0,45),xlabel="Daily observed visits",
           ylabel="Observed POI-days (%)")
    ax.set_yticks([0,10,20,30,40])
    ax.text(0,visits.share_percent.iloc[0]+1.0,f"{meta['zero_share_percent']:.1f}%",
            ha="center",va="bottom",weight="bold",fontsize=9.1)
    bx.bar(poi.lower_percent,poi.share_percent,width=10,align="edge",
           color=BLUE,edgecolor="white",linewidth=.65)
    med = meta["nonzero_ratio_median_percent"]
    bx.set(xlim=(0,100),ylim=(0,28),xlabel="Nonzero ratio (%)",ylabel="POIs (%)")
    bx.set_xticks([0,20,40,60,80,100])
    bx.set_yticks([0,5,10,15,20,25])
    bx.axvline(med,color=RUST,linestyle=(0,(4,2.4)),lw=.95,zorder=4)
    bx.text(med-3.4,26.1,f"Median {med:.1f}%",fontsize=8.7,ha="right",va="top")
    for a in (ax,bx):
        clean(a)
    panel(fig,ax,"a","Daily visit counts")
    panel(fig,bx,"b","POI nonzero ratios")
    output(fig,"06","POI sparsity distribution")


def horizons():
    rows = pd.read_csv(DATA/"07_horizon_skill.csv")
    fig = plt.figure(figsize=(WIDTH,4.07))
    ax = fig.add_axes([.122,.412,.600,.540])
    bx = fig.add_axes([.122,.175,.600,.189],sharex=ax)
    styles = {
        "Historical average": (GRAY,None,(0,(2,2)),True,"Historical average",2.5),
        "EvolveGCN (identity)": (BLUE,"o","-",True,"EvolveGCN\n(identity)",-3.8),
        "EvolveGCN (best non-identity)": (BLUE,"s",(0,(4,2)),False,"EvolveGCN\n(best non-identity)",-15.9),
        "MTGNN (best)": (TEAL,"^",(0,(5,2,1.2,2)),True,"MTGNN (best)",-9.7),
        "LSTM": (PLUM,"D","-",False,"LSTM",-55.1),
        "AR-ridge": (RUST,"v","-",True,"AR-ridge",-94.2),
    }
    for series,(c,m,ls,filled,name,y) in styles.items():
        d = rows.loc[rows.series.eq(series)].sort_values("H")
        assert d.H.tolist()==[7,14,28]
        host = bx if series in {"LSTM","AR-ridge"} else ax
        host.plot(d.H,d.skill_percent,color=c,ls=ls,marker=m,ms=4.45,mew=.85,
                  lw=1.28,mfc=c if filled else "white",zorder=4)
        label(host,(28,d.skill_percent.iloc[-1]),(30.4,y),name,size=8.7)
    ax.set(xlim=(5.5,29.4),ylim=(-18,15))
    ax.set_yticks([-15,-10,-5,0,5,10,15])
    bx.set_ylim(-178,-27)
    bx.set_yticks([-160,-120,-80,-40])
    bx.set_xticks([7,14,28])
    bx.set_xlabel(r"Forecast horizon $H$ (days)")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x",bottom=False,labelbottom=False)
    for axis in (ax,bx):
        clean(axis)
    for axis,y in ((ax,0),(bx,1)):
        axis.plot([-.012,.012],[y-.018,y+.018],transform=axis.transAxes,
                  color=INK,lw=.75,clip_on=False)
    fig.text(.033,.569,"MAE skill relative to historical average (%)",rotation=90,
             va="center",ha="center",fontsize=10.4,weight="bold")
    output(fig,"07","Forecasting performance across horizons")


def main():
    global DATA, ART, SUPPLEMENTARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",nargs="+",choices=list(FILES))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, default=ART)
    args = parser.parse_args()
    DATA, ART = args.data, args.out
    SUPPLEMENTARY = ART/'supplementary'
    ART.mkdir(parents=True,exist_ok=True)
    SUPPLEMENTARY.mkdir(parents=True,exist_ok=True)
    style()
    meta = json.loads((DATA/"provenance.json").read_text(encoding="utf-8"))
    plot = {"01":recovery,"02":outcome_heatmaps,"03":perturbation,"04":fingerprint,
            "05":framework,"06":lambda:sparsity(meta["06"]),"07":horizons,
            "02a":lambda:outcome_heatmaps(True),"04a":lambda:fingerprint(True)}
    for key in args.only or [f"{i:02}" for i in range(1,8)]:
        if key not in meta:
            raise KeyError(f"Missing provenance for {key}")
        plot[key]()


if __name__ == "__main__":
    main()
