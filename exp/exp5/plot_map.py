"""Map formal attribution groups and report their local concentration."""
import argparse
import io
from math import fsum, lgamma
from pathlib import Path
import zipfile

from exp.common.compact_figures import WIDTH, INK, style, validate_text
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.text import Text
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from matplotlib.path import Path as DrawPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile


BOUNDS = (-122.520, -122.185, 47.490, 47.764)
GRID_ORIGIN = np.array([-122.470, 47.490])
KM_PER_DEGREE = np.array([111.32*np.cos(np.radians((BOUNDS[2]+BOUNDS[3])/2)), 111.32])
CELL_KM = 2
MIN_CELL_POIS = 60
NEIGHBOR_COUNTS = (5, 10, 20, 40)
REGIME_NAMES = ("History-driven", "Location-driven", "Identity-driven",
                "Parent-driven", "Others-driven")
CLUSTER_COLORS = np.array(["#F22321", "#0769C1", "#099941", "#8C40AD", "#F79416"])
TITLE_COLORS = ("#FBD6D4", "#D6E3F3", "#DCEFDB", "#EBE0F2", "#FEE8D1")
LAND, WATER = "#EFEFEF", "#C8D9E7"
CONTEXT_COLOR = "#A8B0BA"
REFERENCE_SIZE = (16.2, 7.2)
OVERVIEW_SOUTH = 47.474
from exp.common.paths import DATA_ROOT
STREET_FILE = DATA_ROOT / "cartography/seattle_streets_tiger2020.npz"


def viewport(ax, south=BOUNDS[2]):
    west, east, _, north = BOUNDS
    center = (west+east)/2
    width, height = ax.figure.get_size_inches()
    box = ax.get_position()
    aspect = (width*box.width)/(height*box.height)
    half_width = (north-south)*KM_PER_DEGREE[1]/KM_PER_DEGREE[0]*aspect/2
    return center-half_width, center+half_width, south, north


def water_paths(shapefile_zip, bounds):
    xlo, xhi, ylo, yhi = bounds
    with zipfile.ZipFile(shapefile_zip) as archive:
        shape_name, = [name for name in archive.namelist() if name.endswith(".shp")]
        with shapefile.Reader(shp=io.BytesIO(archive.read(shape_name))) as reader:
            paths = []
            for shape in reader.iterShapes():
                left, bottom, right, top = shape.bbox
                if right < xlo or left > xhi or top < ylo or bottom > yhi:
                    continue
                points = np.asarray(shape.points)
                codes = np.full(len(points), DrawPath.LINETO, dtype=np.uint8)
                starts = np.asarray(shape.parts)
                codes[starts] = DrawPath.MOVETO
                codes[np.r_[starts[1:], len(points)] - 1] = DrawPath.CLOSEPOLY
                paths.append(DrawPath(points, codes))
    return paths


def street_paths(street_file):
    # The compact geometry contains only local and major TIGER/Line 2020 roads
    # in the Seattle view, with the county archive URLs and hashes alongside.
    with np.load(street_file) as layer:
        return [DrawPath(layer[f"{kind}_vertices"], layer[f"{kind}_codes"])
                for kind in ("local", "major")]


def log_choose(total, selected):
    return lgamma(total+1) - lgamma(selected+1) - lgamma(total-selected+1)


def grid_enrichment(xy, groups, sizes):
    """Compare 2 km cells to random labels with the observed cluster sizes."""
    cells, index = np.unique(np.floor(xy/CELL_KM).astype(int), axis=0, return_inverse=True)
    counts = np.zeros((len(cells), len(sizes)), dtype=int)
    np.add.at(counts, (index, groups), 1)
    eligible = counts.sum(axis=1) >= MIN_CELL_POIS
    cells, counts = cells[eligible], counts[eligible]
    assert len(cells)
    total = len(groups)
    totals = counts.sum(axis=1)
    p_values = np.empty(counts.shape)
    for i, sample in enumerate(totals):
        denominator = log_choose(total, sample)
        for k, size in enumerate(sizes):
            lower = max(counts[i, k], sample-total+size)
            upper = min(size, sample)
            p_values[i, k] = min(1., fsum(
                np.exp(log_choose(size, j) + log_choose(total-size, sample-j) - denominator)
                for j in range(lower, upper+1)
            ))

    flat = p_values.ravel()
    order = np.argsort(flat)
    adjusted = np.empty_like(flat)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(flat[order]*(len(flat)-np.arange(len(flat)))))
    strongest = p_values.argmin(axis=1)
    rows = np.arange(len(cells))
    significant = adjusted.reshape(counts.shape)[rows, strongest] < .05
    excess = 100*(counts[rows, strongest]/totals - sizes[strongest]/total)
    return cells, totals, strongest, significant, excess


def neighbor_excess(xy, groups, sizes):
    """Average observed minus random-label same-cluster shares, in pp."""
    same_sums = np.zeros((len(NEIGHBOR_COUNTS), len(sizes)))
    # Work in blocks to avoid storing the full distance matrix.
    for start in range(0, len(xy), 192):
        anchor = xy[start:start+192]
        d2 = (anchor[:, None, 0] - xy[:, 0])**2 + (anchor[:, None, 1] - xy[:, 1])**2
        d2[np.arange(len(anchor)), start+np.arange(len(anchor))] = np.inf
        neighbors = np.argsort(d2, axis=1, kind="stable")[:, :max(NEIGHBOR_COUNTS)]
        labels = groups[start:start+len(anchor)]
        same = groups[neighbors] == labels[:, None]
        for i, count in enumerate(NEIGHBOR_COUNTS):
            same_sums[i] += np.bincount(labels, weights=same[:, :count].mean(axis=1), minlength=len(sizes))
    expected = (sizes-1)/(len(groups)-1)
    return 100*(same_sums/sizes - expected)


def draw_map(ax, water, streets, bounds, overview=False):
    xlo, xhi, ylo, yhi = bounds
    ax.set(xlim=(xlo, xhi), ylim=(ylo, yhi), xticks=[], yticks=[], facecolor=LAND)
    ax.set_aspect(KM_PER_DEGREE[1]/KM_PER_DEGREE[0])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Use the fixed viewport to clip these layers without scanning their bounds.
    for path in water:
        ax.add_artist(PathPatch(path, facecolor=WATER, edgecolor=WATER, lw=.20, zorder=1))
    widths = (.26, .57) if overview else (.14, .30)
    for path, width in zip(streets, widths):
        ax.add_artist(PathPatch(path, facecolor="none", edgecolor="white", lw=width,
                                capstyle="round", joinstyle="round", zorder=2))


def draw_overview_context(ax):
    for x, y, name in [(-122.475, 47.645, "Puget\nSound"),
                       (-122.225, 47.626, "Lake\nWashington")]:
        ax.text(x, y, name, ha="center", va="center", fontsize=9.4,
                fontfamily="Times New Roman", fontstyle="italic", color="#152C46", zorder=6)
    ax.text(-122.420, 47.617, "Seattle", ha="center", va="center", fontsize=10.2,
            fontfamily="Times New Roman", fontstyle="italic", weight="bold", zorder=6)
    west, east = ax.get_xlim()
    south, north = ax.get_ylim()
    x = west + .726*(east-west)
    y = south + .016*(north-south)
    ticks = x + np.array([0, 2.5, 5])/KM_PER_DEGREE[0]
    ax.add_patch(Rectangle((.652, .007), .342, .065, transform=ax.transAxes,
                           facecolor="white", alpha=.92, edgecolor="none", zorder=3))
    ax.hlines(y, ticks[0], ticks[-1], colors=INK, linewidth=.95, zorder=3.5)
    ax.vlines(ticks, y, y+.0025, colors=INK, linewidth=.9, zorder=3.5)
    for tick, label in zip(ticks, ("0", "2.5", "5 km")):
        ax.text(tick, y+.0038, label, fontsize=7.8, ha="center", va="bottom", zorder=6)


def draw_north_arrow(ax):
    compass = AnchoredDrawingArea(14, 27, 0, 0, loc="lower left",
                                  pad=0, borderpad=.55, frameon=False)
    compass.drawing_area.add_artist(FancyArrowPatch(
        (7, 1), (7, 18), arrowstyle="-|>", mutation_scale=8,
        linewidth=.8, color=INK, shrinkA=0, shrinkB=0))
    compass.drawing_area.add_artist(Text(
        7, 23, "N", ha="center", va="center", fontsize=7.8,
        fontweight="bold", color=INK))
    compass.set_zorder(9)
    ax.add_artist(compass)
    return compass


def draw_clusters(points, groups, cluster_names, sizes, hydrography, streets, *, named):
    ref_width, ref_height = REFERENCE_SIZE
    scale = WIDTH/ref_width
    fig = plt.figure(figsize=(WIDTH, scale*ref_height))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.95, 1, 1, 1],
                          wspace=.095, hspace=.17,
                          left=.014, right=.986, bottom=.040, top=.931)
    overall = fig.add_subplot(gs[:, 0])
    maps = [fig.add_subplot(gs[0, j]) for j in range(1, 4)]
    maps += [fig.add_subplot(gs[1, j]) for j in range(1, 3)]
    legend = fig.add_subplot(gs[1, 3])
    title_slots = [ax.get_position() for ax in maps]
    for ax, slot in zip(maps, title_slots):
        ax.set_position((slot.x0, slot.y0, slot.width, slot.height-.053))

    overview_bounds = viewport(overall, south=OVERVIEW_SOUTH)
    facet_bounds = viewport(maps[0])
    west, east, _, north = facet_bounds
    extent = (min(west, overview_bounds[0]), max(east, overview_bounds[1]),
              OVERVIEW_SOUTH, north)
    water = [path for archive in hydrography for path in water_paths(archive, extent)]
    assert water

    border = "#A0A3A6"
    fig.add_artist(Rectangle((.006, .016), .988, .973, transform=fig.transFigure,
                             fill=False, edgecolor=border, lw=.65, zorder=8))
    overview_box = overall.get_position()
    divider = (overview_box.x1+title_slots[0].x0)/2
    fig.add_artist(Line2D([divider, divider], [.016, .989], transform=fig.transFigure,
                          color=border, lw=.65, zorder=8))
    group_kind = "regime" if named else "cluster"
    for x, title in [(overview_box.x0, f"(a) All POIs ({len(cluster_names)} {group_kind}s)"),
                     (title_slots[0].x0, f"(b) Spatial distribution by {group_kind}")]:
        fig.text(x, .962, title, fontsize=10, weight="bold", va="center")

    draw_map(overall, water, streets, overview_bounds, overview=True)
    # Randomize draw order to balance overplotting across clusters.
    order = np.random.default_rng(2020).permutation(len(points))
    overall.scatter(*points[order].T, c=CLUSTER_COLORS[groups[order]], s=1.8,
                    alpha=.88, linewidths=0, zorder=4)
    draw_overview_context(overall)

    # Matplotlib marker areas are in points squared. Keep their relative size
    # from the 16.2-inch reference when exporting at the manuscript width.
    area_scale = scale**2
    for k, (ax, slot, name, color) in enumerate(zip(maps, title_slots, cluster_names, CLUSTER_COLORS)):
        selected = groups == k
        draw_map(ax, water, streets, facet_bounds)
        ax.scatter(*points[~selected].T, color=CONTEXT_COLOR, s=1.4*area_scale, alpha=.23,
                   linewidths=0, zorder=3)
        ax.scatter(*points[selected].T, color=color, s=(10 if k == 0 else 3.6)*area_scale, alpha=.80,
                   linewidths=0, zorder=4)
        strip_height = .043
        strip_y = slot.y1-strip_height
        fig.add_artist(FancyBboxPatch((slot.x0, strip_y), slot.width, strip_height,
                                      boxstyle="round,pad=0,rounding_size=.0035",
                                      mutation_aspect=ref_width/ref_height,
                                      transform=fig.transFigure, facecolor=TITLE_COLORS[k], lw=0))
        if named:
            title = name
        else:
            cluster, source = name.split(": ", 1)
            title = f"{cluster} ({source})"
        fig.text(slot.x0+slot.width/2, strip_y+strip_height/2,
                 title, fontsize=8.65, weight="bold",
                 ha="center", va="center")

    legend.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    legend.set_axis_off()
    legend_box = legend.get_position()
    if named:
        legend.set_position((legend_box.x0-.006, legend_box.y0, legend_box.width+.013, legend_box.height))
        legend_box = legend.get_position()
    legend_aspect = (legend_box.width*ref_width)/(legend_box.height*ref_height)
    legend.add_patch(FancyBboxPatch((0, .073), 1, .855, boxstyle="round,pad=0,rounding_size=.067",
                                    mutation_aspect=legend_aspect,
                                    transform=legend.transAxes, facecolor="#F2F6FD",
                                    edgecolor="#6080AD", lw=.64, clip_on=False, zorder=0))
    marker_x, label_x, count_x = (.070, .122, .955) if named else (.112, .187, .907)
    legend.text(marker_x-.02, .851, group_kind.capitalize(), fontsize=9.3, weight="bold", va="center")
    legend.text(count_x, .851, "POIs", fontsize=8.6, weight="bold", ha="right", va="center")
    legend.hlines(.796, marker_x-.02, count_x, colors="#BECDE0", linewidth=.55)
    for k, (name, color, size) in enumerate(zip(cluster_names, CLUSTER_COLORS, sizes)):
        y = .727-k*.111
        label = name if named else f"{k+1}  {name.split(': ', 1)[1]}"
        legend.scatter(marker_x, y, s=19, color=color, linewidths=0)
        legend.text(label_x, y, label, fontsize=8.5, va="center")
        legend.text(count_x, y, f"{size:,}", fontsize=8.6, ha="right", va="center")
    legend.hlines(.222, marker_x-.02, count_x, colors="#BECDE0", linewidth=.55)
    legend.scatter(marker_x, .148, s=14, color=CONTEXT_COLOR, linewidths=0)
    legend.text(label_x, .148, f"Other {group_kind}s", fontsize=8.5, va="center")
    for ax in [overall, *maps]:
        draw_north_arrow(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--hydrography-zip", type=Path, action="append", required=True,
                        help="2020 TIGER/Line area hydrography archive (repeat for each county)")
    parser.add_argument("--streets", type=Path, default=STREET_FILE,
                        help="Compact TIGER/Line road geometry for the Seattle view")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assignments = pd.read_csv(args.assignments)
    with zipfile.ZipFile(args.panel) as panel:
        coordinates = pd.read_csv(io.BytesIO(panel.read("poi_attributes.csv")),
                                  usecols=["placekey", "latitude", "longitude"])
    assert len(assignments) == len(coordinates) == 9884
    points = assignments.merge(coordinates, on="placekey", validate="one_to_one")
    assert len(points) == 9884 and points[["latitude", "longitude"]].notna().all().all()
    xlo, xhi, ylo, yhi = BOUNDS
    shown = points.loc[points.longitude.between(xlo, xhi) & points.latitude.between(ylo, yhi)].sort_values("placekey")
    outside = len(points) - len(shown)
    named = points.cluster.isin(REGIME_NAMES).all()
    if named:
        cluster_names = REGIME_NAMES
    else:
        cluster_names = sorted(points.cluster.dropna().unique())
        if (points.cluster.isna().any() or len(cluster_names) != len(CLUSTER_COLORS)
                or any(not name.startswith(f"Cluster {k+1}: ")
                       for k, name in enumerate(cluster_names))):
            raise ValueError("Expected attribution regime names or five numbered cluster labels")
    groups = pd.Categorical(shown.cluster, categories=cluster_names).codes
    sizes = np.bincount(groups, minlength=len(cluster_names))
    locations = shown[["longitude", "latitude"]].to_numpy()
    xy = (locations-GRID_ORIGIN)*KM_PER_DEGREE
    active = np.flatnonzero(sizes)
    active_codes = np.searchsorted(active, groups)
    cells, totals, strongest, significant, cell_excess = grid_enrichment(xy, active_codes, sizes[active])
    local_excess = neighbor_excess(xy, active_codes, sizes[active])
    style()
    fig = draw_clusters(locations, groups, cluster_names, sizes,
                        args.hydrography_zip, street_paths(args.streets), named=named)
    title = "Seattle attribution regimes" if named else "Seattle attribution clusters"
    validate_text(fig, title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=2000,
                metadata={"Title": f"Spatial distribution of {title}",
                          "Subject": "GNN-static attribution; 2020 TIGER/Line roads and hydrography"})
    plt.close(fig)
    group_label = "Regime" if named else "Cluster"
    print(f"{len(shown):,} mapped POIs, {outside} outside the extent. {group_label} counts: {sizes.tolist()}.")
    print(f"{len(cells)} tested cells ({totals.sum():,} POIs), {significant.sum()} overrepresented.")
    for cell, group, excess in zip(cells[significant], strongest[significant], cell_excess[significant]):
        label = cluster_names[active[group]] if named else f"Cluster {active[group]+1}"
        print(f"Cell {cell.tolist()}, {label}: {excess:+.2f} pp")
    for count, excess in zip(NEIGHBOR_COUNTS, local_excess):
        excess_by_group = [None] * len(cluster_names)
        for group, value in zip(active, excess.round(3).tolist()):
            excess_by_group[group] = value
        print(f"{count} neighbors, same-{group_label.lower()} excess (pp): {excess_by_group}")


if __name__ == "__main__":
    main()
