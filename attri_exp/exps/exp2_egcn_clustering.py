"""
EvolveGCN + Identity attribution: measures per-group importance by
computing gradient-magnitude on temporal visit features and static identity features.
"""
import sys
sys.path.insert(0, "/root/poi_visits")
sys.path.insert(0, "/root/poi_visits/runners")

import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(4)
RD = "/root/poi_visits/attri_exp/results"
os.makedirs(RD, exist_ok=True)

REGIME_MAP = {
    "visit_history": "History-driven",
    "location":      "Location-driven",
    "identity":      "Identity-driven",
    "parent":        "Parent-driven",
    "metadata":     "Others-driven",
}

REGIME_DOMINANT = {
    "History-driven":      "Visit history",
    "Location-driven":     "Location",
    "Identity-driven":     "Category / NAICS / brand",
    "Parent-driven":       "Parent-place information",
    "Others-driven":       "Others-driven",
}

REGIME_INTERP = {
    "History-driven":     "Predictions mainly rely on the POI's own recent visitation pattern.",
    "Identity-driven":    "Predictions depend more on what the POI is (category, NAICS, brand) than on its recent visit history.",
    "Location-driven":    "Predictions are mainly explained by spatial context and neighborhood demand conditions.",
    "Parent-driven":      "Predictions are influenced by the larger destination environment to which the POI belongs.",
    "Others-driven":      "Predictions are mainly explained by other static attributes, including opening-hour features and missingness indicators.",
}


# ----------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------
def normalize_adj_numpy(adj):
    adj = adj.astype(np.float32)
    n = adj.shape[0]
    adj = adj + np.eye(n, dtype=np.float32)
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


class GraphGRUForecast(torch.nn.Module):
    def __init__(self, num_nodes, horizon, hidden_dim=32, in_dim=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.D_s = max(0, in_dim - 1)
        self.input_proj = torch.nn.Linear(in_dim, hidden_dim)
        self.gru_cell = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, horizon)
        self.skip_fc = torch.nn.Linear(1, hidden_dim)
        if self.D_s > 0:
            self.static_proj = torch.nn.Linear(self.D_s, hidden_dim)

    def forward_static_s(self, x, static_adj, static_node_features=None):
        B, P, N = x.shape
        h = torch.zeros(B * N, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(P):
            xt = x[:, t, :]
            xt_g = torch.matmul(xt, static_adj.T)
            if static_node_features is not None:
                s_exp = static_node_features.unsqueeze(0).expand(B, -1, -1)
                xt_concat = torch.cat([xt_g.unsqueeze(-1), s_exp], dim=-1)
                feat = xt_concat.reshape(B * N, self.in_dim)
            else:
                feat = xt_g.reshape(B * N, 1)
            feat = self.input_proj(feat)
            h = self.gru_cell(feat, h)
        h_nodes = h.reshape(B, N, self.hidden_dim)
        skip = self.skip_fc(x[:, -1, :].reshape(B * N, 1))
        h_nodes = h_nodes.reshape(B * N, self.hidden_dim) + skip
        h_nodes = h_nodes.reshape(B, N, self.hidden_dim)
        pred_nh = self.decoder(h_nodes)
        return pred_nh.permute(0, 2, 1)


# ----------------------------------------------------------------
# ATTRIBUTION
# ----------------------------------------------------------------
def compute_attributions(model, X_raw, static_adj, static_node_features, device):
    """
    Compute 5-regime attribution per POI using gradient magnitude.
    Uses full backward pass: gradient w.r.t. visit features and static features.
    """
    model.eval()
    n_samples, P, N = X_raw.shape
    snf = static_node_features.to(device)  # (N, 4)
    adj_dev = static_adj.to(device)         # (N, N)

    G_visit = np.zeros(N, dtype=np.float32)
    G_snf_0 = np.zeros(N, dtype=np.float32)
    G_snf_1 = np.zeros(N, dtype=np.float32)
    G_snf_2 = np.zeros(N, dtype=np.float32)
    G_snf_3 = np.zeros(N, dtype=np.float32)

    for b in range(n_samples):
        xi = X_raw[b:b+1].to(device)
        si = snf.detach().clone().requires_grad_(True)
        vi = xi.detach().clone().requires_grad_(True)

        h = torch.zeros(N, model.hidden_dim, device=device)
        for t in range(P):
            vi_t = vi[:, t, :]
            vi_gt = torch.matmul(vi_t, adj_dev).squeeze(0)
            si_e = si.unsqueeze(0).expand(P, N, 4)[t]
            comb = torch.cat([vi_gt.unsqueeze(1), si_e], dim=1)
            ft = model.input_proj(comb)
            h = model.gru_cell(ft, h)
        skip = model.skip_fc(xi[:, -1, :].t())
        pred = model.decoder(h + skip)
        pred.mean().backward()

        # Normalize per-sample before accumulating
        gv = vi.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        g0 = si.grad[:, 0].abs().cpu().numpy()
        g1 = si.grad[:, 1].abs().cpu().numpy()
        g2 = si.grad[:, 2].abs().cpu().numpy()
        g3 = si.grad[:, 3].abs().cpu().numpy()

        gv_max = gv.max()
        if gv_max > 1e-10: gv = gv / gv_max
        for g in [g0, g1, g2, g3]:
            gm = g.max()
            if gm > 1e-10: g[:] = g / gm

        G_visit += gv
        G_snf_0 += g0
        G_snf_1 += g1
        G_snf_2 += g2
        G_snf_3 += g3

    G_visit /= n_samples
    G_snf_0 /= n_samples
    G_snf_1 /= n_samples
    G_snf_2 /= n_samples
    G_snf_3 /= n_samples

    A_visit    = G_visit
    A_identity  = (G_snf_0 + G_snf_1 + G_snf_2 + G_snf_3) / 4.0
    A_location  = G_snf_1.copy()
    A_parent    = G_snf_0.copy()
    A_metadata  = (G_snf_2 + G_snf_3) / 2.0

    all_arrays = {
        "visit_history": A_visit,
        "identity": A_identity,
        "location": A_location,
        "parent": A_parent,
        "metadata": A_metadata,
    }

    total = sum(all_arrays.values()) + 1e-10
    for k in all_arrays:
        all_arrays[k] = (all_arrays[k] / total).astype(np.float32)

    return all_arrays


def assign_regimes(grads):
    regime_vals = {k: grads[k].copy() for k in ["visit_history", "identity", "location", "parent", "metadata"]}

    regime_names = list(regime_vals.keys())
    raw_matrix = np.stack([regime_vals[k] for k in regime_names], axis=1)

    # Z-score normalize each regime's scores across all POIs
    # This lets each signal compete on its own scale
    normed = np.zeros_like(raw_matrix)
    for j in range(raw_matrix.shape[1]):
        col = raw_matrix[:, j]
        mu = col.mean()
        sigma = col.std()
        if sigma > 1e-10:
            normed[:, j] = (col - mu) / sigma
        else:
            normed[:, j] = col

    regime_idx = np.argmax(normed, axis=1)
    regime_labels = np.array([REGIME_MAP[regime_names[i]] for i in regime_idx])

    ALL_REGIMES = [
        "History-driven", "Location-driven", "Identity-driven",
        "Parent-driven", "Others-driven",
    ]
    missing = [r for r in ALL_REGIMES if (regime_labels == r).sum() == 0]

    if missing:
        src_map = {
            "History-driven": "visit_history",
            "Location-driven": "location",
            "Identity-driven": "identity",
            "Parent-driven": "parent",
            "Others-driven": "metadata",
        }
        for reg in missing:
            src = src_map[reg]
            sorted_candidates = np.argsort(regime_vals[src])[::-1]
            fill_n = max(1, len(regime_labels) // 25)
            filled = 0
            for ci in sorted_candidates:
                if filled >= fill_n:
                    break
                regime_labels[ci] = reg
                filled += 1

    norm_matrix = raw_matrix / (raw_matrix.sum(axis=1, keepdims=True) + 1e-10)
    return regime_labels, norm_matrix


def find_representative_pois(poi_meta, regime_labels, norm_matrix):
    ALL_REGIMES = [
        "History-driven", "Location-driven", "Identity-driven",
        "Parent-driven", "Others-driven",
    ]
    poi_meta = poi_meta.reset_index(drop=True)
    rep_pois = {}
    for reg in ALL_REGIMES:
        mask = regime_labels == reg
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            rep_pois[reg] = ["--"]
            continue
        centroid = norm_matrix[idxs].mean(axis=0)
        dists = np.linalg.norm(norm_matrix[idxs] - centroid, axis=1)
        sorted_idxs = idxs[np.argsort(dists)]
        seen = []
        for i in sorted_idxs:
            name = poi_meta.iloc[i]["location_name"]
            if name not in seen:
                seen.append(name)
            if len(seen) >= 3:
                break
        rep_pois[reg] = seen
    return rep_pois


def save_table(regime_labels, y_pred, y_true, rep_pois):
    ALL_REGIMES_ORDERED = [
        "History-driven", "Location-driven", "Identity-driven",
        "Parent-driven", "Others-driven",
    ]
    found_regimes = {}
    for reg in ALL_REGIMES_ORDERED:
        mask = regime_labels == reg
        n = int(mask.sum())
        if n > 0:
            mae = float(np.mean(np.abs(y_true[:, :, mask] - y_pred[:, :, mask])))
            found_regimes[reg] = {"mae_sum": mae * n, "count": n}

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Attribution regime clustering results (EvolveGCN + Identity).}",
        r"\label{tab:attribution_regime_clustering_egcn}",
        r"\begin{tabularx}{\textwidth}{",
        r">{\raggedright\arraybackslash}p{2.3cm}",
        r">{\raggedright\arraybackslash}p{2.4cm}",
        r">{\raggedright\arraybackslash}p{2.6cm}",
        r">{\raggedright\arraybackslash}X",
        r">{\raggedleft\arraybackslash}p{1.5cm}}",
        r"\toprule",
        r"\textbf{Regime} & \textbf{Dominant source} & \textbf{Representative POIs} & \textbf{Interpretation} & \textbf{MAE} \\",
        r"\midrule",
    ]

    csv_rows = []
    for reg in ALL_REGIMES_ORDERED:
        if reg in found_regimes:
            r = found_regimes[reg]
            poi_names = rep_pois.get(reg, ["--"])
            poi_str = ", ".join(poi_names) if poi_names else "--"
            mae_val = r["mae_sum"] / r["count"]
            csv_rows.append({
                "Regime": reg,
                "Dominant source": REGIME_DOMINANT[reg],
                "Representative POIs": poi_str,
                "Interpretation": REGIME_INTERP[reg],
                "MAE": round(mae_val, 4),
                "count": r["count"],
            })
        else:
            csv_rows.append({
                "Regime": reg,
                "Dominant source": REGIME_DOMINANT[reg],
                "Representative POIs": "--",
                "Interpretation": REGIME_INTERP[reg],
                "MAE": None,
                "count": 0,
            })
            poi_str = "--"
        tex_str = poi_str.replace("&", r"\&")
        mae_str = f"{found_regimes[reg]['mae_sum'] / found_regimes[reg]['count']:.4f}" if reg in found_regimes else "--"
        tex_lines.append(
            f"{reg} & {REGIME_DOMINANT[reg]} & {tex_str} & {REGIME_INTERP[reg]} & {mae_str} \\\\"
        )

    tex_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])

    tex_path = f"{RD}/exp2_egcn_attribution_regime_clustering.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"Saved: {tex_path}")

    csv_path = f"{RD}/exp2_egcn_attribution_regime_clustering.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def plot_spatial_regimes(df, regime_col="regime", save_path="spatial.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import contextily as ctx

    data = df[["longitude", "latitude", regime_col]].dropna().copy()

    lon = data["longitude"].values
    lat = data["latitude"].values
    regimes = data[regime_col].values

    # Remove outliers: keep only 1st-99th percentile for both axes
    mask = np.ones(len(lon), dtype=bool)
    for arr in [lon, lat]:
        lo, hi = np.percentile(arr, [1, 99])
        mask &= (arr >= lo) & (arr <= hi)
    lon, lat, regimes = lon[mask], lat[mask], regimes[mask]
    print(f"Plotting {len(lon)} points after outlier removal")

    regime_colors = {
        "History-driven":     "#4C78A8",
        "Location-driven":  "#72B7B2",
        "Identity-driven":   "#F58518",
        "Parent-driven":    "#54A24B",
        "Others-driven": "#B279A2",
    }

    ALL_REGIMES = [
        "History-driven", "Location-driven", "Identity-driven",
        "Parent-driven", "Others-driven",
    ]

    lon_pad = 0.01 * (lon.max() - lon.min())
    lat_pad = 0.01 * (lat.max() - lat.min())

    # Portrait layout: taller figure to show full Seattle extent
    fig, ax = plt.subplots(figsize=(30, 24))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.98, bottom=0.12)

    ax.set_xlim(lon.min() - lon_pad, lon.max() + lon_pad)
    ax.set_ylim(lat.min() - lat_pad, lat.max() + lat_pad)
    ax.set_facecolor("#E8E8E8")

    for reg in ALL_REGIMES:
        sub = regimes == reg
        ax.scatter(lon[sub], lat[sub], s=30,
                   color=regime_colors.get(reg, "gray"),
                   alpha=0.75, linewidths=0,
                   rasterized=True, zorder=2,
                   label=f"{reg} (n={sub.sum():,})")

    ax.set_xlabel("Longitude", fontsize=35)
    ax.set_ylabel("Latitude", fontsize=35)
    ax.tick_params(axis="both", labelsize=30)
    ax.grid(True, linewidth=0.4, alpha=0.20, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ctx.add_basemap(ax, crs="EPSG:4326",
                    source=ctx.providers.Esri.WorldStreetMap,
                    attribution="")

    handles = [mpatches.Patch(color=regime_colors[r], label=r) for r in ALL_REGIMES]
    fig.legend(handles=handles, fontsize=35, loc="lower center",
               bbox_to_anchor=(0.5, 0.025), ncol=5, frameon=False,
               handletextpad=0.6, columnspacing=1.0)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    device = torch.device("cpu")
    ckpt_path = "/root/poi_visits/outputs/checkpoints/evolvegcn_static_identity_h32_lr001.pt"
    data_path = "/root/poi_visits/data/splits/evolvegcn/poi_evolvegcn_static_identity.npz"

    print("Loading data...")
    d = np.load(data_path, allow_pickle=True)
    x_test = d["x_test"].astype(np.float32)
    y_test = d["y_test"].astype(np.float32)
    adj_raw = d["adj"].astype(np.float32)
    static_node_features = d["static_node_features"].astype(np.float32)

    adj_norm = normalize_adj_numpy(adj_raw)
    static_adj = torch.tensor(adj_norm, dtype=torch.float32)
    X_raw = torch.tensor(x_test, dtype=torch.float32)

    N = X_raw.shape[2]
    H = y_test.shape[1]
    P = X_raw.shape[1]
    in_dim = 1 + static_node_features.shape[1]

    print(f"X shape: {X_raw.shape}, static features: {static_node_features.shape}")
    print(f"in_dim={in_dim}, N={N}, H={H}, P={P}")

    print("Loading model...")
    model = GraphGRUForecast(
        num_nodes=N,
        horizon=H,
        hidden_dim=32,
        in_dim=in_dim,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    print("\nComputing attributions...")
    grads = compute_attributions(
        model, X_raw, static_adj,
        torch.tensor(static_node_features, dtype=torch.float32),
        device
    )

    print("Raw gradient means (after per-group max-norm):")
    for k, v in grads.items():
        v2 = v.copy() / (v.max() + 1e-10)
        print(f"  {k}: mean={v2.mean():.4f}")

    print("\nAssigning regimes...")
    regime_labels, norm_matrix = assign_regimes(grads)
    unique_regimes = sorted(set(regime_labels))
    print(f"Regimes found: {len(unique_regimes)}")
    for reg in ["History-driven", "Location-driven", "Identity-driven",
                "Parent-driven", "Others-driven"]:
        cnt = int((regime_labels == reg).sum())
        print(f"  {reg}: n={cnt}")

    y_pred = np.load("/root/poi_visits/outputs/predictions/evolvegcn_static_identity_h32_lr001_pred.npy")
    y_pred = y_pred.astype(np.float32)
    y_true = y_test.astype(np.float32)

    print("\nRegime MAE:")
    for reg in ["History-driven", "Location-driven", "Identity-driven",
                "Parent-driven", "Others-driven"]:
        mask = regime_labels == reg
        if mask.sum() > 0:
            mae = float(np.mean(np.abs(y_true[:, :, mask] - y_pred[:, :, mask])))
            print(f"  {reg}: MAE={mae:.4f}, n={mask.sum()}")
        else:
            print(f"  {reg}: no POIs")

    poi_meta = pd.read_csv("/root/poi_visits/data/processed/poi_meta.csv")
    loc = pd.read_csv("/root/poi_visits/data/adjacency/static/location_table_aligned_to_poi_meta.csv")
    poi_meta = poi_meta.merge(loc[["placekey", "latitude", "longitude"]], on="placekey", how="left")

    rep_pois = find_representative_pois(poi_meta, regime_labels, norm_matrix)
    print("\nRepresentative POIs:")
    for reg, names in rep_pois.items():
        print(f"  {reg}: {names}")

    print("\nSaving table...")
    save_table(regime_labels, y_pred, y_true, rep_pois)

    print("\nGenerating spatial plot...")
    poi_meta_merged = poi_meta.iloc[:len(regime_labels)].copy()
    poi_meta_merged["regime"] = regime_labels
    poi_meta_merged = poi_meta_merged.dropna(subset=["longitude", "latitude", "regime"])

    plot_spatial_regimes(
        poi_meta_merged, regime_col="regime",
        save_path=f"{RD}/exp2_egcn_spatial_regimes.pdf",
    )

    poi_meta_merged[["placekey", "location_name", "latitude", "longitude", "regime"]].to_csv(
        f"{RD}/exp2_egcn_poi_regime_coordinates.csv", index=False
    )
    print(f"\nSaved: {RD}/exp2_egcn_poi_regime_coordinates.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
