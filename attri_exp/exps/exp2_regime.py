import sys
sys.path.insert(0, "/root/poi_visits")
sys.path.insert(0, "/root/poi_visits/external/MTGNN")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

torch.set_num_threads(4)


def load_mtgnn_model(checkpoint_path, device="cpu"):
    from external.MTGNN.net import gtnet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt if isinstance(ckpt, dict) else ckpt

    model = gtnet(
        gcn_true=True, buildA_true=True, gcn_depth=2,
        num_nodes=9926, device=device, predefined_A=None,
        dropout=0.0, subgraph_size=20, node_dim=40,
        dilation_exponential=1, conv_channels=32,
        residual_channels=32, skip_channels=64, end_channels=128,
        seq_length=19, in_dim=10, out_dim=7, layers=3,
        propalpha=0.05, tanhalpha=3,
        cat_vocab_size=118, cat_emb_dim=8,
        naics2_vocab_size=23, naics2_emb_dim=8,
        naics4_vocab_size=118, naics4_emb_dim=8,
        brand_vocab_size=8, brand_emb_dim=8,
        parent_vocab_size=4, parent_emb_dim=8,
    ).to(device)

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, 19


def load_data():
    d_test = np.load("/root/poi_visits/data/splits/mtgnn/test.npz")

    X_raw = torch.tensor(d_test["x"], dtype=torch.float32)
    y_test = torch.tensor(d_test["y"], dtype=torch.float32)

    def clip(ids, lo, hi):
        return torch.tensor(np.clip(ids.astype(np.int64), lo, hi), dtype=torch.long)

    brand_ids  = clip(d_test["x"][0, 0, :, 10], 0, 7)
    parent_ids = clip(d_test["x"][0, 0, :, 11], 0, 3)
    naics2_ids = clip(d_test["x"][0, 0, :, 12], 0, 22)
    naics4_ids = clip(d_test["x"][0, 0, :, 13], 0, 117)
    cat_ids    = clip(d_test["x"][0, 0, :,  9], 0, 117)

    return X_raw, y_test, brand_ids, parent_ids, naics2_ids, naics4_ids, cat_ids


def build_node_info():
    pm = pd.read_csv("/root/poi_visits/data/processed/poi_meta.csv")
    lt = pd.read_csv("/root/poi_visits/data/raw/location_table.csv")
    sa = pd.read_csv("/root/poi_visits/data/raw/Seattle_all.csv", usecols=["placekey", "location_name"])
    sa = sa.drop_duplicates("placekey").rename(columns={"location_name": "poi_name"})

    merged = pm.merge(lt[["placekey", "latitude", "longitude"]], on="placekey", how="left")
    merged = merged.merge(sa, on="placekey", how="left")
    return merged


def compute_attributions(model, X_raw, brand_ids, parent_ids, naics2_ids, naics4_ids, cat_ids,
                         receptive_field, device, batch_size=2):
    model.eval()
    n_samples, T_in, N, F_in = X_raw.shape
    in_dim = 10

    be = model.brand_emb.weight.detach().cpu().numpy()
    pe = model.parent_emb.weight.detach().cpu().numpy()
    ne = model.naics2_emb.weight.detach().cpu().numpy()
    n4 = model.naics4_emb.weight.detach().cpu().numpy()
    ce = model.cat_emb.weight.detach().cpu().numpy()

    A_brand  = np.linalg.norm(be[brand_ids.numpy()],  axis=1).astype(np.float32)
    A_parent = np.linalg.norm(pe[parent_ids.numpy()], axis=1).astype(np.float32)
    A_naics2 = np.linalg.norm(ne[naics2_ids.numpy()], axis=1).astype(np.float32)
    A_naics4 = np.linalg.norm(n4[naics4_ids.numpy()], axis=1).astype(np.float32)
    A_cat    = np.linalg.norm(ce[cat_ids.numpy()],    axis=1).astype(np.float32)

    for arr in [A_brand, A_parent, A_naics2, A_naics4, A_cat]:
        m = arr.max()
        if m > 1e-10:
            arr /= m

    A_visit = np.zeros(N, dtype=np.float32)

    gb_buf = torch.zeros(N, device="cpu")
    gp_buf = torch.zeros(N, device="cpu")
    gn2_buf = torch.zeros(N, device="cpu")
    gn4_buf = torch.zeros(N, device="cpu")
    gc_buf = torch.zeros(N, device="cpu")

    def mk_hook(buf, idx):
        def h(grad):
            normed = grad.abs().mean(dim=1)
            counted = normed[idx]
            buf.add_(counted.cpu())
        return h

    hooks = [
        model.brand_emb.weight.register_hook(mk_hook(gb_buf, brand_ids.to(device))),
        model.parent_emb.weight.register_hook(mk_hook(gp_buf, parent_ids.to(device))),
        model.naics2_emb.weight.register_hook(mk_hook(gn2_buf, naics2_ids.to(device))),
        model.naics4_emb.weight.register_hook(mk_hook(gn4_buf, naics4_ids.to(device))),
        model.cat_emb.weight.register_hook(mk_hook(gc_buf, cat_ids.to(device))),
    ]

    bk_dev = brand_ids.to(device)
    pk_dev = parent_ids.to(device)
    nk_dev = naics2_ids.to(device)
    nf_dev = naics4_ids.to(device)
    ck_dev = cat_ids.to(device)

    sample_count = 0

    for b_start in range(0, n_samples, batch_size):
        b_end = min(b_start + batch_size, n_samples)
        batch_raw = X_raw[b_start:b_end]

        x_vis = batch_raw[:, :, :, :in_dim].permute(0, 3, 2, 1).to(device)
        x_padded = F.pad(x_vis, (receptive_field - T_in, 0))

        for bi in range(b_end - b_start):
            px = x_padded[bi:bi+1].detach().requires_grad_(True)
            out = model(px, brand_idx=bk_dev[None, :],
                        parent_idx=pk_dev[None, :],
                        naics2_idx=nk_dev[None, :],
                        naics4_idx=nf_dev[None, :])
            out.mean().backward()
            sample_count += 1
            A_visit += px.grad[0, :in_dim, :, :].abs().mean(dim=(0, 2)).cpu().numpy()
            px.grad.zero_()

    for h in hooks:
        h.remove()

    A_brand_hook  = gb_buf.numpy()[:N].astype(np.float32)
    A_parent_hook = gp_buf.numpy()[:N].astype(np.float32)
    A_naics2_hook = gn2_buf.numpy()[:N].astype(np.float32)
    A_naics4_hook = gn4_buf.numpy()[:N].astype(np.float32)
    A_cat_hook    = gc_buf.numpy()[:N].astype(np.float32)

    A_visit /= sample_count
    for arr in [A_brand_hook, A_parent_hook, A_naics2_hook, A_naics4_hook, A_cat_hook]:
        m = arr.max()
        if m > 1e-10:
            arr /= m

    m = A_visit.max()
    if m > 1e-10:
        A_visit /= m

    A_identity = (A_brand_hook + A_parent_hook + A_naics2_hook + A_naics4_hook + A_cat_hook) / 5

    total = A_visit + A_identity + 1e-10

    return {
        "visit_history": (A_visit / total).astype(np.float32),
        "identity":      (A_identity / total).astype(np.float32),
        "brand":        (A_brand_hook / total).astype(np.float32),
        "parent":       (A_parent_hook / total).astype(np.float32),
        "naics2":       (A_naics2_hook / total).astype(np.float32),
        "naics4":       (A_naics4_hook / total).astype(np.float32),
        "category":     (A_cat_hook / total).astype(np.float32),
        "metadata":     np.zeros(N, dtype=np.float32),
    }


def cluster_attributions(A, n_clusters=5):
    sources = ["visit_history", "identity", "brand", "parent", "naics2", "naics4", "category", "metadata"]
    profiles = np.stack([A[s] for s in sources], axis=1)
    profiles /= (profiles.sum(axis=1, keepdims=True) + 1e-10)

    silhouettes = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sil = silhouette_score(profiles, km.fit_predict(profiles))
        silhouettes[k] = sil

    best_k = max(silhouettes, key=silhouettes.get)
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(profiles)
    return profiles, labels, km.cluster_centers_, best_k


def dominant_source(centroid):
    names = ["Visit history", "Identity", "Brand", "Parent", "NAICS2", "NAICS4", "Category", "Metadata"]
    return names[np.argmax(centroid)]


def regime_of(dominant):
    return {
        "Visit history": "History-driven",
        "Identity": "Identity-driven",
        "Brand": "Identity-driven",
        "Parent": "Parent-driven",
        "NAICS2": "Identity-driven",
        "NAICS4": "Identity-driven",
        "Category": "Identity-driven",
        "Metadata": "Metadata-sensitive",
    }.get(dominant, dominant)


def interp(regime):
    return {
        "History-driven": "Predictions mainly rely on the POI's own recent visitation pattern.",
        "Identity-driven": "Predictions depend more on what the POI is (category, NAICS, brand) than on its recent visit history.",
        "Location-driven": "Predictions are mainly explained by spatial context and neighborhood demand conditions.",
        "Parent-driven": "Predictions are influenced by the larger destination environment to which the POI belongs.",
        "Metadata-sensitive": "Predictions are sensitive to operational information or metadata completeness.",
    }.get(regime, regime)


REGIME_ROWS = [
    ("History-driven",     "Visit history",          "--", "Predictions mainly rely on the POI's own recent visitation pattern."),
    ("Location-driven",    "Location",                "--", "Predictions are mainly explained by spatial context and neighborhood demand conditions."),
    ("Identity-driven",    "Category / NAICS / brand","--", "Predictions depend more on what the POI is than on its recent visit history."),
    ("Parent-driven",      "Parent-place information","--", "Predictions are influenced by the larger destination environment to which the POI belongs."),
    ("Metadata-sensitive", "Opening hours / missingness","--","Predictions are sensitive to operational information or metadata completeness."),
]


def save_table(labels, centroids, best_k, y_pred, y_true):
    csv_path = "/root/poi_visits/attri_exp/results/exp2_attribution_regime_clustering.csv"
    tex_path = "/root/poi_visits/attri_exp/results/exp2_attribution_regime_clustering.tex"

    rows = [{"Regime": r[0], "Dominant source": r[1],
             "Representative POIs": r[2], "Interpretation": r[3], "MAE": "--"}
            for r in REGIME_ROWS]

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Attribution regime clustering results.}",
        r"\label{tab:attribution_regime_clustering}",
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
    for r in rows:
        tex.append(f"{r['Regime']} & {r['Dominant source']} & {r['Representative POIs']} & "
                   f"{r['Interpretation']} & {r['MAE']} \\\\")
    tex.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])

    with open(tex_path, "w") as f:
        f.write("\n".join(tex))

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    print("\n".join(tex))


def fig_centroid_heatmap(centroids, best_k, out):
    names = ["Visit\nhistory", "Identity", "Brand", "Parent", "NAICS2", "NAICS4", "Category", "Metadata"]
    fig, ax = plt.subplots(figsize=(10, max(2, best_k * 0.9)))
    im = ax.imshow(centroids, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(centroids.shape[1])); ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(best_k)); ax.set_yticklabels([f"Cluster {k}" for k in range(best_k)], fontsize=9)
    ax.set_title("Attribution profile centroids by regime cluster", fontsize=11)
    for i in range(best_k):
        for j in range(centroids.shape[1]):
            v = centroids[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if v > 0.5 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Attribution share", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def fig_distribution(labels, centroids, best_k, out):
    regimes = [regime_of(dominant_source(centroids[k])) for k in range(best_k)]
    sizes = [(labels == k).sum() for k in range(best_k)]

    all_regimes = [r[0] for r in REGIME_ROWS]
    regime_c = {
        "History-driven": "#1f77b4", "Identity-driven": "#ff7f0e",
        "Location-driven": "#2ca02c", "Parent-driven": "#9467bd",
        "Metadata-sensitive": "#8c564b",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for k in range(best_k):
        axes[0].barh(k, sizes[k], color=regime_c.get(regimes[k], "gray"),
                      edgecolor="black", linewidth=0.5)
        axes[0].text(sizes[k] + 20, k, f"{sizes[k]} ({sizes[k]/len(labels)*100:.1f}%)",
                     va="center", fontsize=8)
    axes[0].set_yticks(range(best_k))
    axes[0].set_yticklabels([f"C{k}: {regimes[k]}" for k in range(best_k)], fontsize=8)
    axes[0].set_xlabel("Number of POIs", fontsize=10)
    axes[0].set_title("POI distribution across clusters", fontsize=10)
    axes[0].set_xlim(0, max(sizes) * 1.25)

    used_colors = [regime_c.get(regimes[k], "gray") for k in range(best_k)]
    wedges, _, autotexts = axes[1].pie(sizes, labels=[f"C{k}" for k in range(best_k)],
        colors=used_colors, autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="black", linewidth=0.5))
    for t in autotexts: t.set_fontsize(8)
    axes[1].set_title("Cluster proportions", fontsize=10)

    axes[0].legend(
        handles=[mpatches.Patch(color=regime_c[r], label=r) for r in sorted(set(regimes)) if r in regime_c],
        fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def fig_source_importance(A, labels, centroids, best_k, out):
    sources = ["visit_history", "identity", "brand", "parent", "naics2", "naics4", "category", "metadata"]
    names   = ["Visit history", "Identity", "Brand", "Parent", "NAICS2", "NAICS4", "Category", "Metadata"]
    regimes = [regime_of(dominant_source(centroids[k])) for k in range(best_k)]
    dom_src = [dominant_source(centroids[k]) for k in range(best_k)]

    ncols = min(best_k, 3)
    nrows = (best_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 4), squeeze=False)
    pal = plt.cm.Set2(np.linspace(0, 1, 8))

    for k in range(best_k):
        r, c = k // ncols, k % ncols
        ax = axes[r][c]
        mask = labels == k
        means = [np.mean(A[s][mask]) for s in sources]
        stds  = [np.std(A[s][mask]) / np.sqrt(max(mask.sum(), 1)) for s in sources]
        ax.barh(names, means, xerr=stds, color=pal, edgecolor="black", linewidth=0.5, capsize=2)
        ax.set_xlim(0, 1)
        ax.set_title(f"Cluster {k}: {regimes[k]}\n(n={mask.sum()})", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        if c > 0: ax.set_yticklabels([])
        ax.set_xlabel(f"Dom: {dom_src[k]}", fontsize=7, style="italic")
        ax.grid(axis="x", alpha=0.3)

    for k in range(best_k, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    plt.suptitle("Mean attribution share per regime cluster", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def fig_regime_mae(labels, centroids, best_k, y_pred, y_true, out):
    regimes = [regime_of(dominant_source(centroids[k])) for k in range(best_k)]
    maes = []
    sizes = []
    for k in range(best_k):
        mask = labels == k
        maes.append(float(np.mean(np.abs(y_true[:, :, mask] - y_pred[:, :, mask]))))
        sizes.append(int(mask.sum()))

    regime_c = {
        "History-driven": "#1f77b4", "Identity-driven": "#ff7f0e",
        "Location-driven": "#2ca02c", "Parent-driven": "#9467bd",
        "Metadata-sensitive": "#8c564b",
    }
    bcs = [regime_c.get(regimes[k], plt.cm.tab10(k / max(best_k - 1, 1))) for k in range(best_k)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(best_k), maes, color=bcs, edgecolor="black", linewidth=0.5)
    for i, (bar, mae, sz) in enumerate(zip(bars, maes, sizes)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mae:.3f}", ha="center", fontsize=9)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                f"n={sz}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(range(best_k))
    ax.set_xticklabels([f"C{k}\n({regimes[k][:8]})" for k in range(best_k)], fontsize=8)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_xlabel("Attribution regime", fontsize=10)
    ax.set_title("Regime-level MAE across attribution clusters", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    patches = [mpatches.Patch(color=regime_c[r], label=r)
               for r in sorted(set(regimes)) if r in regime_c]
    if patches:
        ax.legend(handles=patches, fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def fig_regime_map(labels, centroids, best_k, node_info, out):
    regimes = [regime_of(dominant_source(centroids[k])) for k in range(best_k)]
    regime_c = {
        "History-driven": "#1f77b4", "Identity-driven": "#ff7f0e",
        "Location-driven": "#2ca02c", "Parent-driven": "#9467bd",
        "Metadata-sensitive": "#8c564b",
    }

    lat = node_info["latitude"].values
    lon = node_info["longitude"].values

    valid = ~(np.isnan(lat) | np.isnan(lon))

    fig, ax = plt.subplots(figsize=(9, 8))

    for k in range(best_k):
        mask = (labels == k) & valid
        ax.scatter(lon[mask], lat[mask],
                   c=[regime_c.get(regimes[k], "gray")],
                   label=f"C{k}: {regimes[k]} (n={mask.sum()})",
                   alpha=0.4, s=3)

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("Geographic distribution of POI attribution regimes", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", markerscale=4)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def fig_tsne(profiles, labels, centroids, best_k, node_info, out):
    regimes = [regime_of(dominant_source(centroids[k])) for k in range(best_k)]
    regime_c = {
        "History-driven": "#1f77b4", "Identity-driven": "#ff7f0e",
        "Location-driven": "#2ca02c", "Parent-driven": "#9467bd",
        "Metadata-sensitive": "#8c564b",
    }

    lat = node_info["latitude"].values
    lon = node_info["longitude"].values
    valid = ~(np.isnan(lat) | np.isnan(lon))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    np.random.seed(42)
    tsne_coords = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000).fit_transform(profiles)

    for k in range(best_k):
        mask = labels == k
        axes[0].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=[regime_c.get(regimes[k], "gray")],
                   label=f"C{k}: {regimes[k]} (n={mask.sum()})",
                   alpha=0.5, s=5)
    axes[0].set_xlabel("t-SNE dim 1", fontsize=10)
    axes[0].set_ylabel("t-SNE dim 2", fontsize=10)
    axes[0].set_title("t-SNE of POI attribution profiles", fontsize=11)
    axes[0].legend(fontsize=7, loc="best", markerscale=3)

    for k in range(best_k):
        mask = (labels == k) & valid
        axes[1].scatter(lon[mask], lat[mask],
                   c=[regime_c.get(regimes[k], "gray")],
                   label=f"C{k}: {regimes[k]} (n={mask.sum()})",
                   alpha=0.4, s=3)
    axes[1].set_xlabel("Longitude", fontsize=10)
    axes[1].set_ylabel("Latitude", fontsize=10)
    axes[1].set_title("Geographic distribution of POI regimes", fontsize=11)
    axes[1].legend(fontsize=7, loc="upper right", markerscale=4)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def main():
    device = torch.device("cpu")
    ckpt_path = "/root/poi_visits/external/MTGNN/save/poi_identity_lr0005_do3_bs8_exp1_0.pth"

    print("Loading model..."); model, rf = load_mtgnn_model(ckpt_path, device)
    print("Loading data..."); X_raw, y_test, b_ids, p_ids, n2, n4, c_ids = load_data()
    print("Loading node info..."); node_info = build_node_info()
    print(f"X shape: {X_raw.shape}, node_info: {node_info.shape}")

    print("\nComputing attributions...")
    A = compute_attributions(model, X_raw, b_ids, p_ids, n2, n4, c_ids, rf, device)
    print("Attribution means:", {k: f"{v.mean():.4f}" for k, v in A.items()})

    print("\nClustering...")
    profiles, labels, centroids, best_k = cluster_attributions(A)
    print(f"Best K={best_k}")
    print("Silhouette:", {k: round(float(v), 3) for k, v in {
        k: silhouette_score(profiles, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(profiles))
        for k in range(2, 8)
    }.items()})
    for k in range(best_k):
        mask = labels == k
        dom = dominant_source(centroids[k])
        print(f"  C{k}: {regime_of(dom)} ({dom}), n={mask.sum()}")

    y_pred = np.load("/root/poi_visits/outputs/predictions/mtgnn_identity_lr0005_do3_bs8_pred.npy")
    y_true = y_test.numpy()

    print("\nRegime MAE:")
    for k in range(best_k):
        mask = labels == k
        mae = float(np.mean(np.abs(y_true[:, :, mask] - y_pred[:, :, mask])))
        print(f"  C{k}: MAE={mae:.4f} (n={mask.sum()})")

    rd = "/root/poi_visits/attri_exp/results"
    os.makedirs(rd, exist_ok=True)

    print("\nGenerating figures...")
    fig_centroid_heatmap(centroids, best_k, f"{rd}/exp2_attribution_profile_heatmap.png")
    fig_distribution(labels, centroids, best_k, f"{rd}/exp2_regime_distribution.png")
    fig_source_importance(A, labels, centroids, best_k, f"{rd}/exp2_source_importance.png")
    fig_regime_mae(labels, centroids, best_k, y_pred, y_true, f"{rd}/exp2_regime_mae_bar.png")
    fig_tsne(profiles, labels, centroids, best_k, node_info, f"{rd}/exp2_profile_tsne.png")
    fig_regime_map(labels, centroids, best_k, node_info, f"{rd}/exp2_regime_map.png")

    print("\nLaTeX table:")
    save_table(labels, centroids, best_k, y_pred, y_true)


if __name__ == "__main__":
    main()
