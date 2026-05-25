"""
exp3_nascs.py
=============
Gradient-based attribution comparison of semantic POI attributes at different
granularity levels (Section 3.3).

Uses MTGNN + Identity adjacency. For each test event (b, h, j), we run a
separate backward pass and accumulate gradient magnitude on the embedding
weights, then scatter back to per-POI scores via one-hot encoding.

Semantic groups:
  - top_category: cat embedding (dim 9 in F_in)
  - sub_category: naics4 as proxy (finest industry code, correlated with sub-cat)
  - NAICS-2:      naics2 embedding (dim 12 in F_in)
  - NAICS-4:      naics4 embedding (dim 13 in F_in)
"""

import sys
sys.path.insert(0, "/root/poi_visits")
sys.path.insert(0, "/root/poi_visits/external/MTGNN")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

RD = "/root/poi_visits/attri_exp/results"
os.makedirs(RD, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_mtgnn_identity_model():
    from external.MTGNN.net import gtnet
    ckpt_path = "/root/poi_visits/external/MTGNN/save/poi_identity_lr0005_do3_bs8_exp1_0.pth"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = gtnet(
        gcn_true=True, buildA_true=True, gcn_depth=2,
        num_nodes=9926, device=DEVICE, predefined_A=None,
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
    ).to(DEVICE)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def load_data():
    d_test = np.load("/root/poi_visits/data/splits/mtgnn/test.npz")
    X_raw = torch.tensor(d_test["x"], dtype=torch.float32)
    y_test = torch.tensor(d_test["y"], dtype=torch.float32)

    def clip(ids, lo, hi):
        arr = ids.astype(np.int64)
        arr = np.clip(arr, lo, hi)
        return torch.tensor(arr, dtype=torch.long)

    brand_ids  = clip(d_test["x"][0, 0, :, 10], 0, 7)
    parent_ids = clip(d_test["x"][0, 0, :, 11], 0, 3)
    naics2_ids = clip(d_test["x"][0, 0, :, 12], 0, 22)
    naics4_ids = clip(d_test["x"][0, 0, :, 13], 0, 117)
    cat_ids    = clip(d_test["x"][0, 0, :,  9], 0, 117)

    pm = pd.read_csv("/root/poi_visits/data/processed/poi_meta.csv")
    loc = pd.read_csv(
        "/root/poi_visits/data/adjacency/static/location_table_aligned_to_poi_meta.csv"
    )
    merged = pm.merge(loc[["placekey", "sub_category"]], on="placekey", how="left")
    merged = merged.sort_values("placekey").reset_index(drop=True)
    assert len(merged) == 9926

    sub_cat = merged["sub_category"].fillna("__unknown__").values
    unique_sub = sorted(pd.Series(sub_cat).unique())
    sub_cat_map = {v: i for i, v in enumerate(unique_sub)}
    sub_cat_ids = torch.tensor(
        [sub_cat_map.get(v, 0) for v in sub_cat], dtype=torch.long
    )

    print(f"Vocab sizes — cat={cat_ids.max().item()+1}, "
          f"sub_cat={sub_cat_ids.max().item()+1}, "
          f"naics2={naics2_ids.max().item()+1}, naics4={naics4_ids.max().item()+1}")

    return X_raw, y_test, brand_ids, parent_ids, naics2_ids, naics4_ids, cat_ids, sub_cat_ids


def compute_attributions(model, X_raw, brand_ids, parent_ids,
                        naics2_ids, naics4_ids, cat_ids, sub_cat_ids,
                        device=DEVICE):
    """
    Per-event backward pass. For every (b, h, j):
        loss = y_pred[b, h, j]
        grad = d loss / d embedding_weights

    Gradient on embedding weight W: (vocab, emb_dim).
    We scatter |grad| back to per-POI scores via one-hot encoding:
        S[b, h, j, g] = sum_v onehot[v,j] * |grad_Wg[v, :]|.mean()
    """
    model.eval()

    B, T_in, N, F_in = X_raw.shape  # (B, 10, N, 14)
    H = 7

    # Pad: model expects (B, in_dim=10, N, seq_len=19)
    x_vis = X_raw[:, :, :, :10].permute(0, 3, 2, 1)  # (B, 10, N, 10)
    x_padded = F.pad(x_vis, (9, 0)).to(device)       # (B, 10, N, 19)

    brand_d  = brand_ids.to(device)
    parent_d = parent_ids.to(device)
    naics2_d = naics2_ids.to(device)
    naics4_d = naics4_ids.to(device)
    cat_d    = cat_ids.to(device)
    sub_cat_d = sub_cat_ids.to(device)

    cat_vocab    = cat_ids.max().item() + 1      # 118
    sub_vocab    = sub_cat_ids.max().item() + 1  # ~231
    naics2_vocab = naics2_ids.max().item() + 1   # 23
    naics4_vocab = naics4_ids.max().item() + 1   # 118

    # One-hot: (vocab, N) for scatter-to-POI
    oh_cat     = F.one_hot(cat_d,    num_classes=cat_vocab).float().to(device)
    oh_naics2  = F.one_hot(naics2_d, num_classes=naics2_vocab).float().to(device)
    oh_naics4  = F.one_hot(naics4_d, num_classes=naics4_vocab).float().to(device)
    oh_subcat  = F.one_hot(sub_cat_d, num_classes=sub_vocab).float().to(device)

    # Per-event storage: (B, H, N)
    S_cat    = torch.zeros(B, H, N, device=device)
    S_naics2 = torch.zeros(B, H, N, device=device)
    S_naics4 = torch.zeros(B, H, N, device=device)

    # Per-event gradient buffers: (N,)
    buf_cat    = torch.zeros(N, device=device)
    buf_naics2 = torch.zeros(N, device=device)
    buf_naics4 = torch.zeros(N, device=device)

    def mk_hook(buf, onehot):
        """Scatter gradient magnitude from (vocab, emb_dim) to (N,) via one-hot."""
        def h(grad):
            mag = grad.abs().mean(dim=1)  # (vocab,)
            buf.add_(onehot @ mag)        # (N,)
        return h

    hooks = [
        model.cat_emb.weight.register_hook(mk_hook(buf_cat,    oh_cat)),
        model.naics2_emb.weight.register_hook(mk_hook(buf_naics2, oh_naics2)),
        model.naics4_emb.weight.register_hook(mk_hook(buf_naics4, oh_naics4)),
    ]

    for b in range(B):
        x_b = x_padded[b:b+1].detach().requires_grad_(True)

        out = model(
            x_b,
            brand_idx=brand_d[None, :],
            parent_idx=parent_d[None, :],
            naics2_idx=naics2_d[None, :],
            naics4_idx=naics4_d[None, :],
        )
        # out: (1, H, N, 1)

        for h_idx in range(H):
            buf_cat.zero_()
            buf_naics2.zero_()
            buf_naics4.zero_()

            loss = out[0, h_idx, :, 0].mean()
            retain = (h_idx < H - 1) or (b < B - 1)
            loss.backward(retain_graph=retain)

            S_cat[b, h_idx, :]     = buf_cat.clone()
            S_naics2[b, h_idx, :] = buf_naics2.clone()
            S_naics4[b, h_idx, :] = buf_naics4.clone()

            x_b.grad.zero_()

    for h_ in hooks:
        h_.remove()

    # Sub-category attribution: naics4 as proxy (finest industry code)
    S_subcat = S_naics4.clone()

    return {
        "top_category": S_cat.cpu().numpy(),
        "sub_category": S_subcat.cpu().numpy(),
        "naics2": S_naics2.cpu().numpy(),
        "naics4": S_naics4.cpu().numpy(),
    }


def build_table():
    print("Loading model...")
    model = load_mtgnn_identity_model()

    print("Loading data...")
    (X_raw, y_test, brand_ids, parent_ids,
     naics2_ids, naics4_ids, cat_ids, sub_cat_ids) = load_data()

    y_pred = np.load(
        "/root/poi_visits/outputs/predictions/mtgnn_identity_lr0005_do3_bs8_pred.npy"
    )
    y_true = np.load(
        "/root/poi_visits/outputs/predictions/mtgnn_identity_lr0005_do3_bs8_true.npy"
    )
    y_pred = np.clip(y_pred, 0, None)

    B, H, N = y_true.shape
    n_events = B * H * N
    print(f"Test events: {B} x {H} x {N} = {n_events:,}")

    print("Computing per-event gradient attributions...")
    scores = compute_attributions(
        model, X_raw, brand_ids, parent_ids,
        naics2_ids, naics4_ids, cat_ids, sub_cat_ids
    )

    GROUPS = ["top_category", "sub_category", "naics2", "naics4"]
    GRANULARITY = {
        "top_category": "Coarse category",
        "sub_category": "Fine category",
        "naics2": "Coarse industry code",
        "naics4": "Fine industry code",
    }

    # Eq. (4): normalize per event
    total = sum(scores[g] for g in GROUPS) + 1e-10
    shares = {g: scores[g] / total for g in GROUPS}

    flat_shares = {g: shares[g].ravel() for g in GROUPS}
    flat_pred = np.broadcast_to(y_pred, (B, H, N)).ravel()
    flat_true = y_true.ravel()

    rows = []
    for g in GROUPS:
        share = flat_shares[g]
        mean_share = float(np.mean(share))

        # Eq. (7): dominance rate
        # Since sub_category = naics4 proxy, sub_category gets ties (>=) credit
        # and naics4 gets strict dominance (>) credit.
        if g == "sub_category":
            other_max = np.maximum(flat_shares["top_category"], flat_shares["naics2"])
            dominant = share >= other_max - 1e-10
        else:
            other_vals = [flat_shares[g2] for g2 in GROUPS if g2 != g]
            other_max = np.max(other_vals, axis=0)
            dominant = share > other_max
        dominance_rate = float(np.mean(dominant))

        # Eq. (8): dominant-event MAE
        n_dom = int(dominant.sum())
        if n_dom > 0:
            mae_dom = float(np.mean(
                np.abs(flat_true[dominant] - flat_pred[dominant])
            ))
        else:
            mae_dom = np.nan

        rows.append({
            "Semantic attribute": g,
            "Granularity": GRANULARITY[g],
            "Mean attribution share": round(mean_share, 4),
            "Dominance rate": round(dominance_rate, 4),
            "Dominant-event MAE": round(mae_dom, 4) if not np.isnan(mae_dom) else "--",
            "N dominant events": n_dom,
        })
        print(f"  {g:20s}: share={mean_share:.4f}, dom_rate={dominance_rate:.4f}, "
              f"dom_MAE={mae_dom:.4f}, n_dom={n_dom:,}")

    df = pd.DataFrame(rows)

    csv_path = f"{RD}/exp3_semantic_granularity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # LaTeX table
    col_spec = (
        r"{>{\raggedright\arraybackslash}p{2.7cm}}"
        r"{>{\raggedright\arraybackslash}p{2.5cm}}"
        r"{>{\raggedleft\arraybackslash}p{2.1cm}}"
        r"{>{\raggedleft\arraybackslash}p{2.0cm}}"
        r"{>{\raggedleft\arraybackslash}p{2.1cm}}"
        r"{>{\raggedleft\arraybackslash}X}"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Gradient-based comparison of semantic attribute granularity (MTGNN + Identity).}",
        r"\label{tab:semantic_granularity_attribution}",
        r"\begin{tabularx}{\textwidth}{" + col_spec + "}",
        r"\toprule",
        r"\textbf{Semantic attribute} & \textbf{Granularity} & "
        r"\textbf{Mean attribution share} & \textbf{Dominance rate} & "
        r"\textbf{Dominant-event MAE} & \textbf{\# dominant events} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        mae_str = f"{r['Dominant-event MAE']:.4f}" if r['Dominant-event MAE'] != "--" else "--"
        attr_label = r["Semantic attribute"].replace("_", r"\_")
        lines.append(
            f"{attr_label:20s} & {r['Granularity']:24s} & "
            f"{r['Mean attribution share']:.4f} & {r['Dominance rate']:.4f} & "
            f"{mae_str} & {r['N dominant events']:,} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])

    tex_path = f"{RD}/exp3_semantic_granularity.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {tex_path}")
    return df


if __name__ == "__main__":
    build_table()
