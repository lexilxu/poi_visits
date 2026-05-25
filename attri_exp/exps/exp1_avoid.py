import sys
sys.path.insert(0, "/root/poi_visits")
import numpy as np
import pandas as pd
from src.evaluate_all import mae, rmse


def build_table():
    X = np.load("/root/poi_visits/data/splits/all_samples.npz")["X"]
    Y = np.load("/root/poi_visits/data/splits/all_samples.npz")["Y"]
    idx = np.load("/root/poi_visits/data/splits/split_indices.npz")

    X_tr = X[idx["train_idx"]]
    y_hist = X_tr.mean(axis=(0, 1))
    zero_ratio = np.mean(X_tr == 0, axis=(0, 1))
    pos_avg = y_hist[y_hist > 0]
    Q_alpha = float(np.quantile(pos_avg, 0.25))
    eta_z = 0.7

    I0_mask = (zero_ratio >= eta_z) & (y_hist > 0) & (y_hist <= Q_alpha)
    I0 = np.where(I0_mask)[0]

    y_true_test = Y[idx["test_idx"]]
    y_hist_pred = np.broadcast_to(
        y_hist[np.newaxis, np.newaxis, :], y_true_test.shape
    ).copy()

    print(f"I_0 count: {len(I0)} / {len(y_hist)}  (eta_z={eta_z}, Q_alpha={Q_alpha:.4f})")

    pd_dir = "/root/poi_visits/outputs/predictions"

    models = [
        ("Historical average", "--",  None),
        ("AR-ridge",           "--",  "ar_ridge_pred.npy"),
        ("LSTM",               "--",  "lstm_pred.npy"),
        ("MTGNN",  "Identity",          "mtgnn_identity_lr0005_do3_bs8_pred.npy"),
        ("MTGNN",  "NAICS2",            "mtgnn_naics2_lr001_do3_bs8_pred.npy"),
        ("MTGNN",  "Top category",      "mtgnn_top_category_lr001_do1_bs8_pred.npy"),
        ("MTGNN",  "Distance KNN",      "mtgnn_distance_knn_lr001_do1_bs16_pred.npy"),
        ("EvolveGCN", "Identity",       "evolvegcn_static_identity_h32_lr001_pred.npy"),
        ("EvolveGCN", "NAICS2",        "evolvegcn_static_naics2_h32_lr0005_pred.npy"),
        ("EvolveGCN", "Top category",   "evolvegcn_static_top_category_h32_lr0005_pred.npy"),
        ("EvolveGCN", "Distance RBF",   "evolvegcn_static_distance_rbf_h32_lr0005_pred.npy"),
        ("EvolveGCN", "Visit correlation","evolvegcn_static_visit_correlation_h32_lr0005_pred.npy"),
        ("EvolveGCN", "Distance KNN",  "evolvegcn_static_distance_knn_h32_lr001_pred.npy"),
    ]

    rows = []
    for name, adj, pred_file in models:
        if pred_file is None:
            yp = y_hist_pred.astype(np.float32)
        else:
            yp = np.load(f"{pd_dir}/{pred_file}").astype(np.float32)
        yp = np.clip(yp, 0, None)

        mae_all = mae(y_true_test, yp)
        rmse_all = rmse(y_true_test, yp)
        lift = float(np.mean(yp[:, :, I0] - y_hist_pred[:, :, I0]))

        pos_mask = y_true_test[:, :, I0] > 0
        zero_mask = y_true_test[:, :, I0] == 0

        dmr_m = (
            float(np.sum(yp[:, :, I0][pos_mask]) / np.sum(y_true_test[:, :, I0][pos_mask]))
            if pos_mask.sum() > 0 else 0.0
        )
        dmr_ha = (
            float(np.sum(y_hist_pred[:, :, I0][pos_mask]) / np.sum(y_true_test[:, :, I0][pos_mask]))
            if pos_mask.sum() > 0 else 0.0
        )
        delta_dmr = dmr_m - dmr_ha

        far = (
            float(np.mean(yp[:, :, I0][zero_mask] >= 0.5))
            if zero_mask.sum() > 0 else None
        )
        mean_pred = float(np.mean(yp[:, :, I0]))

        rows.append({
            "Model": name,
            "Adjacency": adj,
            "MAE": round(mae_all, 2),
            "RMSE": round(rmse_all, 2),
            "Lift": round(lift, 2),
            "DMR_model": round(dmr_m, 2),
            "DMR_HA": round(dmr_ha, 2),
            "Delta_DMR": round(delta_dmr, 2),
            "FAR": far,
            "Mean_pred": round(mean_pred, 2),
        })

    df = pd.DataFrame(rows)

    csv_path = "/root/poi_visits/attri_exp/results/exp1_avoid_zero_demand_collapse.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zero-demand collapse diagnostics for low-history POIs in the $P=28$, $H=14$ forecasting task.}",
        r"\label{tab:zero_demand_collapse}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Adjacency & MAE & RMSE & Lift & $\mathrm{DMR}^{\mathrm{model}}$ & $\mathrm{DMR}^{\mathrm{HA}}$ & $\Delta \mathrm{DMR}$ & $\mathrm{FAR}_{0.5}$ & Mean pred. \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        model = r["Model"]
        adj = r["Adjacency"]
        far_str = "--" if pd.isna(r["FAR"]) else f"{r['FAR']:.2f}"
        tex_lines.append(
            f"{model:20s} & {adj:20s} & {r['MAE']:.2f} & {r['RMSE']:.2f} & "
            f"{r['Lift']:.2f} & {r['DMR_model']:.2f} & {r['DMR_HA']:.2f} & "
            f"{r['Delta_DMR']:.2f} & {far_str} & {r['Mean_pred']:.2f} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    tex_path = "/root/poi_visits/attri_exp/results/exp1_avoid_zero_demand_collapse.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"Saved: {tex_path}\n")
    print("\n".join(tex_lines))


if __name__ == "__main__":
    build_table()
