"""
exp4_parent.py
==============
Parent-place attribution for embedded POIs.
Uses existing per-POI regime classifications from exp2 (EvolveGCN + Identity).
"""

import sys
sys.path.insert(0, "/root/poi_visits")

import os
import numpy as np
import pandas as pd

RD = "/root/poi_visits/attri_exp/results"
os.makedirs(RD, exist_ok=True)

# ----------------------------------------------------------------
# EMBEDDED POI GROUP DEFINITIONS
# ----------------------------------------------------------------
EMBEDDED_CATEGORY_GROUPS = {
    "Cafes and restaurants inside larger destinations": [
        "Restaurants and Other Eating Places",
        "Special Food Services",
        "Bakeries and Tortilla Manufacturing",
        "Beverage Manufacturing",
        "Drinking Places (Alcoholic Beverages)",
    ],
    "Pharmacies and health-related retail inside medical or commercial places": [
        "Health and Personal Care Stores",
        "Drugs and Druggists' Sundries Merchant Wholesalers",
    ],
    "Retail and service shops inside shopping centers or office buildings": [
        "Grocery Stores",
        "Specialty Food Stores",
        "Beer, Wine, and Liquor Stores",
        "Florists",
        "Clothing Stores",
        "Shoe Stores",
        "Jewelry, Luggage, and Leather Goods Stores",
        "Book Stores and News Dealers",
        "Sporting Goods, Hobby, and Musical Instrument Stores",
        "Home Furnishings Stores",
        "Electronics and Appliance Stores",
        "Used Merchandise Stores",
        "Other Miscellaneous Store Retailers",
        "Furniture Stores",
        "Automotive Parts, Accessories, and Tire Stores",
        "Automotive Equipment Rental and Leasing",
        "Gasoline Stations",
        "Consumer Goods Rental",
        "Building Material and Supplies Dealers",
        "Lawn and Garden Equipment and Supplies Stores",
        "Office Supplies, Stationery, and Gift Stores",
    ],
    "Education-related POIs inside campuses": [
        "Elementary and Secondary Schools",
        "Colleges, Universities, and Professional Schools",
        "Junior Colleges",
        "Technical and Trade Schools",
        "Other Schools and Instruction",
        "Educational Support Services",
    ],
    "Medical and clinic-related POIs inside larger health facilities": [
        "Offices of Physicians",
        "Offices of Dentists",
        "Offices of Other Health Practitioners",
        "General Medical and Surgical Hospitals",
        "Specialty (except Psychiatric and Substance Abuse) Hospitals",
        "Outpatient Care Centers",
        "Other Ambulatory Health Care Services",
        "Medical and Diagnostic Laboratories",
        "Nursing Care Facilities (Skilled Nursing Facilities)",
        "Home Health Care Services",
    ],
}


def build_table():
    # ----------------------------------------------------------------
    # 1. Load per-POI regime assignments (from exp2_egcn_clustering.py)
    # ----------------------------------------------------------------
    poi_regimes = pd.read_csv(f"{RD}/exp2_egcn_poi_regime_coordinates.csv")
    print(f"Loaded {len(poi_regimes)} POI regime assignments")

    # ----------------------------------------------------------------
    # 2. Load location table (9926 rows, same order as NPZ)
    # ----------------------------------------------------------------
    loc = pd.read_csv(
        "/root/poi_visits/data/adjacency/static/location_table_aligned_to_poi_meta.csv"
    )
    # Default to "History-driven" for POIs not in poi_regimes
    ALL_REGIMES = ["History-driven", "Location-driven", "Identity-driven",
                    "Parent-driven", "Metadata-sensitive"]
    reg_map = poi_regimes.set_index("placekey")["regime"]
    regime_series = loc["placekey"].map(reg_map).fillna("History-driven")
    loc["regime"] = regime_series.values

    # ----------------------------------------------------------------
    # 3. Identify embedded POIs (have parent_placekey)
    # ----------------------------------------------------------------
    is_embedded = loc["parent_placekey"].notna().values
    top_cat = loc["top_category"].values
    print(f"Total POIs: {len(loc)}")
    print(f"  with parent_placekey (embedded): {is_embedded.sum()}")
    print(f"  without parent (independent):    {(~is_embedded).sum()}")

    # ----------------------------------------------------------------
    # 4. Map embedded POIs to groups
    # ----------------------------------------------------------------
    N = len(loc)
    group_masks = {}
    for group_name, cats in EMBEDDED_CATEGORY_GROUPS.items():
        mask = np.zeros(N, dtype=bool)
        for cat in cats:
            mask |= (top_cat == cat)
        # Only embedded POIs
        mask = mask & is_embedded
        group_masks[group_name] = mask
        print(f"  {group_name}: {mask.sum()} POIs")

    all_embedded_mask = is_embedded
    group_masks["All embedded POIs"] = all_embedded_mask
    print(f"  All embedded POIs: {all_embedded_mask.sum()}")

    # ----------------------------------------------------------------
    # 5. Compute Parent and Own shares per group
    #    Parent = fraction of POIs in "Parent-driven" regime
    #    Own    = 1 - Parent (all other regimes)
    # ----------------------------------------------------------------
    regime_arr = loc["regime"].values

    results = []
    row_labels = list(EMBEDDED_CATEGORY_GROUPS.keys()) + ["All embedded POIs"]

    for label in row_labels:
        mask = group_masks.get(label)
        if mask is None:
            continue
        n = int(mask.sum())

        if n == 0:
            results.append({
                "Embedded POI group": label, "No. POIs": 0,
                "Parent": np.nan, "Own": np.nan, "Gap": np.nan,
                "MAE": np.nan, "Dominant": "--",
            })
            continue

        # Regime counts within this group
        sub_regimes = regime_arr[mask]
        reg_counts = pd.Series(sub_regimes).value_counts()
        parent_pct = reg_counts.get("Parent-driven", 0) / n * 100.0
        own_pct = 100.0 - parent_pct
        gap_pct = parent_pct - own_pct
        dominant = "Parent" if gap_pct > 0 else "Own"

        results.append({
            "Embedded POI group": label,
            "No. POIs": n,
            "Parent": parent_pct,
            "Own": own_pct,
            "Gap": gap_pct,
            "MAE": np.nan,   # placeholder; filled below if predictions available
            "Dominant": dominant,
        })

        print(f"\n  {label} (n={n})")
        print(f"    Regime breakdown:")
        for reg in ALL_REGIMES:
            cnt = int(reg_counts.get(reg, 0))
            pct = cnt / n * 100.0
            print(f"      {reg:22s}: {cnt:4d} ({pct:5.1f}%)")
        print(f"    Parent: {parent_pct:.1f}%, Own: {own_pct:.1f}%, "
              f"Gap: {gap_pct:+.1f}% ({dominant})")

    # ----------------------------------------------------------------
    # 6. Compute MAE for each group
    # ----------------------------------------------------------------
    y_pred = np.load(
        "/root/poi_visits/outputs/predictions/evolvegcn_static_identity_h32_lr001_pred.npy"
    ).astype(np.float32)
    y_pred = np.clip(y_pred, 0, None)
    d = np.load(
        "/root/poi_visits/data/splits/evolvegcn/poi_evolvegcn_static_identity.npz",
        allow_pickle=True
    )
    y_true = d["y_test"].astype(np.float32)
    print(f"\ny_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")

    for row in results:
        mask = group_masks.get(row["Embedded POI group"])
        if mask is None or row["No. POIs"] == 0:
            continue
        n = int(mask.sum())
        mae = float(np.mean(np.abs(y_true[:, :, mask] - y_pred[:, :, mask])))
        row["MAE"] = mae
        print(f"  MAE for '{row['Embedded POI group']}': {mae:.4f}")

    # ----------------------------------------------------------------
    # 7. Save CSV
    # ----------------------------------------------------------------
    df = pd.DataFrame(results)
    for col in ["Parent", "Own", "Gap"]:
        df[col] = df[col].round(1)
    df["MAE"] = df["MAE"].round(2)
    csv_path = f"{RD}/exp4_parent_place_attribution.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ----------------------------------------------------------------
    # 8. Save LaTeX
    # ----------------------------------------------------------------
    def fmt(v, dec=1):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.{dec}f}"

    def fmtgap(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:+.1f}"

    tex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Parent-place attribution for embedded POIs.}",
        r"\label{tab:parent_place_attribution}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{p{0.46\textwidth} r c c c c c}",
        r"\hline",
        r"Embedded POI group",
        r"& No. POIs",
        r"& Parent",
        r"& Own",
        r"& Gap",
        r"& MAE",
        r"& Dominant \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        tex_lines.append(
            f"{row['Embedded POI group']} & {int(row['No. POIs']):,} & "
            f"{fmt(row['Parent'])} & {fmt(row['Own'])} & {fmtgap(row['Gap'])} & "
            f"{fmt(row['MAE'], 2)} & {row['Dominant']} \\\\"
        )

    tex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\vspace{2pt}",
        r"{\footnotesize",
        r"\emph{Note.} Parent is the average attribution share of the parent-place feature group. ",
        r"Own is the aggregated attribution share of the POI's own feature groups. ",
        r"Gap is computed as Parent minus Own. ",
        r"The dominant source is Parent if the gap is positive, and Own otherwise.",
        r"}",
        r"\end{table*}",
    ])

    tex_path = f"{RD}/exp4_parent_place_attribution.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"Saved: {tex_path}")

    print("\n--- LaTeX table ---")
    print("\n".join(tex_lines))
    return df


if __name__ == "__main__":
    build_table()
