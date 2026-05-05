# src/build_static_features.py
# Build static node features: 4 numeric (lat, lon, open_hours, miss_open)
# + 5 categorical IDs (cat, naics2, naics4, brand, parent) -> embedding in models

import os
import ast
import argparse
import numpy as np
import pandas as pd

CAT_EMBED_DIM = 8
NAICS2_EMBED_DIM = 8
NAICS4_EMBED_DIM = 8
BRAND_EMBED_DIM = 8
PARENT_EMBED_DIM = 8

CAT_FREQ_THRESH = 1
NAICS2_FREQ_THRESH = 1
NAICS4_FREQ_THRESH = 1
BRAND_FREQ_THRESH = 20
PARENT_FREQ_THRESH = 20


def clean_naics(x):
    if pd.isna(x):
        return None
    s = str(x).strip().rstrip("0").rstrip(".")
    s = "".join(c for c in s if c.isdigit())
    return s if len(s) > 0 else None


def frequency_group(values, freq_threshold, unknown_token="__unknown__", other_token="__other__"):
    series = pd.Series(values)
    freq = series.value_counts()
    keep_set = set(freq[freq > freq_threshold].index)

    mapped = pd.Series(values).astype(str)
    for i in range(len(mapped)):
        v = mapped.iloc[i]
        if v == "nan" or v == "" or pd.isna(values.iloc[i]):
            mapped.iloc[i] = unknown_token
        elif v not in keep_set:
            mapped.iloc[i] = other_token

    unique_vals = sorted(mapped.unique().tolist())
    name_to_id = {v: i for i, v in enumerate(unique_vals)}
    ids = np.array([name_to_id[str(v)] for v in mapped], dtype=np.int64)
    return ids, unique_vals


def parse_open_hours(s):
    if pd.isna(s) or str(s).strip() == "" or str(s) == "nan":
        return None
    try:
        obj = ast.literal_eval(str(s))
        total = 0.0
        for day, intervals in obj.items():
            for interval in intervals:
                if isinstance(interval, list) and len(interval) == 2:
                    start, end = interval
                    try:
                        sh, sm = map(int, start.split(":"))
                        eh, em = map(int, end.split(":"))
                        total += (eh + em / 60) - (sh + sm / 60)
                    except Exception:
                        pass
        return total if total > 0 else None
    except Exception:
        return None


def build_static_features(poi_meta, location_table, location_table_path=None):
    merged = poi_meta.copy()

    if location_table_path is not None:
        lt = pd.read_csv(location_table_path)
        if "placekey" in merged.columns and "placekey" in lt.columns:
            lt = lt.drop_duplicates(subset=["placekey"])
            merged = merged.merge(lt, on="placekey", how="left")

    lat_col = next((c for c in ["latitude", "lat"] if c in merged.columns), None)
    lon_col = next((c for c in ["longitude", "lng", "lon", "long"] if c in merged.columns), None)
    cat_col = next((c for c in ["top_category", "top_category_name", "category"] if c in merged.columns), None)
    naics_col = next((c for c in ["naics_code", "naics"] if c in merged.columns), None)
    brand_col = next((c for c in ["brands", "brand", "brand_name"] if c in merged.columns), None)
    parent_col = next((c for c in ["parent_placekey", "parent"] if c in merged.columns), None)
    open_col = next((c for c in ["open_hours"] if c in merged.columns), None)

    feats = []
    feat_names = []

    if lat_col is not None:
        lat_vals = pd.to_numeric(merged[lat_col], errors="coerce").fillna(0).to_numpy()
        feats.append(lat_vals[:, None])
        feat_names.append("lat")

    if lon_col is not None:
        lon_vals = pd.to_numeric(merged[lon_col], errors="coerce").fillna(0).to_numpy()
        feats.append(lon_vals[:, None])
        feat_names.append("lon")

    if open_col is not None:
        open_vals = merged[open_col].apply(parse_open_hours)
        miss_mask = open_vals.isna().to_numpy().astype(np.float32)
        open_filled = open_vals.fillna(0).to_numpy()
        feats.append(open_filled[:, None])
        feats.append(miss_mask[:, None])
        feat_names += ["open_hours", "miss_open"]
    else:
        feats.append(np.zeros((len(merged), 1), dtype=np.float32))
        feats.append(np.zeros((len(merged), 1), dtype=np.float32))
        feat_names += ["open_hours", "miss_open"]

    S_cont = np.concatenate(feats, axis=1).astype(np.float32)

    cat_ids, cat_vocab = None, []
    if cat_col is not None:
        cat_ids, cat_vocab = frequency_group(
            merged[cat_col], CAT_FREQ_THRESH, "__unknown__", "__other__"
        )

    naics2_ids, naics2_vocab = None, []
    naics4_ids, naics4_vocab = None, []
    if naics_col is not None:
        naics_cleaned = merged[naics_col].apply(clean_naics)
        naics2_vals = naics_cleaned.apply(
            lambda x: x[:2] if x and len(x) >= 2 else "__unk__"
        )
        naics2_ids, naics2_vocab = frequency_group(
            naics2_vals, NAICS2_FREQ_THRESH, "__unk__", "__other__"
        )
        naics4_vals = naics_cleaned.apply(
            lambda x: x[:4] if x and len(x) >= 4 else "__unk__"
        )
        naics4_ids, naics4_vocab = frequency_group(
            naics4_vals, NAICS4_FREQ_THRESH, "__unk__", "__other__"
        )

    brand_ids, brand_vocab = None, []
    if brand_col is not None:
        brand_ids, brand_vocab = frequency_group(
            merged[brand_col], BRAND_FREQ_THRESH, "__unknown__", "__other__"
        )

    parent_ids, parent_vocab = None, []
    if parent_col is not None:
        parent_ids, parent_vocab = frequency_group(
            merged[parent_col], PARENT_FREQ_THRESH, "__unknown__", "__other__"
        )

    feat_info = {
        "D_s_continuous": int(S_cont.shape[1]),
        "N": int(S_cont.shape[0]),
        "feature_names": feat_names,
        "cat_vocab": cat_vocab,
        "cat_emb_dim": CAT_EMBED_DIM,
        "naics2_vocab": naics2_vocab,
        "naics2_emb_dim": NAICS2_EMBED_DIM,
        "naics4_vocab": naics4_vocab,
        "naics4_emb_dim": NAICS4_EMBED_DIM,
        "brand_vocab": brand_vocab,
        "brand_emb_dim": BRAND_EMBED_DIM,
        "brand_freq_threshold": BRAND_FREQ_THRESH,
        "parent_vocab": parent_vocab,
        "parent_emb_dim": PARENT_EMBED_DIM,
        "parent_freq_threshold": PARENT_FREQ_THRESH,
    }

    return S_cont, feat_info, cat_ids, naics2_ids, naics4_ids, brand_ids, parent_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poi_meta", type=str, default="data/processed/poi_meta.csv")
    parser.add_argument("--location_table", type=str, default="data/raw/location_table.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    poi_meta = pd.read_csv(args.poi_meta)
    S, feat_info, cat_ids, naics2_ids, naics4_ids, brand_ids, parent_ids = build_static_features(
        poi_meta, None, args.location_table
    )

    np.save(os.path.join(args.output_dir, "static_node_features.npy"), S)
    np.save(os.path.join(args.output_dir, "static_cat_ids.npy"), cat_ids)
    np.save(os.path.join(args.output_dir, "static_naics2_ids.npy"), naics2_ids)
    np.save(os.path.join(args.output_dir, "static_naics4_ids.npy"), naics4_ids)
    np.save(os.path.join(args.output_dir, "static_brand_ids.npy"), brand_ids)
    np.save(os.path.join(args.output_dir, "static_parent_ids.npy"), parent_ids)

    import json
    info_path = os.path.join(args.output_dir, "static_node_features_info.json")
    with open(info_path, "w") as f:
        json.dump(feat_info, f, indent=2)

    total_emb = CAT_EMBED_DIM + NAICS2_EMBED_DIM + NAICS4_EMBED_DIM + BRAND_EMBED_DIM + PARENT_EMBED_DIM
    print(f"Static continuous: shape={S.shape}")
    print(f"  cat: {len(feat_info['cat_vocab'])} classes, emb_dim={CAT_EMBED_DIM}")
    print(f"  naics2: {len(feat_info['naics2_vocab'])} classes, emb_dim={NAICS2_EMBED_DIM}")
    print(f"  naics4: {len(feat_info['naics4_vocab'])} classes, emb_dim={NAICS4_EMBED_DIM}")
    print(f"  brand: {len(feat_info['brand_vocab'])} classes, emb_dim={BRAND_EMBED_DIM}")
    print(f"  parent: {len(feat_info['parent_vocab'])} classes, emb_dim={PARENT_EMBED_DIM}")
    print(f"  Total emb dim: {total_emb}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
