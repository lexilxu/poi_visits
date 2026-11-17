# scripts/create_adjacencies.py

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def find_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(
            f"Could not find any of these columns in location_table.csv: {candidates}"
        )

    return None


def row_normalize(A):
    A = A.astype(np.float32)
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return A / row_sum


def sym_normalize(A):
    A = A.astype(np.float32)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]


def add_self_loops(A):
    A = A.copy().astype(np.float32)
    np.fill_diagonal(A, 1.0)
    return A


def haversine_distance_matrix(lat, lon):
    """
    Returns distance matrix in kilometers.
    """
    R = 6371.0

    lat = np.radians(lat)
    lon = np.radians(lon)

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_identity_adj(N):
    return np.eye(N, dtype=np.float32)


def build_distance_knn_adj(lat, lon, k=10, self_loops=True, normalize="row"):
    """
    k-nearest-neighbor spatial adjacency.
    Edge weight = 1 for k nearest POIs.
    """
    coords = np.column_stack([lat, lon])

    valid = ~np.isnan(coords).any(axis=1)
    N = len(lat)

    A = np.zeros((N, N), dtype=np.float32)

    if valid.sum() <= k:
        print("Warning: not enough valid coordinates for kNN adjacency.")
        return add_self_loops(A)

    coords_valid = coords[valid]
    valid_indices = np.where(valid)[0]

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, coords_valid.shape[0]),
        metric="haversine",
    )

    coords_rad = np.radians(coords_valid)
    nbrs.fit(coords_rad)

    distances, indices = nbrs.kneighbors(coords_rad)

    for row_valid_pos, neigh_valid_positions in enumerate(indices):
        i = valid_indices[row_valid_pos]

        for neigh_valid_pos in neigh_valid_positions:
            j = valid_indices[neigh_valid_pos]

            if i != j:
                A[i, j] = 1.0

    # Make undirected
    A = np.maximum(A, A.T)

    if self_loops:
        A = add_self_loops(A)

    if normalize == "row":
        A = row_normalize(A)
    elif normalize == "sym":
        A = sym_normalize(A)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")

    return A.astype(np.float32)


def build_distance_rbf_adj(lat, lon, radius_km=2.0, sigma_km=1.0, self_loops=True, normalize="row"):
    """
    Distance-thresholded RBF adjacency.

    A_ij = exp(-d_ij^2 / (2 sigma^2)) if d_ij <= radius_km.
    """
    N = len(lat)

    coords = np.column_stack([lat, lon])
    valid = ~np.isnan(coords).any(axis=1)

    A = np.zeros((N, N), dtype=np.float32)

    if valid.sum() <= 1:
        print("Warning: not enough valid coordinates for RBF adjacency.")
        return add_self_loops(A)

    valid_indices = np.where(valid)[0]

    dist = haversine_distance_matrix(lat[valid], lon[valid])
    W = np.exp(-(dist ** 2) / (2 * sigma_km ** 2))
    W[dist > radius_km] = 0.0
    np.fill_diagonal(W, 0.0)

    for a, i in enumerate(valid_indices):
        for b, j in enumerate(valid_indices):
            A[i, j] = W[a, b]

    if self_loops:
        A = add_self_loops(A)

    if normalize == "row":
        A = row_normalize(A)
    elif normalize == "sym":
        A = sym_normalize(A)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")

    return A.astype(np.float32)


def build_category_adj(values, self_loops=True, normalize="row"):
    """
    Connect POIs sharing the same category.
    """
    values = pd.Series(values).astype("string")
    N = len(values)

    A = np.zeros((N, N), dtype=np.float32)

    for cat, idx in values.groupby(values).groups.items():
        if pd.isna(cat):
            continue

        idx = list(idx)
        if len(idx) <= 1:
            continue

        A[np.ix_(idx, idx)] = 1.0

    np.fill_diagonal(A, 0.0)

    if self_loops:
        A = add_self_loops(A)

    if normalize == "row":
        A = row_normalize(A)
    elif normalize == "sym":
        A = sym_normalize(A)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")

    return A.astype(np.float32)


def clean_naics_code(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    s = "".join(ch for ch in s if ch.isdigit())

    if len(s) == 0:
        return None

    return s


def build_naics_adj(naics_values, prefix_len=2, self_loops=True, normalize="row"):
    cleaned = []

    for x in naics_values:
        s = clean_naics_code(x)
        if s is None or len(s) < prefix_len:
            cleaned.append(pd.NA)
        else:
            cleaned.append(s[:prefix_len])

    return build_category_adj(
        cleaned,
        self_loops=self_loops,
        normalize=normalize,
    )


def build_visit_correlation_adj(visits, threshold=0.3, self_loops=True, normalize="row"):
    """
    Connect POIs with absolute visit correlation above threshold.
    """
    corr = np.corrcoef(visits.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    A = np.abs(corr).astype(np.float32)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)

    if self_loops:
        A = add_self_loops(A)

    if normalize == "row":
        A = row_normalize(A)
    elif normalize == "sym":
        A = sym_normalize(A)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")

    return A.astype(np.float32)


def print_adj_summary(name, A):
    density = np.count_nonzero(A) / A.size
    print(
        f"{name:25s} shape={A.shape}, "
        f"nonzero={np.count_nonzero(A)}, "
        f"density={density:.6f}, "
        f"min={A.min():.6f}, max={A.max():.6f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--poi_meta", type=str, default="data/processed/poi_meta.csv")
    parser.add_argument("--location_table", type=str, default="data/raw/location_table.csv")
    parser.add_argument("--visits_path", type=str, default="data/processed/visits.npy")
    parser.add_argument("--output_dir", type=str, default="data/adjacency")

    parser.add_argument("--placekey_col", type=str, default="placekey")

    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--rbf_radius_km", type=float, default=2.0)
    parser.add_argument("--rbf_sigma_km", type=float, default=1.0)
    parser.add_argument("--corr_threshold", type=float, default=0.3)

    parser.add_argument(
        "--normalize",
        type=str,
        default="row",
        choices=["row", "sym", "none"],
    )

    parser.add_argument(
        "--self_loops",
        action="store_true",
        help="Add self loops. Recommended for GNNs.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    poi_meta = pd.read_csv(args.poi_meta)
    loc = pd.read_csv(args.location_table)

    if args.placekey_col not in poi_meta.columns:
        raise ValueError(f"{args.placekey_col} not found in poi_meta.csv")

    if args.placekey_col not in loc.columns:
        raise ValueError(f"{args.placekey_col} not found in location_table.csv")

    # Keep exact POI order from poi_meta.
    poi_meta[args.placekey_col] = poi_meta[args.placekey_col].astype(str)
    loc[args.placekey_col] = loc[args.placekey_col].astype(str)

    loc = loc.drop_duplicates(subset=[args.placekey_col])

    merged = poi_meta[[args.placekey_col]].merge(
        loc,
        on=args.placekey_col,
        how="left",
    )

    N = merged.shape[0]

    lat_col = find_column(
        merged,
        ["latitude", "lat", "LATITUDE", "Latitude"],
        required=True,
    )

    lon_col = find_column(
        merged,
        ["longitude", "lng", "lon", "long", "LONGITUDE", "Longitude"],
        required=True,
    )

    top_cat_col = find_column(
        merged,
        ["top_category", "top_category_name", "category"],
        required=False,
    )

    sub_cat_col = find_column(
        merged,
        ["sub_category", "subcategory", "sub_category_name"],
        required=False,
    )

    naics_col = find_column(
        merged,
        ["naics_code", "naics", "NAICS"],
        required=False,
    )

    lat = pd.to_numeric(merged[lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(merged[lon_col], errors="coerce").to_numpy()

    visits = np.load(args.visits_path)

    if visits.shape[1] != N:
        raise ValueError(
            f"visits has {visits.shape[1]} nodes, but merged location table has {N}"
        )

    missing_location = merged[lat_col].isna().sum()
    print(f"Number of POIs: {N}")
    print(f"Missing latitude values: {missing_location}")

    adjacencies = {}

    adjacencies["adj_identity"] = build_identity_adj(N)

    adjacencies["adj_distance_knn"] = build_distance_knn_adj(
        lat=lat,
        lon=lon,
        k=args.knn_k,
        self_loops=args.self_loops,
        normalize=args.normalize,
    )

    adjacencies["adj_distance_rbf"] = build_distance_rbf_adj(
        lat=lat,
        lon=lon,
        radius_km=args.rbf_radius_km,
        sigma_km=args.rbf_sigma_km,
        self_loops=args.self_loops,
        normalize=args.normalize,
    )

    if top_cat_col is not None:
        adjacencies["adj_top_category"] = build_category_adj(
            merged[top_cat_col],
            self_loops=args.self_loops,
            normalize=args.normalize,
        )
    else:
        print("Skipping top-category adjacency: column not found.")

    if sub_cat_col is not None:
        adjacencies["adj_sub_category"] = build_category_adj(
            merged[sub_cat_col],
            self_loops=args.self_loops,
            normalize=args.normalize,
        )
    else:
        print("Skipping sub-category adjacency: column not found.")

    if naics_col is not None:
        adjacencies["adj_naics2"] = build_naics_adj(
            merged[naics_col],
            prefix_len=2,
            self_loops=args.self_loops,
            normalize=args.normalize,
        )

        adjacencies["adj_naics4"] = build_naics_adj(
            merged[naics_col],
            prefix_len=4,
            self_loops=args.self_loops,
            normalize=args.normalize,
        )
    else:
        print("Skipping NAICS adjacency: column not found.")

    adjacencies["adj_visit_correlation"] = build_visit_correlation_adj(
        visits,
        threshold=args.corr_threshold,
        self_loops=args.self_loops,
        normalize=args.normalize,
    )

    for name, A in adjacencies.items():
        path = os.path.join(args.output_dir, f"{name}.npy")
        np.save(path, A.astype(np.float32))
        print_adj_summary(name, A)

    # Save merged aligned table for auditing.
    aligned_path = os.path.join(args.output_dir, "location_table_aligned_to_poi_meta.csv")
    merged.to_csv(aligned_path, index=False)
    print(f"Saved aligned location table: {aligned_path}")


if __name__ == "__main__":
    main()