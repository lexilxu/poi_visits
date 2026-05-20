#!/usr/bin/env python3
"""
Export EvolveGCN splits with the SAME static feature format as MTGNN.

MTGNN x format: (B, 10, N, T) where channels are:
  C=0:  visit signal
  C=1:  lat (continuous)
  C=2:  lon (continuous)
  C=3:  open_hours (continuous)
  C=4:  miss_open (continuous)
  C=5:  cat category id
  C=6:  naics2 category id
  C=7:  naics4 category id
  C=8:  brand category id
  C=9:  parent category id

Continuous features are normalized with train-set statistics.
All channels are broadcast to (B, C, N, T) to match MTGNN's convention.
"""

import os
import argparse
import numpy as np


def save_evolvegcn_file(path, X, Y, S_cont, cat_ids, naics2_ids, naics4_ids,
                         brand_ids, parent_ids, seq_in_len, seq_out_len,
                         scaler_mean=None, scaler_std=None):
    """
    X: (B, P, N) raw visit counts
    Y: (B, H, N) labels
    S_cont: (N, 4) continuous static features
    cat_ids: (N,) category id
    naics2_ids: (N,) naics2 id
    naics4_ids: (N,) naics4 id
    brand_ids: (N,) brand id
    parent_ids: (N,) parent id
    """
    B, P, N = X.shape

    # C=0: visit signal — same as MTGNN convention
    visit = X.transpose(0, 2, 1).reshape(B, 1, N, P).astype(np.float32)

    # C=1-4: continuous static features — normalized per-channel, broadcast
    S_norm = S_cont.copy()
    if S_norm.shape[1] > 0:
        s_mean = S_norm.mean(axis=0, keepdims=True)
        s_std = S_norm.std(axis=0, keepdims=True)
        s_std[s_std == 0] = 1.0
        S_norm = (S_norm - s_mean) / s_std
    S_expanded = np.broadcast_to(S_norm.reshape(1, S_norm.shape[1], N, 1),
                                  (B, S_norm.shape[1], N, P)).astype(np.float32)

    # C=5-9: categorical ids — broadcast to (B, C, N, T), keep as int64
    def add_cat(ids):
        if ids is not None:
            return np.broadcast_to(ids.reshape(1, 1, N, 1), (B, 1, N, P)).astype(np.int64)
        return None

    cat_expanded = add_cat(cat_ids)
    naics2_expanded = add_cat(naics2_ids)
    naics4_expanded = add_cat(naics4_ids)
    brand_expanded = add_cat(brand_ids)
    parent_expanded = add_cat(parent_ids)

    parts = [visit, S_expanded]
    for exp in [cat_expanded, naics2_expanded, naics4_expanded, brand_expanded, parent_expanded]:
        if exp is not None:
            parts.append(exp)

    x = np.concatenate(parts, axis=1).astype(np.float32)
    y = Y.astype(np.float32)

    out = dict(
        x=x,          # (B, 10, N, T)
        y=y,          # (B, H, N)
        adjacency_mode="static",
    )
    if scaler_mean is not None:
        out["scaler_mean"] = scaler_mean
        out["scaler_std"] = scaler_std

    np.savez_compressed(path, **out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_path", type=str,
                        default="data/splits/all_samples.npz")
    parser.add_argument("--indices_path", type=str,
                        default="data/splits/split_indices.npz")
    parser.add_argument("--output_dir", type=str,
                        default="data/splits/evolvegcn")
    parser.add_argument("--adj_path", type=str,
                        default="data/adjacency/static/adj_identity.npy")
    parser.add_argument("--static_features_path", type=str,
                        default="data/splits/static_node_features.npy")
    parser.add_argument("--cat_ids_path", type=str,
                        default="data/splits/static_cat_ids.npy")
    parser.add_argument("--naics2_ids_path", type=str,
                        default="data/splits/static_naics2_ids.npy")
    parser.add_argument("--naics4_ids_path", type=str,
                        default="data/splits/static_naics4_ids.npy")
    parser.add_argument("--brand_ids_path", type=str,
                        default="data/splits/static_brand_ids.npy")
    parser.add_argument("--parent_ids_path", type=str,
                        default="data/splits/static_parent_ids.npy")
    parser.add_argument("--seq_in_len", type=int, default=14)
    parser.add_argument("--seq_out_len", type=int, default=7)
    parser.add_argument("--output_name", type=str,
                        default="poi_evolvegcn_static_attrs")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = np.load(args.samples_path)
    indices = np.load(args.indices_path)

    X = samples["X"].astype(np.float32)   # (B, P, N)
    Y = samples["Y"].astype(np.float32)   # (B, H, N)

    train_idx = indices["train_idx"]
    val_idx = indices["val_idx"]
    test_idx = indices["test_idx"]

    N = X.shape[2]

    # Load static features (same as MTGNN)
    def load_optional(path, name):
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape[0] != N:
                raise ValueError(f"{name} nodes {arr.shape[0]} != N={N}")
            print(f"  {name}: shape={arr.shape}")
            return arr
        print(f"  {name}: NOT FOUND")
        return None

    print("Loading static features:")
    S_cont = load_optional(args.static_features_path, "S_cont (continuous)")
    cat_ids = load_optional(args.cat_ids_path, "cat_ids")
    naics2_ids = load_optional(args.naics2_ids_path, "naics2_ids")
    naics4_ids = load_optional(args.naics4_ids_path, "naics4_ids")
    brand_ids = load_optional(args.brand_ids_path, "brand_ids")
    parent_ids = load_optional(args.parent_ids_path, "parent_ids")

    # Compute visit normalization stats from train set
    scaler_mean = float(X[train_idx].mean())
    scaler_std = float(X[train_idx].std())
    print(f"\nVisit scaler: mean={scaler_mean:.4f}, std={scaler_std:.4f}")

    # Load adjacency
    adj = np.load(args.adj_path).astype(np.float32)
    print(f"Adjacency: shape={adj.shape}")
    if adj.shape != (N, N):
        raise ValueError(f"Adj shape {adj.shape} != (N,N)=({N},{N})")

    # Save one .npz per split (train, val, test)
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        out_path = os.path.join(args.output_dir, f"{args.output_name}_{split_name}.npz")

        # x for this split: (B, 10, N, T)
        save_evolvegcn_file(
            path=out_path,
            X=X[idx],
            Y=Y[idx],
            S_cont=S_cont,
            cat_ids=cat_ids,
            naics2_ids=naics2_ids,
            naics4_ids=naics4_ids,
            brand_ids=brand_ids,
            parent_ids=parent_ids,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        # Append adjacency separately (not in x, used by the model)
        d = np.load(out_path)
        out = dict(d)
        out["adj"] = adj
        np.savez_compressed(out_path, **out)
        print(f"Saved {split_name}: {out_path}  x.shape={d['x'].shape}")

    print(f"\nAll splits saved to: {args.output_dir}")
    print(f"  x shape: (B, 10={1+4+5}, N, T)")
    print(f"  channels: [visit, lat, lon, open_hours, miss_open, cat, naics2, naics4, brand, parent]")


if __name__ == "__main__":
    main()
