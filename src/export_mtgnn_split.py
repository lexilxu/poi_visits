# scripts/export_mtgnn_split.py

import os
import json
import argparse
import numpy as np


def save_mtgnn_file(path, X, Y, S_cont, cat_ids, naics2_ids, naics4_ids, brand_ids, parent_ids, seq_in_len, seq_out_len, scaler_mean=None, scaler_std=None):
    P = X.shape[1]
    B = X.shape[0]
    N = X.shape[2]

    visit = X.transpose(0, 2, 1).reshape(B, 1, N, P).astype(np.float32)

    S_norm = S_cont.copy()
    if S_norm.shape[1] > 0:
        s_mean = S_norm.mean(axis=0, keepdims=True)
        s_std = S_norm.std(axis=0, keepdims=True)
        s_std[s_std == 0] = 1.0
        S_norm = (S_norm - s_mean) / s_std
    S_expanded = np.broadcast_to(S_norm.reshape(1, S_norm.shape[1], N, 1), (B, S_norm.shape[1], N, P))

    parts = [visit, S_expanded]

    def add_id(ids, name):
        if ids is not None:
            exp = np.broadcast_to(ids.reshape(1, 1, N, 1), (B, 1, N, P)).astype(np.int64)
            parts.append(exp)

    add_id(cat_ids, "cat")
    add_id(naics2_ids, "naics2")
    add_id(naics4_ids, "naics4")
    add_id(brand_ids, "brand")
    add_id(parent_ids, "parent")

    x = np.concatenate(parts, axis=1).astype(np.float32)
    y = Y.astype(np.float32)

    x_offsets = np.arange(-seq_in_len + 1, 1).reshape(-1, 1)
    y_offsets = np.arange(1, seq_out_len + 1).reshape(-1, 1)

    out = dict(x=x, y=y, x_offsets=x_offsets, y_offsets=y_offsets)
    if scaler_mean is not None:
        out["scaler_mean"] = scaler_mean
        out["scaler_std"] = scaler_std
    np.savez_compressed(path, **out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_path", type=str, default="data/splits/all_samples.npz")
    parser.add_argument("--indices_path", type=str, default="data/splits/split_indices.npz")
    parser.add_argument("--output_dir", type=str, default="data/splits/mtgnn")
    parser.add_argument("--static_features_path", type=str, default="data/splits/static_node_features.npy")
    parser.add_argument("--cat_ids_path", type=str, default="data/splits/static_cat_ids.npy")
    parser.add_argument("--naics2_ids_path", type=str, default="data/splits/static_naics2_ids.npy")
    parser.add_argument("--naics4_ids_path", type=str, default="data/splits/static_naics4_ids.npy")
    parser.add_argument("--brand_ids_path", type=str, default="data/splits/static_brand_ids.npy")
    parser.add_argument("--parent_ids_path", type=str, default="data/splits/static_parent_ids.npy")
    parser.add_argument("--seq_in_len", type=int, default=14)
    parser.add_argument("--seq_out_len", type=int, default=7)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = np.load(args.samples_path)
    indices = np.load(args.indices_path)

    X = samples["X"]
    Y = samples["Y"]

    if os.path.exists(args.static_features_path):
        S = np.load(args.static_features_path).astype(np.float32)
        print(f"Loaded S: {S.shape}")
    else:
        S = np.zeros((X.shape[2], 0), dtype=np.float32)

    def load_id(path, name):
        if os.path.exists(path):
            ids = np.load(path)
            print(f"Loaded {name}: {ids.shape}, vocab_size={int(ids.max())+1}")
            return ids
        print(f"{name}: not found")
        return None

    cat_ids = load_id(args.cat_ids_path, "cat_ids")
    naics2_ids = load_id(args.naics2_ids_path, "naics2_ids")
    naics4_ids = load_id(args.naics4_ids_path, "naics4_ids")
    brand_ids = load_id(args.brand_ids_path, "brand_ids")
    parent_ids = load_id(args.parent_ids_path, "parent_ids")

    N = X.shape[2]
    for name, ids in [("S", S), ("cat", cat_ids), ("naics2", naics2_ids),
                       ("naics4", naics4_ids), ("brand", brand_ids), ("parent", parent_ids)]:
        if ids is not None and ids.shape[0] != N:
            raise ValueError(f"{name} nodes {ids.shape[0]} != num_nodes {N}")

    train_idx = indices["train_idx"]
    val_idx = indices["val_idx"]
    test_idx = indices["test_idx"]

    scaler_mean = float(X[train_idx].mean())
    scaler_std = float(X[train_idx].std())

    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        save_mtgnn_file(
            os.path.join(args.output_dir, f"{split_name}.npz"),
            X[idx], Y[idx], S,
            cat_ids, naics2_ids, naics4_ids, brand_ids, parent_ids,
            args.seq_in_len, args.seq_out_len,
            scaler_mean, scaler_std,
        )

    print("Saved MTGNN splits to:", args.output_dir)
    info_path = os.path.join(os.path.dirname(args.output_dir.rstrip("/\\")), "static_node_features_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)

    D_s = S.shape[1]
    emb = 5 * 8
    print(f"in_dim = 1(visit) + {D_s}(cont) + {emb}(5 categorical emb) = {1 + D_s + emb}")


if __name__ == "__main__":
    main()
