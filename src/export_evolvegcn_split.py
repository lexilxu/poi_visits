# scripts/export_evolvegcn_split.py

import os
import argparse
import numpy as np


def export_static(args, X, Y, train_idx, val_idx, test_idx, S=None):
    adj = np.load(args.adj_path).astype(np.float32)
    N = X.shape[2]

    if adj.shape != (N, N):
        raise ValueError(f"Static adjacency shape {adj.shape} does not match N={N}")

    out = dict(
        adjacency_mode="static",
        x_train=X[train_idx],
        y_train=Y[train_idx],
        x_val=X[val_idx],
        y_val=Y[val_idx],
        x_test=X[test_idx],
        y_test=Y[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        adj=adj,
    )
    if S is not None:
        s_mean = S.mean(axis=0, keepdims=True)
        s_std = S.std(axis=0, keepdims=True)
        s_std[s_std == 0] = 1.0
        S_norm = (S - s_mean) / s_std
        out["static_node_features"] = S_norm

    np.savez_compressed(args.output_path, **out)


def subset_dynamic_edges(dynamic_adj, sample_indices, seq_in_len):
    """
    dynamic_adj stores graphs for all samples.

    Graph index:
        graph_id = sample_id * seq_in_len + p
    """
    edge_src = dynamic_adj["edge_src"]
    edge_dst = dynamic_adj["edge_dst"]
    edge_weight = dynamic_adj["edge_weight"]
    edge_ptr = dynamic_adj["edge_ptr"]

    new_src = []
    new_dst = []
    new_weight = []
    new_ptr = [0]

    for sample_id in sample_indices:
        for p in range(seq_in_len):
            graph_id = int(sample_id) * seq_in_len + p
            start = edge_ptr[graph_id]
            end = edge_ptr[graph_id + 1]

            src = edge_src[start:end]
            dst = edge_dst[start:end]
            weight = edge_weight[start:end]

            new_src.append(src)
            new_dst.append(dst)
            new_weight.append(weight)

            new_ptr.append(new_ptr[-1] + len(src))

    return (
        np.concatenate(new_src),
        np.concatenate(new_dst),
        np.concatenate(new_weight),
        np.array(new_ptr, dtype=np.int64),
    )


def export_dynamic(args, X, Y, train_idx, val_idx, test_idx, S=None):
    dynamic_adj = np.load(args.dynamic_adj_path)

    seq_in_len = X.shape[1]
    N = X.shape[2]

    dyn_P = int(dynamic_adj["seq_in_len"])
    dyn_N = int(dynamic_adj["num_nodes"])

    if dyn_P != seq_in_len:
        raise ValueError(f"Dynamic adj seq_in_len {dyn_P} does not match X seq_in_len {seq_in_len}")

    if dyn_N != N:
        raise ValueError(f"Dynamic adj num_nodes {dyn_N} does not match X num_nodes {N}")

    train_src, train_dst, train_weight, train_ptr = subset_dynamic_edges(
        dynamic_adj, train_idx, seq_in_len
    )

    val_src, val_dst, val_weight, val_ptr = subset_dynamic_edges(
        dynamic_adj, val_idx, seq_in_len
    )

    test_src, test_dst, test_weight, test_ptr = subset_dynamic_edges(
        dynamic_adj, test_idx, seq_in_len
    )

    out = dict(
        adjacency_mode="dynamic",
        x_train=X[train_idx],
        y_train=Y[train_idx],
        x_val=X[val_idx],
        y_val=Y[val_idx],
        x_test=X[test_idx],
        y_test=Y[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_edge_src=train_src,
        train_edge_dst=train_dst,
        train_edge_weight=train_weight,
        train_edge_ptr=train_ptr,
        val_edge_src=val_src,
        val_edge_dst=val_dst,
        val_edge_weight=val_weight,
        val_edge_ptr=val_ptr,
        test_edge_src=test_src,
        test_edge_dst=test_dst,
        test_edge_weight=test_weight,
        test_edge_ptr=test_ptr,
        seq_in_len=np.array(seq_in_len),
        num_nodes=np.array(N),
    )
    if S is not None:
        s_mean = S.mean(axis=0, keepdims=True)
        s_std = S.std(axis=0, keepdims=True)
        s_std[s_std == 0] = 1.0
        S_norm = (S - s_mean) / s_std
        out["static_node_features"] = S_norm

    np.savez_compressed(args.output_path, **out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_path", type=str, default="data/splits/all_samples.npz")
    parser.add_argument("--indices_path", type=str, default="data/splits/split_indices.npz")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--static_features_path", type=str, default="data/splits/static_node_features.npy")

    parser.add_argument(
        "--adjacency_mode",
        type=str,
        choices=["static", "dynamic"],
        required=True,
    )

    parser.add_argument("--adj_path", type=str, default=None)
    parser.add_argument("--dynamic_adj_path", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    samples = np.load(args.samples_path)
    indices = np.load(args.indices_path)

    X = samples["X"]
    Y = samples["Y"]

    train_idx = indices["train_idx"]
    val_idx = indices["val_idx"]
    test_idx = indices["test_idx"]

    S = None
    if os.path.exists(args.static_features_path):
        S = np.load(args.static_features_path).astype(np.float32)
        if S.shape[0] != X.shape[2]:
            raise ValueError(f"Static features nodes {S.shape[0]} != num_nodes {X.shape[2]}")
        print(f"Loaded static features: S.shape={S.shape}")
    else:
        print("Static features not found, proceeding without them")

    if args.adjacency_mode == "static":
        if args.adj_path is None:
            raise ValueError("--adj_path is required for static adjacency mode")

        export_static(args, X, Y, train_idx, val_idx, test_idx, S)

    elif args.adjacency_mode == "dynamic":
        if args.dynamic_adj_path is None:
            raise ValueError("--dynamic_adj_path is required for dynamic adjacency mode")

        export_dynamic(args, X, Y, train_idx, val_idx, test_idx, S)

    print("Saved EvolveGCN split:", args.output_path)
    print("mode:", args.adjacency_mode)
    print("X:", X.shape)
    print("Y:", Y.shape)


if __name__ == "__main__":
    main()