# scripts/create_dynamic_adjacencies.py

import os
import argparse
import numpy as np


def topk_corr_edges(window, k=10, threshold=0.0):
    """
    window: (W, N), rolling visits window

    Returns:
        edge_src, edge_dst, edge_weight
    """
    W, N = window.shape

    if W < 2:
        src = np.arange(N)
        dst = np.arange(N)
        weight = np.ones(N, dtype=np.float32)
        return src, dst, weight

    corr = np.corrcoef(window.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Use absolute correlation as edge strength
    score = np.abs(corr).astype(np.float32)
    np.fill_diagonal(score, 0.0)

    src_list = []
    dst_list = []
    weight_list = []

    for i in range(N):
        row = score[i]

        if k >= N:
            candidates = np.argsort(-row)
        else:
            candidates = np.argpartition(-row, kth=min(k, N - 1))[:k]
            candidates = candidates[np.argsort(-row[candidates])]

        for j in candidates:
            if row[j] > threshold:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(row[j])

        # self-loop
        src_list.append(i)
        dst_list.append(i)
        weight_list.append(1.0)

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    weight = np.array(weight_list, dtype=np.float32)

    return src, dst, weight


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--visits_path", type=str, default="data/processed/visits.npy")
    parser.add_argument("--samples_path", type=str, default="data/splits/all_samples.npz")
    parser.add_argument("--output_path", type=str, default="data/adjacency/dynamic/dynamic_corr_k10.npz")

    parser.add_argument("--rolling_window", type=int, default=28)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    visits = np.load(args.visits_path).astype(np.float32)
    samples = np.load(args.samples_path)

    sample_start_times = samples["sample_start_times"]
    X = samples["X"]

    num_samples, P, N = X.shape

    edge_src_all = []
    edge_dst_all = []
    edge_weight_all = []
    edge_ptr = [0]

    # We create one graph per sample per input step.
    # Total graph count = num_samples * P.
    for sample_id in range(num_samples):
        forecast_start_t = int(sample_start_times[sample_id])

        for p in range(P):
            current_t = forecast_start_t - P + p

            start = max(0, current_t - args.rolling_window + 1)
            end = current_t + 1

            window = visits[start:end, :]

            src, dst, weight = topk_corr_edges(
                window=window,
                k=args.top_k,
                threshold=args.threshold,
            )

            edge_src_all.append(src)
            edge_dst_all.append(dst)
            edge_weight_all.append(weight)

            edge_ptr.append(edge_ptr[-1] + len(src))

        if (sample_id + 1) % 50 == 0:
            print(f"Processed {sample_id + 1}/{num_samples} samples")

    edge_src = np.concatenate(edge_src_all)
    edge_dst = np.concatenate(edge_dst_all)
    edge_weight = np.concatenate(edge_weight_all)
    edge_ptr = np.array(edge_ptr, dtype=np.int64)

    np.savez_compressed(
        args.output_path,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_weight=edge_weight,
        edge_ptr=edge_ptr,
        num_samples=np.array(num_samples),
        seq_in_len=np.array(P),
        num_nodes=np.array(N),
        rolling_window=np.array(args.rolling_window),
        top_k=np.array(args.top_k),
        threshold=np.array(args.threshold),
    )

    print("Saved dynamic adjacency:", args.output_path)
    print("num_samples:", num_samples)
    print("seq_in_len:", P)
    print("num_nodes:", N)
    print("num_graphs:", num_samples * P)
    print("num_edges total:", len(edge_src))


if __name__ == "__main__":
    main()