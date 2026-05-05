# scripts/make_mtgnn_adj.py

import os
import pickle
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--adj_npy", type=str, required=True)
    parser.add_argument("--poi_meta", type=str, default="data/processed/poi_meta.csv")
    parser.add_argument("--output_pkl", type=str, required=True)
    parser.add_argument("--id_col", type=str, default="placekey")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    adj = np.load(args.adj_npy).astype(np.float32)
    poi_meta = pd.read_csv(args.poi_meta)

    if args.id_col not in poi_meta.columns:
        raise ValueError(f"{args.id_col} not found in {args.poi_meta}")

    sensor_ids = poi_meta[args.id_col].astype(str).tolist()
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}

    N = len(sensor_ids)

    if adj.shape != (N, N):
        raise ValueError(
            f"Adjacency shape {adj.shape} does not match number of POIs {N}."
        )

    with open(args.output_pkl, "wb") as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj), f)

    print("Saved MTGNN adjacency pickle:")
    print("  ", args.output_pkl)
    print("Adjacency shape:", adj.shape)
    print("Number of sensor ids:", len(sensor_ids))


if __name__ == "__main__":
    main()