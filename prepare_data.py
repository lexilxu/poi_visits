#!/usr/bin/env python
"""
Standalone data preparation script for POI Visits.
Combines all data preprocessing steps into a single script.

Usage:
    python prepare_data.py
    python prepare_data.py --raw_csv data/raw/Seattle_all.csv --output_dir data/processed
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from prepare_data import prepare_poi_visit_matrix
from create_shared_split import main as create_split_main
from build_static_features import main as build_features_main
from create_static_adjacencies import main as create_adjacency_main


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare POI visits data")
    parser.add_argument("--raw_csv", type=str, default="data/raw/Seattle_all.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--location_table", type=str, default="data/raw/location_table.csv")
    parser.add_argument("--split_dir", type=str, default="data/splits")
    parser.add_argument("--adjacency_dir", type=str, default="data/adjacency/static")
    
    # Split parameters
    parser.add_argument("--seq_in_len", type=int, default=14)
    parser.add_argument("--seq_out_len", type=int, default=7)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    
    # Adjacency parameters
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--rbf_radius_km", type=float, default=2.0)
    parser.add_argument("--rbf_sigma_km", type=float, default=1.0)
    parser.add_argument("--corr_threshold", type=float, default=0.3)
    
    return parser.parse_args()


def step1_prepare_data(args):
    """Step 1: Convert raw CSV to processed format."""
    print("\n" + "=" * 80)
    print("Step 1: Preparing data...")
    print("=" * 80)
    
    prepare_poi_visit_matrix(
        input_csv=args.raw_csv,
        output_dir=args.output_dir,
    )
    
    print(f"Saved processed data to {args.output_dir}")


def step2_create_split(args):
    """Step 2: Create train/val/test split and static features."""
    print("\n" + "=" * 80)
    print("Step 2: Creating data splits...")
    print("=" * 80)
    
    visits_path = os.path.join(args.output_dir, "visits.npy")
    
    sys.argv = [
        "create_shared_split.py",
        "--visits_path", visits_path,
        "--output_dir", args.split_dir,
        "--seq_in_len", str(args.seq_in_len),
        "--seq_out_len", str(args.seq_out_len),
        "--train_ratio", str(args.train_ratio),
        "--val_ratio", str(args.val_ratio),
    ]
    create_split_main()
    
    print(f"Created split at {args.split_dir}")


def step2b_build_features(args):
    """Step 2b: Build static node features."""
    print("\n" + "=" * 80)
    print("Step 2b: Building static features...")
    print("=" * 80)
    
    poi_meta = os.path.join(args.output_dir, "poi_meta.csv")
    
    sys.argv = [
        "build_static_features.py",
        "--poi_meta", poi_meta,
        "--location_table", args.location_table,
        "--output_dir", args.split_dir,
    ]
    build_features_main()
    
    print(f"Built static features at {args.split_dir}")


def step3_create_adjacency(args):
    """Step 3: Create static adjacency matrices."""
    print("\n" + "=" * 80)
    print("Step 3: Creating adjacency matrices...")
    print("=" * 80)
    
    poi_meta = os.path.join(args.output_dir, "poi_meta.csv")
    visits_path = os.path.join(args.output_dir, "visits.npy")
    
    sys.argv = [
        "create_static_adjacencies.py",
        "--poi_meta", poi_meta,
        "--location_table", args.location_table,
        "--visits_path", visits_path,
        "--output_dir", args.adjacency_dir,
        "--knn_k", str(args.knn_k),
        "--rbf_radius_km", str(args.rbf_radius_km),
        "--rbf_sigma_km", str(args.rbf_sigma_km),
        "--corr_threshold", str(args.corr_threshold),
        "--normalize", "row",
        "--self_loops",
    ]
    create_adjacency_main()
    
    print(f"Created adjacency matrices at {args.adjacency_dir}")


def main():
    args = parse_args()
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.split_dir, exist_ok=True)
    os.makedirs(args.adjacency_dir, exist_ok=True)
    
    # Run all steps
    step1_prepare_data(args)
    step2_create_split(args)
    step2b_build_features(args)
    step3_create_adjacency(args)
    
    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("Now you can run model training with run_experiments.sh")
    print("=" * 80)


if __name__ == "__main__":
    main()
