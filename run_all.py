# run_all.py

import os
import json
import yaml
import shutil
import argparse
import subprocess
import numpy as np
from pathlib import Path


def run(cmd):
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_yaml(path):
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find hyperparameter config: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_config(default_cfg, adj_cfg):
    out = dict(default_cfg)
    if adj_cfg is not None:
        out.update(adj_cfg)
    return out


def tag_float(x):
    """
    Convert values like 0.0005 to 0005 and 0.3 to 3
    for compact model names.
    """
    s = str(x)
    s = s.replace("0.", "")
    s = s.replace(".", "")
    return s


def get_num_nodes(split_info_path, visits_path):
    if os.path.exists(split_info_path):
        with open(split_info_path, "r") as f:
            info = json.load(f)
        if "num_nodes" in info:
            return int(info["num_nodes"])

    import numpy as np
    visits = np.load(visits_path)
    return visits.shape[1]


def copy_if_exists(src, dst):
    if not os.path.exists(src):
        print(f"Warning: missing file: {src}")
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.copy(src, dst)
    print(f"Copied {src} -> {dst}")
    return True


def prepare_data(args):
    run([
        "python", "src/prepare_data.py",
        "--input_csv", args.raw_csv,
        "--output_dir", args.processed_dir,
    ])


def build_static_features(args):
    run([
        "python", "src/build_static_features.py",
        "--poi_meta", os.path.join(args.processed_dir, "poi_meta.csv"),
        "--location_table", args.location_table,
        "--output_dir", args.split_dir,
    ])


def create_shared_split(args):
    run([
        "python", "src/create_shared_split.py",
        "--visits_path", os.path.join(args.processed_dir, "visits.npy"),
        "--output_dir", args.split_dir,
        "--seq_in_len", str(args.seq_in_len),
        "--seq_out_len", str(args.seq_out_len),
        "--train_ratio", str(args.train_ratio),
        "--val_ratio", str(args.val_ratio),
    ])


def export_benchmark_split(args):
    run([
        "python", "src/export_benchmark_split.py",
        "--samples_path", os.path.join(args.split_dir, "all_samples.npz"),
        "--indices_path", os.path.join(args.split_dir, "split_indices.npz"),
        "--output_path", os.path.join(args.split_dir, "benchmarks", "benchmark_split.npz"),
    ])


def export_mtgnn_split(args):
    run([
        "python", "src/export_mtgnn_split.py",
        "--samples_path", os.path.join(args.split_dir, "all_samples.npz"),
        "--indices_path", os.path.join(args.split_dir, "split_indices.npz"),
        "--output_dir", os.path.join(args.split_dir, "mtgnn"),
        "--static_features_path", os.path.join(args.split_dir, "static_node_features.npy"),
        "--cat_ids_path", os.path.join(args.split_dir, "static_cat_ids.npy"),
        "--naics2_ids_path", os.path.join(args.split_dir, "static_naics2_ids.npy"),
        "--naics4_ids_path", os.path.join(args.split_dir, "static_naics4_ids.npy"),
        "--brand_ids_path", os.path.join(args.split_dir, "static_brand_ids.npy"),
        "--parent_ids_path", os.path.join(args.split_dir, "static_parent_ids.npy"),
        "--seq_in_len", str(args.seq_in_len),
        "--seq_out_len", str(args.seq_out_len),
    ])


def create_static_adjacencies(args):
    run([
        "python", "src/create_static_adjacencies.py",
        "--poi_meta", os.path.join(args.processed_dir, "poi_meta.csv"),
        "--location_table", args.location_table,
        "--visits_path", os.path.join(args.processed_dir, "visits.npy"),
        "--output_dir", args.static_adj_dir,
        "--knn_k", str(args.knn_k),
        "--rbf_radius_km", str(args.rbf_radius_km),
        "--rbf_sigma_km", str(args.rbf_sigma_km),
        "--corr_threshold", str(args.corr_threshold),
        "--normalize", args.adj_normalize,
        "--self_loops",
    ])


def run_benchmarks(args):
    benchmark_split = os.path.join(args.split_dir, "benchmarks", "benchmark_split.npz")

    if args.skip_benchmarks:
        print("Skipping benchmark models.")
        return

    run([
        "python", "runners/run_naive.py",
        "--split_path", benchmark_split,
        "--output_dir", args.prediction_dir,
        "--model_name", "naive",
    ])

    run([
        "python", "runners/run_ar.py",
        "--split_path", benchmark_split,
        "--output_dir", args.prediction_dir,
        "--model_name", "ar_ridge",
        "--alpha", str(args.ar_alpha),
    ])

    run([
        "python", "runners/run_lstm.py",
        "--split_path", benchmark_split,
        "--output_dir", args.prediction_dir,
        "--model_name", "lstm",
        "--epochs", str(args.lstm_epochs),
        "--batch_size", str(args.lstm_batch_size),
        "--hidden_dim", str(args.lstm_hidden_dim),
        "--num_layers", str(args.lstm_num_layers),
        "--dropout", str(args.lstm_dropout),
        "--lr", str(args.lstm_lr),
        "--device", args.device,
    ])


def run_mtgnn_for_adjacencies(args, hyperparams, num_nodes):
    if args.skip_mtgnn:
        print("Skipping MTGNN.")
        return

    mtgnn_split_dir = os.path.join(args.split_dir, "mtgnn")
    train_npz = np.load(os.path.join(mtgnn_split_dir, "train.npz"))
    in_dim = int(train_npz["x"].shape[1])
    print(f"Detected MTGNN in_dim={in_dim} from split files")

    info_path = os.path.join(args.split_dir, "static_node_features_info.json")
    cat_vocab_size = naics2_vocab_size = naics4_vocab_size = 0
    brand_vocab_size = parent_vocab_size = 0
    emb_dim = 8
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        cat_vocab_size = len(info.get("cat_vocab", []))
        naics2_vocab_size = len(info.get("naics2_vocab", []))
        naics4_vocab_size = len(info.get("naics4_vocab", []))
        brand_vocab_size = len(info.get("brand_vocab", []))
        parent_vocab_size = len(info.get("parent_vocab", []))
        emb_dim = info.get("cat_emb_dim", 8)
    print(f"  cat: {cat_vocab_size}, naics2: {naics2_vocab_size}, naics4: {naics4_vocab_size}")
    print(f"  brand: {brand_vocab_size}, parent: {parent_vocab_size}")
    print(f"  All categorical emb_dim: {emb_dim}")

    mtgnn_hyp = hyperparams.get("mtgnn", {})
    default_cfg = mtgnn_hyp.get("default", {})

    mtgnn_split_dir = os.path.join(args.split_dir, "mtgnn")
    poi_meta_path = os.path.join(args.processed_dir, "poi_meta.csv")

    for adj_name in args.static_adj_names:
        print("\n" + "#" * 80)
        print(f"MTGNN final run for adjacency: {adj_name}")
        print("#" * 80)

        adj_npy = os.path.join(args.static_adj_dir, f"adj_{adj_name}.npy")
        adj_pkl = os.path.join(args.static_adj_dir, f"mtgnn_adj_{adj_name}.pkl")

        if not os.path.exists(adj_npy):
            print(f"Skipping {adj_name}: missing adjacency file {adj_npy}")
            continue

        run([
            "python", "src/make_mtgnn_adj.py",
            "--adj_npy", adj_npy,
            "--poi_meta", poi_meta_path,
            "--output_pkl", adj_pkl,
        ])

        adj_cfg = mtgnn_hyp.get(adj_name, {})
        cfg = merge_config(default_cfg, adj_cfg)

        lr = cfg.get("learning_rate", args.mtgnn_learning_rate)
        dropout = cfg.get("dropout", args.mtgnn_dropout)
        batch_size = cfg.get("batch_size", args.mtgnn_batch_size)

        model_name = (
            f"mtgnn_{adj_name}"
            f"_lr{tag_float(lr)}"
            f"_do{tag_float(dropout)}"
            f"_bs{batch_size}"
        )

        save_prefix = f"./save/poi_{adj_name}_lr{tag_float(lr)}_do{tag_float(dropout)}_bs{batch_size}_"

        cmd = [
            "python", "runners/run_mtgnn.py",
            "--mtgnn_dir", args.mtgnn_dir,
            "--data_dir", mtgnn_split_dir,
            "--adj_pkl", adj_pkl,
            "--num_nodes", str(num_nodes),
            "--seq_in_len", str(args.seq_in_len),
            "--seq_out_len", str(args.seq_out_len),
            "--in_dim", str(in_dim),
            "--device", args.device,
            "--epochs", str(args.mtgnn_epochs),
            "--batch_size", str(batch_size),
            "--runs", str(args.mtgnn_runs),
            "--expid", str(args.expid),
            "--seed", str(args.seed),
            "--save_prefix", save_prefix,

            "--gcn_true", str(cfg.get("gcn_true", "True")),
            "--buildA_true", str(cfg.get("buildA_true", "True")),
            "--load_static_feature", str(cfg.get("load_static_feature", "False")),
            "--cl", str(cfg.get("cl", "True")),

            "--static_node_features", os.path.join(args.split_dir, "static_node_features.npy"),

            "--gcn_depth", str(cfg.get("gcn_depth", 2)),
            "--dropout", str(dropout),
            "--subgraph_size", str(cfg.get("subgraph_size", 20)),
            "--node_dim", str(cfg.get("node_dim", 40)),
            "--dilation_exponential", str(cfg.get("dilation_exponential", 1)),

            "--conv_channels", str(cfg.get("conv_channels", 32)),
            "--residual_channels", str(cfg.get("residual_channels", 32)),
            "--skip_channels", str(cfg.get("skip_channels", 64)),
            "--end_channels", str(cfg.get("end_channels", 128)),

            "--layers", str(cfg.get("layers", 3)),
            "--learning_rate", str(lr),
            "--weight_decay", str(cfg.get("weight_decay", 0.0001)),
            "--clip", str(cfg.get("clip", 5)),
            "--step_size1", str(cfg.get("step_size1", 2500)),
            "--step_size2", str(cfg.get("step_size2", 100)),
            "--print_every", str(cfg.get("print_every", 50)),

            "--propalpha", str(cfg.get("propalpha", 0.05)),
            "--tanhalpha", str(cfg.get("tanhalpha", 3)),
            "--num_split", str(cfg.get("num_split", args.mtgnn_num_split)),

            "--brand_vocab_size", str(brand_vocab_size),
            "--brand_emb_dim", str(emb_dim),
            "--parent_vocab_size", str(parent_vocab_size),
            "--parent_emb_dim", str(emb_dim),
            "--cat_vocab_size", str(cat_vocab_size),
            "--cat_emb_dim", str(emb_dim),
            "--naics2_vocab_size", str(naics2_vocab_size),
            "--naics2_emb_dim", str(emb_dim),
            "--naics4_vocab_size", str(naics4_vocab_size),
            "--naics4_emb_dim", str(emb_dim),
        ]

        run(cmd)

        mtgnn_output_dir = os.path.join(
            args.mtgnn_dir,
            "save",
            f"poi_{adj_name}_lr{tag_float(lr)}_do{tag_float(dropout)}_bs{batch_size}_outputs",
        )

        pred_src = os.path.join(mtgnn_output_dir, f"mtgnn_pred_expid{args.expid}.npy")
        true_src = os.path.join(mtgnn_output_dir, f"mtgnn_true_expid{args.expid}.npy")

        pred_dst = os.path.join(args.prediction_dir, f"{model_name}_pred.npy")
        true_dst = os.path.join(args.prediction_dir, f"{model_name}_true.npy")

        copied_pred = copy_if_exists(pred_src, pred_dst)
        copied_true = copy_if_exists(true_src, true_dst)

        if not (copied_pred and copied_true):
            print(f"Warning: MTGNN outputs were not fully copied for {model_name}")


def run_evolvegcn_for_adjacencies(args, hyperparams):
    if args.skip_evolvegcn:
        print("Skipping EvolveGCN.")
        return

    evol_hyp = hyperparams.get("evolvegcn", {})
    default_cfg = evol_hyp.get("default", {})

    samples_path = os.path.join(args.split_dir, "all_samples.npz")
    indices_path = os.path.join(args.split_dir, "split_indices.npz")
    evolvegcn_split_dir = os.path.join(args.split_dir, "evolvegcn")
    ensure_dir(evolvegcn_split_dir)

    for adj_name in args.static_adj_names:
        print("\n" + "#" * 80)
        print(f"EvolveGCN final run for adjacency: {adj_name}")
        print("#" * 80)

        adj_npy = os.path.join(args.static_adj_dir, f"adj_{adj_name}.npy")
        evolvegcn_data_path = os.path.join(
            evolvegcn_split_dir,
            f"poi_evolvegcn_static_{adj_name}.npz",
        )

        if not os.path.exists(adj_npy):
            print(f"Skipping {adj_name}: missing adjacency file {adj_npy}")
            continue

        run([
            "python", "src/export_evolvegcn_split.py",
            "--samples_path", samples_path,
            "--indices_path", indices_path,
            "--adjacency_mode", "static",
            "--adj_path", adj_npy,
            "--output_path", evolvegcn_data_path,
            "--static_features_path", os.path.join(args.split_dir, "static_node_features.npy"),
        ])

        adj_cfg = evol_hyp.get(adj_name, {})
        cfg = merge_config(default_cfg, adj_cfg)

        hidden_dim = cfg.get("hidden_dim", args.evolvegcn_hidden_dim)
        lr = cfg.get("lr", args.evolvegcn_lr)
        batch_size = cfg.get("batch_size", args.evolvegcn_batch_size)
        loss = cfg.get("loss", args.evolvegcn_loss)
        normalize = cfg.get("normalize", args.evolvegcn_normalize)

        model_name = (
            f"evolvegcn_static_{adj_name}"
            f"_h{hidden_dim}"
            f"_lr{tag_float(lr)}"
        )

        cmd = [
            "python", "runners/run_evolvegcn.py",
            "--evolvegcn_dir", args.evolvegcn_dir,
            "--data_path", evolvegcn_data_path,
            "--output_dir", args.prediction_dir,
            "--model_name", model_name,
            "--hidden_dim", str(hidden_dim),
            "--epochs", str(args.evolvegcn_epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--loss", str(loss),
            "--device", args.device,
        ]

        if normalize:
            cmd.append("--normalize")

        run(cmd)


def evaluate_outputs(args):
    run([
        "python", "src/evaluate_all.py",
        "--prediction_dir", args.prediction_dir,
        "--metric_dir", args.metric_dir,
        "--table_dir", args.table_dir,
    ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_csv", type=str, default="data/raw/Seattle_all.csv")
    parser.add_argument("--location_table", type=str, default="data/raw/location_table.csv")

    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--split_dir", type=str, default="data/splits")
    parser.add_argument("--static_adj_dir", type=str, default="data/adjacency/static")

    parser.add_argument("--prediction_dir", type=str, default="outputs/predictions")
    parser.add_argument("--metric_dir", type=str, default="outputs/metrics")
    parser.add_argument("--table_dir", type=str, default="outputs/tables")

    parser.add_argument("--mtgnn_dir", type=str, default="external/MTGNN")
    parser.add_argument("--evolvegcn_dir", type=str, default="external/EvolveGCN")

    parser.add_argument("--hyperparam_config", type=str, default="configs/best_hyperparams.yaml")

    parser.add_argument("--seq_in_len", type=int, default=14)
    parser.add_argument("--seq_out_len", type=int, default=7)
    parser.add_argument("--in_dim", type=int, default=1)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--expid", type=int, default=1)

    parser.add_argument(
        "--static_adj_names",
        nargs="+",
        default=[
            "identity",
            "distance_knn",
            "distance_rbf",
            "top_category",
            "naics2",
            "visit_correlation",
        ],
    )

    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--rbf_radius_km", type=float, default=2.0)
    parser.add_argument("--rbf_sigma_km", type=float, default=1.0)
    parser.add_argument("--corr_threshold", type=float, default=0.3)
    parser.add_argument("--adj_normalize", type=str, default="row")

    parser.add_argument("--skip_prepare_data", action="store_true")
    parser.add_argument("--skip_split", action="store_true")
    parser.add_argument("--skip_adjacency", action="store_true")
    parser.add_argument("--skip_benchmarks", action="store_true")
    parser.add_argument("--skip_mtgnn", action="store_true")
    parser.add_argument("--skip_evolvegcn", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")

    # Benchmark settings
    parser.add_argument("--ar_alpha", type=float, default=1.0)

    parser.add_argument("--lstm_epochs", type=int, default=50)
    parser.add_argument("--lstm_batch_size", type=int, default=32)
    parser.add_argument("--lstm_hidden_dim", type=int, default=64)
    parser.add_argument("--lstm_num_layers", type=int, default=1)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--lstm_lr", type=float, default=0.001)

    # MTGNN fallback defaults
    parser.add_argument("--mtgnn_epochs", type=int, default=100)
    parser.add_argument("--mtgnn_batch_size", type=int, default=8)
    parser.add_argument("--mtgnn_learning_rate", type=float, default=0.001)
    parser.add_argument("--mtgnn_dropout", type=float, default=0.3)
    parser.add_argument("--mtgnn_runs", type=int, default=1)
    parser.add_argument("--mtgnn_num_split", type=int, default=4)

    # EvolveGCN fallback defaults
    parser.add_argument("--evolvegcn_epochs", type=int, default=50)
    parser.add_argument("--evolvegcn_hidden_dim", type=int, default=16)
    parser.add_argument("--evolvegcn_batch_size", type=int, default=1)
    parser.add_argument("--evolvegcn_lr", type=float, default=0.001)
    parser.add_argument("--evolvegcn_loss", type=str, default="mse", choices=["mse", "mae", "huber"])
    parser.add_argument("--evolvegcn_normalize", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.processed_dir)
    ensure_dir(args.split_dir)
    ensure_dir(args.static_adj_dir)
    ensure_dir(args.prediction_dir)
    ensure_dir(args.metric_dir)
    ensure_dir(args.table_dir)

    hyperparams = load_yaml(args.hyperparam_config)

    if not args.skip_prepare_data:
        prepare_data(args)

    if not args.skip_split:
        create_shared_split(args)
        build_static_features(args)
        export_benchmark_split(args)
        export_mtgnn_split(args)

    if not args.skip_adjacency:
        create_static_adjacencies(args)

    split_info_path = os.path.join(args.split_dir, "split_info.json")
    visits_path = os.path.join(args.processed_dir, "visits.npy")
    num_nodes = get_num_nodes(split_info_path, visits_path)

    print("\nDetected number of nodes:", num_nodes)

    run_benchmarks(args)

    run_mtgnn_for_adjacencies(
        args=args,
        hyperparams=hyperparams,
        num_nodes=num_nodes,
    )

    run_evolvegcn_for_adjacencies(
        args=args,
        hyperparams=hyperparams,
    )

    if not args.skip_evaluation:
        evaluate_outputs(args)

    print("\nAll requested runs complete.")


if __name__ == "__main__":
    main()