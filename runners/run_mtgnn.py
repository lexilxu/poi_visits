# runners/run_mtgnn.py

import argparse
import subprocess
from pathlib import Path


def bool_to_str(x):
    if isinstance(x, bool):
        return "True" if x else "False"
    x = str(x).lower()
    if x in ["true", "1", "yes", "y"]:
        return "True"
    if x in ["false", "0", "no", "n"]:
        return "False"
    raise ValueError(f"Cannot interpret boolean value: {x}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mtgnn_dir", type=str, default="external/MTGNN")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--adj_pkl", type=str, required=True)

    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--seq_in_len", type=int, default=14)
    parser.add_argument("--seq_out_len", type=int, default=7)
    parser.add_argument("--in_dim", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--expid", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--save_prefix", type=str, default="./save/poi_")

    # MTGNN boolean options. These must be passed to train_multi_step.py as True/False strings.
    parser.add_argument("--gcn_true", type=str, default="True")
    parser.add_argument("--buildA_true", type=str, default="True")
    parser.add_argument("--load_static_feature", type=str, default="False")
    parser.add_argument("--cl", type=str, default="True")

    # MTGNN architecture and training hyperparameters.
    parser.add_argument("--gcn_depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--subgraph_size", type=int, default=20)
    parser.add_argument("--node_dim", type=int, default=40)
    parser.add_argument("--dilation_exponential", type=int, default=1)

    parser.add_argument("--conv_channels", type=int, default=32)
    parser.add_argument("--residual_channels", type=int, default=32)
    parser.add_argument("--skip_channels", type=int, default=64)
    parser.add_argument("--end_channels", type=int, default=128)

    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--clip", type=int, default=5)
    parser.add_argument("--step_size1", type=int, default=2500)
    parser.add_argument("--step_size2", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=50)

    parser.add_argument("--propalpha", type=float, default=0.05)
    parser.add_argument("--tanhalpha", type=float, default=3)
    parser.add_argument("--num_split", type=int, default=1)

    parser.add_argument("--cat_vocab_size", type=int, default=0)
    parser.add_argument("--cat_emb_dim", type=int, default=8)
    parser.add_argument("--naics2_vocab_size", type=int, default=0)
    parser.add_argument("--naics2_emb_dim", type=int, default=8)
    parser.add_argument("--naics4_vocab_size", type=int, default=0)
    parser.add_argument("--naics4_emb_dim", type=int, default=8)
    parser.add_argument("--brand_vocab_size", type=int, default=0)
    parser.add_argument("--brand_emb_dim", type=int, default=8)
    parser.add_argument("--parent_vocab_size", type=int, default=0)
    parser.add_argument("--parent_emb_dim", type=int, default=8)

    args = parser.parse_args()

    mtgnn_dir = Path(args.mtgnn_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    adj_pkl = Path(args.adj_pkl).resolve()

    train_script = mtgnn_dir / "train_multi_step.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Cannot find train_multi_step.py: {train_script}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot find MTGNN data directory: {data_dir}")

    for split in ["train.npz", "val.npz", "test.npz"]:
        split_path = data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing MTGNN split file: {split_path}")

    if not adj_pkl.exists():
        raise FileNotFoundError(f"Cannot find adjacency pickle: {adj_pkl}")

    save_prefix_path = Path(args.save_prefix)
    if not save_prefix_path.is_absolute():
        save_dir = mtgnn_dir / save_prefix_path.parent
    else:
        save_dir = save_prefix_path.parent

    save_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python", "train_multi_step.py",

        "--device", args.device,
        "--data", str(data_dir),
        "--adj_data", str(adj_pkl),

        "--gcn_true", bool_to_str(args.gcn_true),
        "--buildA_true", bool_to_str(args.buildA_true),
        "--load_static_feature", bool_to_str(args.load_static_feature),
        "--cl", bool_to_str(args.cl),

        "--gcn_depth", str(args.gcn_depth),
        "--num_nodes", str(args.num_nodes),
        "--dropout", str(args.dropout),
        "--subgraph_size", str(args.subgraph_size),
        "--node_dim", str(args.node_dim),
        "--dilation_exponential", str(args.dilation_exponential),

        "--conv_channels", str(args.conv_channels),
        "--residual_channels", str(args.residual_channels),
        "--skip_channels", str(args.skip_channels),
        "--end_channels", str(args.end_channels),

        "--in_dim", str(args.in_dim),
        "--seq_in_len", str(args.seq_in_len),
        "--seq_out_len", str(args.seq_out_len),

        "--layers", str(args.layers),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--clip", str(args.clip),
        "--step_size1", str(args.step_size1),
        "--step_size2", str(args.step_size2),

        "--epochs", str(args.epochs),
        "--print_every", str(args.print_every),
        "--seed", str(args.seed),
        "--save", args.save_prefix,
        "--expid", str(args.expid),

        "--propalpha", str(args.propalpha),
        "--tanhalpha", str(args.tanhalpha),
        "--num_split", str(args.num_split),

        "--cat_vocab_size", str(args.cat_vocab_size),
        "--cat_emb_dim", str(args.cat_emb_dim),
        "--naics2_vocab_size", str(args.naics2_vocab_size),
        "--naics2_emb_dim", str(args.naics2_emb_dim),
        "--naics4_vocab_size", str(args.naics4_vocab_size),
        "--naics4_emb_dim", str(args.naics4_emb_dim),
        "--brand_vocab_size", str(args.brand_vocab_size),
        "--brand_emb_dim", str(args.brand_emb_dim),
        "--parent_vocab_size", str(args.parent_vocab_size),
        "--parent_emb_dim", str(args.parent_emb_dim),

        "--runs", str(args.runs),
    ]

    print("MTGNN directory:", mtgnn_dir)
    print("MTGNN save directory:", save_dir)
    print("Running command:")
    print(" ".join(command))

    subprocess.run(command, cwd=mtgnn_dir, check=True)


if __name__ == "__main__":
    main()