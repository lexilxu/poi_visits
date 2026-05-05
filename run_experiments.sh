#!/bin/bash

set -e

# ============================================================
# POI multistep forecasting experiment script
# ============================================================

# -------------------------------
# Basic experiment settings
# -------------------------------
RAW_CSV="data/raw/Seattle_all.csv"
LOCATION_TABLE="data/raw/location_table.csv"

SEQ_IN_LEN=14
SEQ_OUT_LEN=7

TRAIN_RATIO=0.7
VAL_RATIO=0.1

DEVICE="cuda:0"

# -------------------------------
# Training settings
# -------------------------------
BATCH_SIZE=16

BENCHMARK_EPOCHS=20
MTGNN_EPOCHS=100
EVOLVEGCN_EPOCHS=50

EVOLVEGCN_BATCH_SIZE=1
EVOLVEGCN_HIDDEN_DIM=16
EVOLVEGCN_LR=0.001

# -------------------------------
# Adjacency settings
# -------------------------------
STATIC_ADJ_NAMES=(
  identity
  distance_knn
  distance_rbf
  top_category
  naics2
  visit_correlation
)

RUN_DYNAMIC_EVOLVEGCN=true

DYNAMIC_ADJ_NAMES=(
  dynamic_corr_k10
)

KNN_K=10
RBF_RADIUS_KM=2.0
RBF_SIGMA_KM=1.0
CORR_THRESHOLD=0.3
ADJ_NORMALIZE="row"

ROLLING_WINDOW=28
DYNAMIC_TOP_K=10
DYNAMIC_THRESHOLD=0.0

# -------------------------------
# External repo paths
# -------------------------------
MTGNN_DIR="external/MTGNN"
EVOLVEGCN_DIR="external/EvolveGCN"

# -------------------------------
# Standard paths
# -------------------------------
PROCESSED_DIR="data/processed"
SPLIT_DIR="data/splits"

SAMPLES_PATH="${SPLIT_DIR}/all_samples.npz"
INDICES_PATH="${SPLIT_DIR}/split_indices.npz"
SPLIT_INFO_PATH="${SPLIT_DIR}/split_info.json"

MTGNN_SPLIT_DIR="${SPLIT_DIR}/mtgnn"
BENCHMARK_SPLIT_PATH="${SPLIT_DIR}/benchmarks/benchmark_split.npz"

STATIC_ADJ_DIR="data/adjacency/static"
DYNAMIC_ADJ_DIR="data/adjacency/dynamic"

PREDICTION_DIR="outputs/predictions"
METRIC_DIR="outputs/metrics"
TABLE_DIR="outputs/tables"

mkdir -p "${PREDICTION_DIR}"
mkdir -p "${METRIC_DIR}"
mkdir -p "${TABLE_DIR}"

# ============================================================
# Helper function
# ============================================================
run_cmd () {
    echo
    echo "============================================================"
    echo "Running:"
    echo "$@"
    echo "============================================================"
    echo
    "$@"
}

## ============================================================
## 1. Prepare data
## ============================================================
#
#run_cmd python src/prepare_data.py \
#  --input_csv "${RAW_CSV}" \
#  --output_dir "${PROCESSED_DIR}"
#
## ============================================================
## 2. Create the single shared split
## ============================================================
#
#run_cmd python src/create_shared_split.py \
#  --visits_path "${PROCESSED_DIR}/visits.npy" \
#  --output_dir "${SPLIT_DIR}" \
#  --seq_in_len "${SEQ_IN_LEN}" \
#  --seq_out_len "${SEQ_OUT_LEN}" \
#  --train_ratio "${TRAIN_RATIO}" \
#  --val_ratio "${VAL_RATIO}"
#
#run_cmd python src/build_static_features.py \
#  --poi_meta "${PROCESSED_DIR}/poi_meta.csv" \
#  --location_table "${LOCATION_TABLE}" \
#  --output_dir "${SPLIT_DIR}"
#
## ============================================================
## 3. Export model-specific split files
## ============================================================
#
#run_cmd python src/export_mtgnn_split.py \
#  --samples_path "${SAMPLES_PATH}" \
#  --indices_path "${INDICES_PATH}" \
#  --output_dir "${MTGNN_SPLIT_DIR}" \
#  --static_features_path "${SPLIT_DIR}/static_node_features.npy" \
#  --cat_ids_path "${SPLIT_DIR}/static_cat_ids.npy" \
#  --naics2_ids_path "${SPLIT_DIR}/static_naics2_ids.npy" \
#  --naics4_ids_path "${SPLIT_DIR}/static_naics4_ids.npy" \
#  --brand_ids_path "${SPLIT_DIR}/static_brand_ids.npy" \
#  --parent_ids_path "${SPLIT_DIR}/static_parent_ids.npy" \
#  --seq_in_len "${SEQ_IN_LEN}" \
#  --seq_out_len "${SEQ_OUT_LEN}"
#
#run_cmd python src/export_benchmark_split.py \
#  --samples_path "${SAMPLES_PATH}" \
#  --indices_path "${INDICES_PATH}" \
#  --output_path "${BENCHMARK_SPLIT_PATH}"
#
## ============================================================
## 4. Create static adjacency matrices
## ============================================================
#
#run_cmd python src/create_static_adjacencies.py \
#  --poi_meta "${PROCESSED_DIR}/poi_meta.csv" \
#  --location_table "${LOCATION_TABLE}" \
#  --visits_path "${PROCESSED_DIR}/visits.npy" \
#  --output_dir "${STATIC_ADJ_DIR}" \
#  --knn_k "${KNN_K}" \
#  --rbf_radius_km "${RBF_RADIUS_KM}" \
#  --rbf_sigma_km "${RBF_SIGMA_KM}" \
#  --corr_threshold "${CORR_THRESHOLD}" \
#  --normalize "${ADJ_NORMALIZE}" \
#  --self_loops
#
## ============================================================
## 5. Optional dynamic adjacency matrices for EvolveGCN
## ============================================================
#
#if [ "${RUN_DYNAMIC_EVOLVEGCN}" = true ]; then
#  mkdir -p "${DYNAMIC_ADJ_DIR}"
#
#  run_cmd python src/create_dynamic_adjacencies.py \
#    --visits_path "${PROCESSED_DIR}/visits.npy" \
#    --samples_path "${SAMPLES_PATH}" \
#    --output_path "${DYNAMIC_ADJ_DIR}/dynamic_corr_k10.npz" \
#    --rolling_window "${ROLLING_WINDOW}" \
#    --top_k "${DYNAMIC_TOP_K}" \
#    --threshold "${DYNAMIC_THRESHOLD}"
#fi


# ============================================================
# 6. Get in_dim from MTGNN split files
# ============================================================

IN_DIM=$(python - <<EOF
import json, os
info = {}
static_info = os.path.join("${SPLIT_DIR}", "static_node_features_info.json")
if os.path.exists(static_info):
    with open(static_info) as f:
        info = json.load(f)
D_s = info.get("D_s_continuous", 0)
has = sum(1 for k in ["cat_vocab","naics2_vocab","naics4_vocab","brand_vocab","parent_vocab"] if info.get(k))
emb = has * info.get("cat_emb_dim", 8)
print(1 + D_s + emb)
EOF
)

CAT_VOCAB=$(python - <<EOF
import json, os
v = 0
static_info = os.path.join("${SPLIT_DIR}", "static_node_features_info.json")
if os.path.exists(static_info):
    with open(static_info) as f:
        v = len(json.load(f).get("cat_vocab", []))
print(v)
EOF
)

NAICS2_VOCAB=$(python - <<EOF
import json, os
v = 0
static_info = os.path.join("${SPLIT_DIR}", "static_node_features_info.json")
if os.path.exists(static_info):
    with open(static_info) as f:
        v = len(json.load(f).get("naics2_vocab", []))
print(v)
EOF
)

NAICS4_VOCAB=$(python - <<EOF
import json, os
v = 0
static_info = os.path.join("${SPLIT_DIR}", "static_node_features_info.json")
if os.path.exists(static_info):
    with open(static_info) as f:
        v = len(json.load(f).get("naics4_vocab", []))
print(v)
EOF
)

BRAND_VOCAB=$(python - <<EOF
import json, os
v = 0
static_info = os.path.join("${SPLIT_DIR}", "static_node_features_info.json")
if os.path.exists(static_info):
    with open(static_info) as f:
        v = len(json.load(f).get("brand_vocab", []))
print(v)
EOF
)

PARENT_VOCAB=$(python - <<EOF
import json, os
v = 0
static_info = os.path.join("${SPLIT_DIR}", "static_node_features_info.json")
if os.path.exists(static_info):
    with open(static_info) as f:
        v = len(json.load(f).get("parent_vocab", []))
print(v)
EOF
)

echo "Detected IN_DIM=${IN_DIM}"

# ============================================================
# 7. Get number of nodes
# ============================================================

NUM_NODES=$(python - <<EOF
import json
with open("${SPLIT_INFO_PATH}", "r") as f:
    info = json.load(f)
print(info["num_nodes"])
EOF
)

echo "Detected NUM_NODES=${NUM_NODES}"

# ============================================================
# 8. Run benchmark models once
# ============================================================

run_cmd python runners/run_naive.py \
  --split_path "${BENCHMARK_SPLIT_PATH}" \
  --output_dir "${PREDICTION_DIR}" \
  --model_name naive

run_cmd python runners/run_ar.py \
  --split_path "${BENCHMARK_SPLIT_PATH}" \
  --output_dir "${PREDICTION_DIR}" \
  --model_name ar_ridge \
  --alpha 1.0

run_cmd python runners/run_lstm.py \
  --split_path "${BENCHMARK_SPLIT_PATH}" \
  --output_dir "${PREDICTION_DIR}" \
  --model_name lstm \
  --epochs "${BENCHMARK_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --hidden_dim 64 \
  --device "${DEVICE}"

# Optional MLP. Uncomment if you have runners/run_mlp.py.
# run_cmd python runners/run_mlp.py \
#   --split_path "${BENCHMARK_SPLIT_PATH}" \
#   --output_dir "${PREDICTION_DIR}" \
#   --model_name mlp \
#   --epochs "${BENCHMARK_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --device "${DEVICE}"

# ============================================================
# 9. Run static adjacency experiments
#    MTGNN + EvolveGCN-style model
# ============================================================

for ADJ_NAME in "${STATIC_ADJ_NAMES[@]}"; do
  echo
  echo "############################################################"
  echo "Static adjacency experiment: ${ADJ_NAME}"
  echo "############################################################"

  ADJ_NPY="${STATIC_ADJ_DIR}/adj_${ADJ_NAME}.npy"
  MTGNN_ADJ_PKL="${STATIC_ADJ_DIR}/mtgnn_adj_${ADJ_NAME}.pkl"
  EVOLVEGCN_SPLIT_PATH="${SPLIT_DIR}/evolvegcn/poi_evolvegcn_static_${ADJ_NAME}.npz"

  if [ ! -f "${ADJ_NPY}" ]; then
    echo "Skipping ${ADJ_NAME}: missing ${ADJ_NPY}"
    continue
  fi

  # -------------------------------
  # Convert adjacency to MTGNN pkl
  # -------------------------------
  run_cmd python src/make_mtgnn_adj.py \
    --adj_npy "${ADJ_NPY}" \
    --poi_meta "${PROCESSED_DIR}/poi_meta.csv" \
    --output_pkl "${MTGNN_ADJ_PKL}"

  # -------------------------------
  # Export EvolveGCN static split
  # -------------------------------
  run_cmd python src/export_evolvegcn_split.py \
    --samples_path "${SAMPLES_PATH}" \
    --indices_path "${INDICES_PATH}" \
    --adjacency_mode static \
    --adj_path "${ADJ_NPY}" \
    --output_path "${EVOLVEGCN_SPLIT_PATH}" \
    --static_features_path "${SPLIT_DIR}/static_node_features.npy"

  # -------------------------------
  # Run MTGNN
  # -------------------------------
  SAVE_PREFIX="./save/poi_${ADJ_NAME}_"

  run_cmd python runners/run_mtgnn.py \
    --mtgnn_dir "${MTGNN_DIR}" \
    --data_dir "${MTGNN_SPLIT_DIR}" \
    --adj_pkl "${MTGNN_ADJ_PKL}" \
    --num_nodes "${NUM_NODES}" \
    --seq_in_len "${SEQ_IN_LEN}" \
    --seq_out_len "${SEQ_OUT_LEN}" \
    --in_dim "${IN_DIM}" \
    --device "${DEVICE}" \
    --epochs "${MTGNN_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --runs 1 \
    --expid 1 \
    --save_prefix "${SAVE_PREFIX}" \
    --cat_vocab_size "${CAT_VOCAB}" \
    --cat_emb_dim 8 \
    --naics2_vocab_size "${NAICS2_VOCAB}" \
    --naics2_emb_dim 8 \
    --naics4_vocab_size "${NAICS4_VOCAB}" \
    --naics4_emb_dim 8 \
    --brand_vocab_size "${BRAND_VOCAB}" \
    --brand_emb_dim 8 \
    --parent_vocab_size "${PARENT_VOCAB}" \
    --parent_emb_dim 8

  # Copy MTGNN outputs into common prediction folder.
  MTGNN_OUTPUT_DIR="${MTGNN_DIR}/save/poi_${ADJ_NAME}_outputs"

  if [ -f "${MTGNN_OUTPUT_DIR}/mtgnn_pred_expid1.npy" ]; then
    cp "${MTGNN_OUTPUT_DIR}/mtgnn_pred_expid1.npy" \
       "${PREDICTION_DIR}/mtgnn_${ADJ_NAME}_pred.npy"

    cp "${MTGNN_OUTPUT_DIR}/mtgnn_true_expid1.npy" \
       "${PREDICTION_DIR}/mtgnn_${ADJ_NAME}_true.npy"
  else
    echo "Warning: MTGNN output not found for ${ADJ_NAME}"
    echo "Expected: ${MTGNN_OUTPUT_DIR}/mtgnn_pred_expid1.npy"
  fi

  # -------------------------------
  # Run EvolveGCN-style model
  # -------------------------------
  run_cmd python runners/run_evolvegcn.py \
    --evolvegcn_dir "${EVOLVEGCN_DIR}" \
    --data_path "${EVOLVEGCN_SPLIT_PATH}" \
    --output_dir "${PREDICTION_DIR}" \
    --model_name "evolvegcn_static_${ADJ_NAME}" \
    --hidden_dim "${EVOLVEGCN_HIDDEN_DIM}" \
    --epochs "${EVOLVEGCN_EPOCHS}" \
    --batch_size "${EVOLVEGCN_BATCH_SIZE}" \
    --lr "${EVOLVEGCN_LR}" \
    --device "${DEVICE}"

done

# ============================================================
# 10. Dynamic adjacency experiments for EvolveGCN only
# ============================================================

if [ "${RUN_DYNAMIC_EVOLVEGCN}" = true ]; then
  for DYN_NAME in "${DYNAMIC_ADJ_NAMES[@]}"; do
    echo
    echo "############################################################"
    echo "Dynamic EvolveGCN experiment: ${DYN_NAME}"
    echo "############################################################"

    DYNAMIC_ADJ_PATH="${DYNAMIC_ADJ_DIR}/${DYN_NAME}.npz"
    EVOLVEGCN_DYNAMIC_SPLIT_PATH="${SPLIT_DIR}/evolvegcn/poi_evolvegcn_${DYN_NAME}.npz"

    if [ ! -f "${DYNAMIC_ADJ_PATH}" ]; then
      echo "Skipping ${DYN_NAME}: missing ${DYNAMIC_ADJ_PATH}"
      continue
    fi

    run_cmd python src/export_evolvegcn_split.py \
      --samples_path "${SAMPLES_PATH}" \
      --indices_path "${INDICES_PATH}" \
      --adjacency_mode dynamic \
      --dynamic_adj_path "${DYNAMIC_ADJ_PATH}" \
      --output_path "${EVOLVEGCN_DYNAMIC_SPLIT_PATH}" \
      --static_features_path "${SPLIT_DIR}/static_node_features.npy"

    run_cmd python runners/run_evolvegcn.py \
      --evolvegcn_dir "${EVOLVEGCN_DIR}" \
      --data_path "${EVOLVEGCN_DYNAMIC_SPLIT_PATH}" \
      --output_dir "${PREDICTION_DIR}" \
      --model_name "evolvegcn_${DYN_NAME}" \
      --hidden_dim "${EVOLVEGCN_HIDDEN_DIM}" \
      --epochs "${EVOLVEGCN_EPOCHS}" \
      --batch_size "${EVOLVEGCN_BATCH_SIZE}" \
      --lr "${EVOLVEGCN_LR}" \
      --device "${DEVICE}"
  done
fi

# ============================================================
# 11. Final evaluation
# ============================================================

run_cmd python src/evaluate_all.py \
  --prediction_dir "${PREDICTION_DIR}" \
  --metric_dir "${METRIC_DIR}" \
  --table_dir "${TABLE_DIR}"

echo
echo "============================================================"
echo "Experiment complete."
echo "Predictions: ${PREDICTION_DIR}"
echo "Metrics:     ${METRIC_DIR}"
echo "Tables:      ${TABLE_DIR}"
echo "============================================================"
