#!/bin/bash

set -e

# ============================================================
# Tune MTGNN for each static adjacency
# ============================================================

DEVICE="cuda:0"

MTGNN_DIR="external/MTGNN"
MTGNN_SPLIT_DIR="data/splits/mtgnn"
STATIC_ADJ_DIR="data/adjacency/static"
PREDICTION_DIR="outputs/predictions"

PROCESSED_DIR="data/processed"

SEQ_IN_LEN=14
SEQ_OUT_LEN=7
IN_DIM=1

EPOCHS=50
RUNS=1
EXPID=1

GCN_TRUE=True
BUILDA_TRUE=True
CL=True
LOAD_STATIC_FEATURE=False

mkdir -p "${PREDICTION_DIR}"
mkdir -p "${STATIC_ADJ_DIR}"
mkdir -p outputs/mtgnn_tuning_logs

# Static adjacency matrices to tune.
STATIC_ADJ_NAMES=(
  identity
  distance_knn
  distance_rbf
  top_category
  naics2
  visit_correlation
)

# Compact tuning grid.
# Start small because MTGNN is expensive.
LRS=(0.001 0.0005)
DROPOUTS=(0.1 0.3)
BATCH_SIZES=(8 16)

# Optional architecture grid.
# Keep these fixed at first.
SUBGRAPH_SIZE=20
NODE_DIM=40
LAYERS=3
DILATION_EXPONENTIAL=1

CONV_CHANNELS=16
RESIDUAL_CHANNELS=16
SKIP_CHANNELS=32
END_CHANNELS=64

WEIGHT_DECAY=0.0001

# Graph options.
# gcn_true means use graph convolution.
# buildA_true means learn/adapt adjacency internally.
GCN_TRUE=true
BUILDA_TRUE=false

run_cmd () {
    echo
    echo "============================================================"
    echo "Running:"
    echo "$@"
    echo "============================================================"
    echo
    "$@"
}

NUM_NODES=$(python - <<EOF
import numpy as np
print(np.load("${PROCESSED_DIR}/visits.npy").shape[1])
EOF
)

echo "Detected NUM_NODES=${NUM_NODES}"

for ADJ in "${STATIC_ADJ_NAMES[@]}"; do

  echo
  echo "############################################################"
  echo "Tuning MTGNN for adjacency: ${ADJ}"
  echo "############################################################"

  ADJ_NPY="${STATIC_ADJ_DIR}/adj_${ADJ}.npy"
  MTGNN_ADJ_PKL="${STATIC_ADJ_DIR}/mtgnn_adj_${ADJ}.pkl"

  if [ ! -f "${ADJ_NPY}" ]; then
    echo "Skipping ${ADJ}: missing ${ADJ_NPY}"
    continue
  fi

  # Convert adjacency to MTGNN format.
  run_cmd python src/make_mtgnn_adj.py \
    --adj_npy "${ADJ_NPY}" \
    --poi_meta "${PROCESSED_DIR}/poi_meta.csv" \
    --output_pkl "${MTGNN_ADJ_PKL}"

  for LR in "${LRS[@]}"; do
    for DROPOUT in "${DROPOUTS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do

        LR_TAG=$(echo "${LR}" | sed 's/0\.//g' | sed 's/\.//g')
        DROPOUT_TAG=$(echo "${DROPOUT}" | sed 's/0\.//g' | sed 's/\.//g')

        MODEL_NAME="mtgnn_${ADJ}_lr${LR_TAG}_do${DROPOUT_TAG}_bs${BATCH_SIZE}"
        SAVE_PREFIX="./save/poi_${ADJ}_lr${LR_TAG}_do${DROPOUT_TAG}_bs${BATCH_SIZE}_"

        echo
        echo "Running MTGNN configuration:"
        echo "  adjacency: ${ADJ}"
        echo "  lr: ${LR}"
        echo "  dropout: ${DROPOUT}"
        echo "  batch_size: ${BATCH_SIZE}"
        echo "  model_name: ${MODEL_NAME}"

        CMD=(
          python runners/run_mtgnn.py
          --mtgnn_dir "${MTGNN_DIR}"
          --data_dir "${MTGNN_SPLIT_DIR}"
          --adj_pkl "${MTGNN_ADJ_PKL}"
          --num_nodes "${NUM_NODES}"
          --seq_in_len "${SEQ_IN_LEN}"
          --seq_out_len "${SEQ_OUT_LEN}"
          --in_dim "${IN_DIM}"
          --device "${DEVICE}"
          --epochs "${EPOCHS}"
          --batch_size "${BATCH_SIZE}"
          --runs "${RUNS}"
          --expid "${EXPID}"
          --save_prefix "${SAVE_PREFIX}"
          --learning_rate "${LR}"
          --weight_decay "${WEIGHT_DECAY}"
          --dropout "${DROPOUT}"
          --subgraph_size "${SUBGRAPH_SIZE}"
          --node_dim "${NODE_DIM}"
          --layers "${LAYERS}"
          --dilation_exponential "${DILATION_EXPONENTIAL}"
          --conv_channels "${CONV_CHANNELS}"
          --residual_channels "${RESIDUAL_CHANNELS}"
          --skip_channels "${SKIP_CHANNELS}"
          --end_channels "${END_CHANNELS}"
        )

        CMD+=(--gcn_true "${GCN_TRUE}")
        CMD+=(--buildA_true "${BUILDA_TRUE}")
        CMD+=(--cl "${CL}")
        CMD+=(--load_static_feature "${LOAD_STATIC_FEATURE}")

        if [ "${BUILDA_TRUE}" = true ]; then
          CMD+=(--buildA_true)
        fi

        run_cmd "${CMD[@]}" 2>&1 | tee "outputs/mtgnn_tuning_logs/${MODEL_NAME}.log"

        # Copy MTGNN outputs into common prediction folder.
        MTGNN_OUTPUT_DIR="${MTGNN_DIR}/save/poi_${ADJ}_lr${LR_TAG}_do${DROPOUT_TAG}_bs${BATCH_SIZE}_outputs"

        if [ -f "${MTGNN_OUTPUT_DIR}/mtgnn_pred_expid${EXPID}.npy" ]; then
          cp "${MTGNN_OUTPUT_DIR}/mtgnn_pred_expid${EXPID}.npy" \
             "${PREDICTION_DIR}/${MODEL_NAME}_pred.npy"

          cp "${MTGNN_OUTPUT_DIR}/mtgnn_true_expid${EXPID}.npy" \
             "${PREDICTION_DIR}/${MODEL_NAME}_true.npy"

          echo "Copied MTGNN outputs for ${MODEL_NAME}"
        else
          echo "Warning: MTGNN output not found for ${MODEL_NAME}"
          echo "Expected: ${MTGNN_OUTPUT_DIR}/mtgnn_pred_expid${EXPID}.npy"
        fi

      done
    done
  done

done

echo
echo "MTGNN static tuning complete."
echo "Predictions saved in ${PREDICTION_DIR}"
echo "Logs saved in outputs/mtgnn_tuning_logs"
