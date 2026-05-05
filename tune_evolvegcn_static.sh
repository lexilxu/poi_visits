#!/bin/bash

set -e

# ============================================================
# Tune EvolveGCN-style model for each static adjacency
# ============================================================

DEVICE="cuda:0"

SAMPLES_PATH="data/splits/all_samples.npz"
INDICES_PATH="data/splits/split_indices.npz"

STATIC_ADJ_DIR="data/adjacency/static"
EVOLVEGCN_SPLIT_DIR="data/splits/evolvegcn"

PREDICTION_DIR="outputs/predictions"
EVOLVEGCN_DIR="external/EvolveGCN"

mkdir -p "${EVOLVEGCN_SPLIT_DIR}"
mkdir -p "${PREDICTION_DIR}"
mkdir -p outputs/validation
mkdir -p outputs/training_logs

# Static adjacency matrices to tune.
STATIC_ADJ_NAMES=(
  identity
  distance_knn
  distance_rbf
  top_category
  naics2
  visit_correlation
)

# Hyperparameter grid.
HIDDEN_DIMS=(8 16 32)
LEARNING_RATES=(0.001 0.0005 0.0001)

# Keep batch size small because N is large.
BATCH_SIZE=1

# Use moderate epochs for tuning.
EPOCHS=30

run_cmd () {
    echo
    echo "============================================================"
    echo "Running:"
    echo "$@"
    echo "============================================================"
    echo
    "$@"
}

for ADJ in "${STATIC_ADJ_NAMES[@]}"; do

  echo
  echo "############################################################"
  echo "Tuning EvolveGCN-style model for adjacency: ${ADJ}"
  echo "############################################################"

  ADJ_PATH="${STATIC_ADJ_DIR}/adj_${ADJ}.npy"
  DATA_PATH="${EVOLVEGCN_SPLIT_DIR}/poi_evolvegcn_static_${ADJ}.npz"

  if [ ! -f "${ADJ_PATH}" ]; then
    echo "Skipping ${ADJ}: adjacency file not found: ${ADJ_PATH}"
    continue
  fi

  # Export EvolveGCN split for this adjacency.
  run_cmd python src/export_evolvegcn_split.py \
    --samples_path "${SAMPLES_PATH}" \
    --indices_path "${INDICES_PATH}" \
    --adjacency_mode static \
    --adj_path "${ADJ_PATH}" \
    --output_path "${DATA_PATH}"

  for HIDDEN in "${HIDDEN_DIMS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do

      # Make a clean model name.
      LR_TAG=$(echo "${LR}" | sed 's/0\.//g' | sed 's/\.//g')
      MODEL_NAME="evolvegcn_static_${ADJ}_h${HIDDEN}_lr${LR_TAG}"

      echo
      echo "Running configuration:"
      echo "  adjacency: ${ADJ}"
      echo "  hidden_dim: ${HIDDEN}"
      echo "  lr: ${LR}"
      echo "  model_name: ${MODEL_NAME}"

      run_cmd python runners/run_evolvegcn.py \
        --evolvegcn_dir "${EVOLVEGCN_DIR}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${PREDICTION_DIR}" \
        --model_name "${MODEL_NAME}" \
        --hidden_dim "${HIDDEN}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --device "${DEVICE}"

    done
  done

done

echo
echo "EvolveGCN static adjacency tuning complete."
echo "Validation metrics should be in outputs/validation/"
echo "Training logs should be in outputs/training_logs/"
echo "Predictions should be in outputs/predictions/"

