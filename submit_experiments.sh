#!/bin/bash

#SBATCH --job-name=poi_forecasting
#SBATCH --output=logs/poi_forecasting_%j.out
#SBATCH --error=logs/poi_forecasting_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yida0101@uw.edu

#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a100:1    

set -e

mkdir -p logs

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

source ~/.bashrc
conda activate pytorch-cuda11 

which python
python --version
nvidia-smi || true

bash run_experiments.sh

echo "End time: $(date)"
echo "Job finished."

