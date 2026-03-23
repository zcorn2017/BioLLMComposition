#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpushare:2
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

set -euo pipefail

SCRIPT="${1:?Usage: sbatch slurm/train-compute.sbatch <script.py> <config.yaml> [extra args...]}"
CONFIG="${2:?Usage: sbatch slurm/train-compute.sbatch <script.py> <config.yaml> [extra args...]}"
shift 2

echo "=================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-MPS}"
echo "Script:    $SCRIPT"
echo "Config:    $CONFIG"
echo "Extra:     $*"
echo "Start:     $(date)"
echo "=================================================="

export PATH="/home/zcorn/.conda/envs/py314-llm/bin:$PATH"

export DATA_ROOT="$HOME/Projects/proteinDNA_data"
# Compute nodes have no HF egress; models must already be in ~/.cache/huggingface (see slurm/setup_env*.sh).
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

nvidia-smi

python -u "$SCRIPT" --config "$CONFIG" "$@"

echo "=================================================="
echo "Finished:  $(date)"
echo "=================================================="
