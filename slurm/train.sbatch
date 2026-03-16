#!/bin/bash
#SBATCH --partition=cabbi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

SCRIPT="${1:?Usage: sbatch slurm/train.sbatch <script.py> <config.yaml> [extra args...]}"
CONFIG="${2:?Usage: sbatch slurm/train.sbatch <script.py> <config.yaml> [extra args...]}"
shift 2

echo "=================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Script:    $SCRIPT"
echo "Config:    $CONFIG"
echo "Extra:     $*"
echo "Start:     $(date)"
echo "=================================================="

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate py314-llm

export DATA_ROOT="$HOME/Projects/proteinDNA_data"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

nvidia-smi

python -u "$SCRIPT" --config "$CONFIG" "$@"

echo "=================================================="
echo "Finished:  $(date)"
echo "=================================================="
