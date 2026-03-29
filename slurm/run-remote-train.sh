#!/usr/bin/env bash
# Train on a GPU host without SLURM (same role as train-compute.sbatch, no #SBATCH).
# Invoked by: make submit TARGET=testgpu ...
#
# Usage (from repo root):
#   bash slurm/run-remote-train.sh <script.py> <config.yaml> [extra args...]
#
# Uses conda env BIOLLM_CONDA_ENV (default: py314-llm). Override DATA_ROOT if needed.

set -euo pipefail

SCRIPT="${1:?Usage: $0 <script.py> <config.yaml> [extra args...]}"
CONFIG="${2:?Usage: $0 <script.py> <config.yaml> [extra args...]}"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export DATA_ROOT="${DATA_ROOT:-$HOME/Projects/proteinDNA_data}"
ENV_NAME="${BIOLLM_CONDA_ENV:-py314-llm}"

_init_conda_for_run() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  for p in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "$p/etc/profile.d/conda.sh"
      return 0
    fi
  done
  if [ -f "/data/conda/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /data/conda/bin/activate
    return 0
  fi
  return 1
}

_init_conda_for_run

if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
  exec conda run --no-capture-output -n "$ENV_NAME" python -u "$SCRIPT" --config "$CONFIG" "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python -u "$SCRIPT" --config "$CONFIG" "$@"
fi

echo "No conda env '$ENV_NAME' and no python on PATH. Run setup-env on the host first." >&2
exit 1
