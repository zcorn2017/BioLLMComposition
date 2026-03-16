#!/bin/bash
# One-time remote environment setup for biocluster.
#
# Run from the project root:
#   bash slurm/setup_env.sh
#
# Or via Makefile:
#   make setup-env

set -euo pipefail

ENV_NAME="py314-llm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "BioLLMComposition: Remote Environment Setup"
echo "=================================================="
echo "Project dir: $PROJECT_DIR"
echo

# ── Load module ──────────────────────────────────────────────────────────
echo ">> Loading Miniconda3 module..."
module load Miniconda3
eval "$(conda shell.bash hook)"
echo "   conda: $(conda --version)"

# ── Create / update conda env ────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ">> Environment '$ENV_NAME' already exists. Updating..."
    conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment-cluster.yml" --prune
else
    echo ">> Creating environment '$ENV_NAME'..."
    conda env create -f "$PROJECT_DIR/environment-cluster.yml"
fi

echo ">> Activating '$ENV_NAME'..."
conda activate "$ENV_NAME"

# ── Install project in editable mode ─────────────────────────────────────
echo ">> Installing biollmcomposition in editable mode..."
cd "$PROJECT_DIR"
pip install -e .

# ── Pre-download all HF models (login node has internet; compute nodes do not) ─
echo ">> Caching DNABERT-2 tokenizer + model on login node..."
python -c "
from transformers import AutoTokenizer, AutoModel, BertConfig
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)
print('  Downloading model...')
config = BertConfig.from_pretrained('zhihan1996/DNABERT-2-117M')
AutoModel.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True, config=config, low_cpu_mem_usage=False)
print('  DNABERT-2 cached.')
"

echo ">> Caching ESM2 models..."
python -c "
from transformers import AutoModelForMaskedLM, AutoTokenizer
for name in ['facebook/esm2_t36_3B_UR50D']:
    print(f'  Downloading {name}...')
    AutoTokenizer.from_pretrained(name)
    AutoModelForMaskedLM.from_pretrained(name, low_cpu_mem_usage=False)
print('  ESM2 models cached.')
"

if [ -f "$PROJECT_DIR/patch/apply_patch.sh" ]; then
  echo ">> Applying DNABERT-2 Triton patch (if cache exists)..."
  if bash "$PROJECT_DIR/patch/apply_patch.sh" 2>/dev/null; then
    echo "   Patch applied."
  else
    echo "   (patch skipped; apply on first GPU run if needed)"
  fi
fi

# ── Create output directories ────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/runs"
mkdir -p "$PROJECT_DIR/slurm/logs"

echo
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo
echo "Test with:"
echo "  module load Miniconda3"
echo "  eval \"\$(conda shell.bash hook)\""
echo "  conda activate $ENV_NAME"
echo "  python -c \"import biollmcomposition; print('OK')\""
