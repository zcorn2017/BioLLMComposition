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

# ESMC: the ``esm`` package pulls checkpoints via huggingface_hub.snapshot_download
# (EvolutionaryScale/esmc-*-2024-12). Offline GPU jobs need these dirs in the HF cache.
# Optional: BIOLLM_ESMC_REPOS="repo_id ..." space-separated (default: 300M + 600M).
ESMC_REPOS="${BIOLLM_ESMC_REPOS:-EvolutionaryScale/esmc-300m-2024-12 EvolutionaryScale/esmc-600m-2024-12}"
export ESMC_REPOS
echo ">> Caching ESMC Hub snapshots: $ESMC_REPOS"
python - <<'PY'
import os

from huggingface_hub import snapshot_download

for repo_id in os.environ["ESMC_REPOS"].split():
    print(f"  Snapshot {repo_id}...")
    snapshot_download(repo_id=repo_id)
print("  ESMC snapshots cached.")
PY

# NTv3: gated on Hugging Face — on each model page click Access / accept terms, then auth:
#   export HF_TOKEN=...   or   huggingface-cli login  (→ ~/.cache/huggingface/token)
# Fine-grained tokens: Settings → Access tokens → edit token → enable
#   "Access to public gated repositories" (403 without it). Classic read tokens work if you have repo access.
# GPU jobs use HF_HUB_OFFLINE — cache on login node.
# Optional: BIOLLM_NTV3_MODELS="InstaDeepAI/NTv3_8M_pre ..." space-separated.
NTV3_MODELS="${BIOLLM_NTV3_MODELS:-InstaDeepAI/NTv3_100M_pre}"
export NTV3_MODELS
echo ">> Caching NTv3 DNA model(s): $NTV3_MODELS"
echo "   (If download fails with 403: fine-grained HF token needs 'public gated repositories' permission.)"
python - <<'PY'
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer

from biollmcomposition.models.ntv3 import _ensure_ntv3_modules_on_path

for name in os.environ["NTV3_MODELS"].split():
    print(f"  Downloading {name}...")
    AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # Remote code uses absolute imports; transformers checks imports before the cache dir is on sys.path.
    _ensure_ntv3_modules_on_path()
    AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True, low_cpu_mem_usage=False)
print("  NTv3 model(s) cached.")
PY

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
