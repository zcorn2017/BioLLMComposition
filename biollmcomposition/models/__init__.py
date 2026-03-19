"""Unified model registry for protein and DNA language models.

Provides a single interface to look up model metadata (embedding dims,
layer counts), load tokenizers, and load models by short name.

Usage
-----
>>> from biollmcomposition.models import get_model_info, load_tokenizer, load_model
>>> info = get_model_info("esm2-8M")
>>> tok  = load_tokenizer("esm2-8M")
>>> model = load_model("esm2-8M", device="cuda")
"""

from __future__ import annotations

import torch

from .dnabert2 import DNABERT2_VARIANTS, load_dnabert2_model, load_dnabert2_tokenizer
from .esm2 import ESM2_VARIANTS, load_esm2_model, load_esm2_tokenizer
from .esmc import ESMC_VARIANTS, load_esmc_model, load_esmc_tokenizer
from .ntv3 import NTV3_VARIANTS, load_ntv3_model, load_ntv3_tokenizer

MODEL_REGISTRY: dict[str, dict] = {}
MODEL_REGISTRY.update(
    {k: {**v, "family": "esm2", "modality": "protein"} for k, v in ESM2_VARIANTS.items()}
)
MODEL_REGISTRY.update(
    {k: {**v, "family": "esmc", "modality": "protein",
         "hf_name": v["esm_name"]}  # display-compat alias for scripts
     for k, v in ESMC_VARIANTS.items()}
)
MODEL_REGISTRY.update(
    {k: {**v, "family": "dnabert2", "modality": "dna"} for k, v in DNABERT2_VARIANTS.items()}
)
MODEL_REGISTRY.update(
    {k: {**v, "family": "ntv3", "modality": "dna"} for k, v in NTV3_VARIANTS.items()}
)

_TOKENIZER_LOADERS = {
    "esm2": load_esm2_tokenizer,
    "esmc": load_esmc_tokenizer,
    "dnabert2": load_dnabert2_tokenizer,
    "ntv3": load_ntv3_tokenizer,
}
_MODEL_LOADERS = {
    "esm2": load_esm2_model,
    "esmc": load_esmc_model,
    "dnabert2": load_dnabert2_model,
    "ntv3": load_ntv3_model,
}


def get_model_info(short_name: str) -> dict:
    """Return the registry entry for *short_name* (e.g. ``"esm2-8M"``)."""
    if short_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{short_name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[short_name]


def load_tokenizer(short_name: str):
    """Load the tokenizer for a registered model."""
    info = get_model_info(short_name)
    return _TOKENIZER_LOADERS[info["family"]](short_name)


_FAMILY_DEFAULT_DTYPE: dict[str, torch.dtype] = {
    "ntv3": torch.float32,
}


def load_model(short_name: str, device: str = "cpu",
               dtype: torch.dtype | None = None):
    """Load the model in eval mode with frozen weights.

    Parameters
    ----------
    dtype : torch.dtype | None
        Parameter dtype for the frozen model.  When *None* (the default),
        uses ``bfloat16`` for most families but ``float32`` for NTv3
        (whose U-Net Conv1d layers require homogeneous dtypes).
    """
    info = get_model_info(short_name)
    family = info["family"]
    if dtype is None:
        dtype = _FAMILY_DEFAULT_DTYPE.get(family, torch.bfloat16)
    return _MODEL_LOADERS[family](short_name, device, dtype=dtype)
