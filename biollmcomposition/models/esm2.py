"""ESM2 protein language model variants.

All variants share the same architecture and are loaded via
``AutoModelForMaskedLM``.  The model returned exposes internal encoder
layers (``model.esm.encoder.layer[i]``) needed by the composition
framework.

References
----------
Lin et al., "Evolutionary-scale prediction of atomic-level protein
structure with a language model", Science 2023.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ESM2_VARIANTS: dict[str, dict] = {
    "esm2-15B": {
        "hf_name": "facebook/esm2_t48_15B_UR50D",
        "layers": 48, "params": "15B", "emb_dim": 5120,
    },
    "esm2-3B": {
        "hf_name": "facebook/esm2_t36_3B_UR50D",
        "layers": 36, "params": "3B", "emb_dim": 2560,
    },
    "esm2-650M": {
        "hf_name": "facebook/esm2_t33_650M_UR50D",
        "layers": 33, "params": "650M", "emb_dim": 1280,
    },
    "esm2-150M": {
        "hf_name": "facebook/esm2_t30_150M_UR50D",
        "layers": 30, "params": "150M", "emb_dim": 640,
    },
    "esm2-35M": {
        "hf_name": "facebook/esm2_t12_35M_UR50D",
        "layers": 12, "params": "35M", "emb_dim": 480,
    },
    "esm2-8M": {
        "hf_name": "facebook/esm2_t6_8M_UR50D",
        "layers": 6, "params": "8M", "emb_dim": 320,
    },
}


def load_esm2_tokenizer(short_name: str):
    return AutoTokenizer.from_pretrained(ESM2_VARIANTS[short_name]["hf_name"])


def load_esm2_model(short_name: str, device: str = "cpu",
                    dtype: torch.dtype = torch.bfloat16):
    hf_name = ESM2_VARIANTS[short_name]["hf_name"]
    model = AutoModelForMaskedLM.from_pretrained(hf_name, torch_dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)
