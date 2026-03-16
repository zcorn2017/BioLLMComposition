"""ESM-C protein language model variants.

Loaded via the ``esm`` package (``pip install esm``).  ESM-C shares the
same token vocabulary as ESM2, so pre-computed token files are
interchangeable between the two families.

Internal structure (relevant for the composition framework)::

    model.embed            → nn.Embedding(input_ids) → (B, L, D)
    model.transformer      → TransformerStack
      .blocks[i]           → UnifiedTransformerBlock
                              forward(x, sequence_id, frames, frames_mask, chain_id) → x
      .norm                → LayerNorm  (final norm, analogous to ESM2's emb_layer_norm_after)

``sequence_id`` is a boolean ``(B, L)`` mask (True = non-pad).
``frames``, ``frames_mask`` are ``None`` for language-only mode.
``chain_id`` defaults to ones ``(B, L)``.

References
----------
Hayes et al., "Simulating 500 million years of evolution with a
language model", bioRxiv 2024.
"""

from __future__ import annotations

import torch
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer

ESMC_VARIANTS: dict[str, dict] = {
    "esmc-600M": {
        "esm_name": "esmc_600m",
        "layers": 36, "params": "600M", "emb_dim": 1152,
    },
    "esmc-300M": {
        "esm_name": "esmc_300m",
        "layers": 30, "params": "300M", "emb_dim": 960,
    },
}


def load_esmc_tokenizer(short_name: str):
    """Return an ESM-C tokenizer (same vocab as ESM2)."""
    _ = ESMC_VARIANTS[short_name]  # validate name
    return EsmSequenceTokenizer()


def load_esmc_model(short_name: str, device: str = "cpu",
                    dtype: torch.dtype = torch.bfloat16):
    """Load a frozen ESM-C model via the ``esm`` package."""
    esm_name = ESMC_VARIANTS[short_name]["esm_name"]
    model = ESMC.from_pretrained(esm_name, device=torch.device(device))
    model.to(dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
