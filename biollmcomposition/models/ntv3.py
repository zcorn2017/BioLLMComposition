"""Nucleotide Transformer v3 (NTv3) DNA language model.

Uses single-base tokenization (one token per nucleotide) with a U-Net
architecture: convolutional downsampling → Transformer → deconvolutional
upsampling, yielding nucleotide-resolution embeddings.

Loaded via HuggingFace ``AutoModelForMaskedLM`` with
``trust_remote_code=True``.  A thin wrapper
(:class:`NTv3EmbeddingWrapper`) extracts the final hidden states so
that the output interface matches DNABERT-2 (``out.last_hidden_state``).

Sequence lengths must be divisible by ``2 ** num_downsamples`` (128 for
the standard 7-downsample variants).

References
----------
Boshar et al., "A foundational model for joint sequence-function
multi-species modeling at scale for long-range genomic prediction",
bioRxiv 2025.
"""

from __future__ import annotations

import glob
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


def _ensure_ntv3_modules_on_path() -> None:
    """Add cached NTv3 module directory to ``sys.path``.

    HuggingFace's ``check_imports`` tries to ``importlib.import_module``
    every non-relative import before the module directory is on the path.
    NTv3's modeling file uses ``from configuration_ntv3_pretrained import …``
    (absolute, not relative), so the check fails.  Putting the cached
    module directory on ``sys.path`` first resolves the issue.
    """
    pattern = os.path.join(
        os.path.expanduser("~"),
        ".cache", "huggingface", "modules", "transformers_modules",
        "InstaDeepAI", "ntv3_base_model", "*",
    )
    for d in sorted(glob.glob(pattern), reverse=True):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
            return

NTV3_VARIANTS: dict[str, dict] = {
    "ntv3-8M": {
        "hf_name": "InstaDeepAI/NTv3_8M_pre",
        "layers": 2, "params": "8M", "emb_dim": 256,
        "num_downsamples": 7, "has_special_tokens": False,
        "stem_only": True,
    },
    "ntv3-100M": {
        "hf_name": "InstaDeepAI/NTv3_100M_pre",
        "layers": 6, "params": "100M", "emb_dim": 768,
        "num_downsamples": 7, "has_special_tokens": False,
        "stem_only": True,
    },
    "ntv3-650M": {
        "hf_name": "InstaDeepAI/NTv3_650M_pre",
        "layers": 12, "params": "650M", "emb_dim": 1536,
        "num_downsamples": 7, "has_special_tokens": False,
        "stem_only": True,
    },
}


class NTv3EmbeddingWrapper(nn.Module):
    """Wraps an NTv3 ``AutoModelForMaskedLM`` to expose ``.last_hidden_state``.

    Two modes are supported:

    ``stem_only=True`` (default, recommended for short sequences):
        Runs only the learned nucleotide embedding table and the stem
        Conv1d (kernel=15, 16→768 channels).  Each output position sees
        ±7 neighbouring nucleotides.  No downsampling, no length
        constraint.  Output: ``(B, L, emb_dim)``.

    ``stem_only=False``:
        Runs the full U-Net forward pass and returns the final deconv
        hidden state.  Requires sequence length divisible by
        ``2**num_downsamples`` (128 for standard variants).
        Output: ``(B, L, emb_dim)``.
    """

    def __init__(self, model: nn.Module, stem_only: bool = True):
        super().__init__()
        self.model = model
        self.stem_only = stem_only

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if self.stem_only:
            core = self.model.core
            x = core.embed_layer(input_ids)          # (B, L, token_embed_dim)
            x = core.stem(x.permute(0, 2, 1))        # (B, emb_dim, L)
            return SimpleNamespace(last_hidden_state=x.permute(0, 2, 1).float())
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return SimpleNamespace(last_hidden_state=out.hidden_states[-1])


def load_ntv3_tokenizer(short_name: str):
    _ensure_ntv3_modules_on_path()
    hf_name = NTV3_VARIANTS[short_name]["hf_name"]
    return AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)


def load_ntv3_model(short_name: str, device: str = "cpu",
                    dtype: torch.dtype = torch.float32):
    _ensure_ntv3_modules_on_path()
    info = NTV3_VARIANTS[short_name]
    hf_name = info["hf_name"]
    stem_only = info.get("stem_only", True)
    model = AutoModelForMaskedLM.from_pretrained(
        hf_name, trust_remote_code=True,
    )
    model.to(dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    wrapped = NTv3EmbeddingWrapper(model, stem_only=stem_only)
    wrapped.eval()
    return wrapped.to(device)
