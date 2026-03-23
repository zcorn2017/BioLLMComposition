"""Nucleotide Transformer v3 (NTv3) DNA language model.

Uses single-base tokenization (one token per nucleotide) with a U-Net
architecture: convolutional downsampling → Transformer → deconvolutional
upsampling, yielding nucleotide-resolution embeddings.

Loaded via HuggingFace ``AutoModelForMaskedLM`` with
``trust_remote_code=True``.  A thin wrapper
(:class:`NTv3EmbeddingWrapper`) exposes ``out.last_hidden_state`` like
DNABERT-2.  Default ``embed_mode="first_hidden"`` uses the first
conv-tower hidden state at full length; only ``embed_mode="full"`` needs
``L`` divisible by ``2 ** num_downsamples`` (128 for 7 downsamples).

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

    Snapshot layout varies by ``huggingface_hub`` version (e.g. legacy
    ``InstaDeepAI/ntv3_base_model/<rev>/`` vs repo-id folders); fall back
    to scanning for ``configuration_ntv3_pretrained.py``.
    """
    hf_modules = os.path.join(
        os.path.expanduser("~"),
        ".cache", "huggingface", "modules", "transformers_modules",
    )
    pattern = os.path.join(hf_modules, "InstaDeepAI", "ntv3_base_model", "*")
    for d in sorted(glob.glob(pattern), reverse=True):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
            return
    if not os.path.isdir(hf_modules):
        return
    candidates: list[str] = []
    for dirpath, _, filenames in os.walk(hf_modules):
        if "configuration_ntv3_pretrained.py" in filenames:
            candidates.append(dirpath)
    if not candidates:
        return
    best = max(
        candidates,
        key=lambda p: os.path.getmtime(os.path.join(p, "configuration_ntv3_pretrained.py")),
    )
    if best not in sys.path:
        sys.path.insert(0, best)

NTV3_VARIANTS: dict[str, dict] = {
    "ntv3-8M": {
        "hf_name": "InstaDeepAI/NTv3_8M_pre",
        "layers": 2, "params": "8M", "emb_dim": 256,
        "num_downsamples": 7, "has_special_tokens": False,
        "embed_mode": "first_hidden",
    },
    "ntv3-100M": {
        "hf_name": "InstaDeepAI/NTv3_100M_pre",
        "layers": 6, "params": "100M", "emb_dim": 768,
        "num_downsamples": 7, "has_special_tokens": False,
        "embed_mode": "first_hidden",
    },
    "ntv3-650M": {
        "hf_name": "InstaDeepAI/NTv3_650M_pre",
        "layers": 12, "params": "650M", "emb_dim": 1536,
        "num_downsamples": 7, "has_special_tokens": False,
        "embed_mode": "first_hidden",
    },
}


class NTv3EmbeddingWrapper(nn.Module):
    """Wraps an NTv3 ``AutoModelForMaskedLM`` to expose ``.last_hidden_state``.

    ``embed_mode`` selects how deep into the U-Net embeddings are taken
    (all return nucleotide resolution ``(B, L, C)`` except notes below):

    ``"stem"``:
        Embedding table + stem Conv1d only (±7 nt receptive field).
        No length constraint beyond tokenizer limits.

    ``"first_hidden"`` (default):
        Embed → stem → first conv-tower block (k=5 conv + residual conv),
        *without* pooling.  Same ``L`` as stem but wider receptive field.
        No length constraint.

    ``"full"``:
        Full U-Net and final deconv output (``hidden_states[-1]``).
        Requires ``L`` divisible by ``2**num_downsamples`` (128 for
        standard variants).
    """

    def __init__(self, model: nn.Module, embed_mode: str = "first_hidden"):
        super().__init__()
        self.model = model
        if embed_mode not in ("stem", "first_hidden", "full"):
            raise ValueError(
                f"embed_mode must be 'stem', 'first_hidden', or 'full', got {embed_mode!r}"
            )
        self.embed_mode = embed_mode

    def forward(self, input_ids, attention_mask=None, **kwargs):
        core = self.model.core
        if self.embed_mode in ("stem", "first_hidden"):
            x = core.embed_layer(input_ids)          # (B, L, token_embed_dim)
            x = core.stem(x.permute(0, 2, 1))        # (B, C, L)
            if self.embed_mode == "stem":
                return SimpleNamespace(last_hidden_state=x.permute(0, 2, 1).float())
            block = core.conv_tower_blocks[0]
            x = block.res_conv(block.conv(x))         # (B, C', L) — no pooling
            return SimpleNamespace(last_hidden_state=x.permute(0, 2, 1).float())
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return SimpleNamespace(last_hidden_state=out.hidden_states[-1].float())


def load_ntv3_tokenizer(short_name: str):
    _ensure_ntv3_modules_on_path()
    hf_name = NTV3_VARIANTS[short_name]["hf_name"]
    return AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)


def load_ntv3_model(short_name: str, device: str = "cpu",
                    dtype: torch.dtype = torch.float32):
    _ensure_ntv3_modules_on_path()
    info = NTV3_VARIANTS[short_name]
    hf_name = info["hf_name"]
    embed_mode = info.get("embed_mode", "first_hidden")
    model = AutoModelForMaskedLM.from_pretrained(
        hf_name, trust_remote_code=True,
    )
    model.to(dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    wrapped = NTv3EmbeddingWrapper(model, embed_mode=embed_mode)
    wrapped.eval()
    return wrapped.to(device)
