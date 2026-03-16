"""DNABERT-2 DNA language model.

Uses byte-pair encoding (BPE) on raw DNA sequences and requires
``trust_remote_code=True`` with a ``BertConfig`` for proper loading.

References
----------
Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark
For Multi-Species Genome", ICLR 2024.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer, BertConfig

DNABERT2_VARIANTS: dict[str, dict] = {
    "dnabert2-117M": {
        "hf_name": "zhihan1996/DNABERT-2-117M",
        "layers": 12, "params": "117M", "emb_dim": 768,
    },
}


def load_dnabert2_tokenizer(short_name: str):
    return AutoTokenizer.from_pretrained(
        DNABERT2_VARIANTS[short_name]["hf_name"], trust_remote_code=True,
    )


def load_dnabert2_model(short_name: str, device: str = "cpu",
                        dtype: torch.dtype = torch.bfloat16):
    hf_name = DNABERT2_VARIANTS[short_name]["hf_name"]
    config = BertConfig.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(
        hf_name, trust_remote_code=True, config=config,
    )
    model.to(dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)
