"""Cross-attention contact-map model.

Protein (query) attends to each DNA strand (key/value) independently.
A shared bilinear head produces ``(B, 2, R, L)`` logits.
"""

from __future__ import annotations

import torch
from torch import nn


class ContactMapHead(nn.Module):
    """Bilinear contact head: score(r, l) = prot_proj[r] . dna_proj[l]."""

    def __init__(self, prot_dim: int, dna_dim: int, head_dim: int = 64):
        super().__init__()
        self.prot_proj = nn.Linear(prot_dim, head_dim)
        self.dna_proj = nn.Linear(dna_dim, head_dim)

    def forward(self, prot_h, dna_h):
        return torch.bmm(
            self.prot_proj(prot_h),
            self.dna_proj(dna_h).transpose(1, 2),
        )


class AttentionContactMapModel(nn.Module):
    """Cross-attention model predicting per-residue contact maps.

    Protein (query) attends to each DNA strand (key/value) independently,
    then a shared bilinear head produces ``(B, 2, R, L)`` logits.
    """

    def __init__(self, dna_lm, prot_lm, dna_emb_dim: int, prot_emb_dim: int,
                 head_dim: int = 64):
        super().__init__()
        self.dna_lm = dna_lm
        self.prot_lm = prot_lm

        self.dna_proj = nn.Linear(dna_emb_dim, prot_emb_dim)
        self.attn = nn.MultiheadAttention(
            prot_emb_dim, num_heads=1, batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(prot_emb_dim)
        self.contact_head = ContactMapHead(prot_emb_dim, prot_emb_dim, head_dim)

        for p in self.dna_lm.parameters():
            p.requires_grad = False
        for p in self.prot_lm.parameters():
            p.requires_grad = False

    def _dna_hidden(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.dna_lm(input_ids=input_ids,
                              attention_mask=attention_mask)
        h = out[0] if isinstance(out, tuple) else out.last_hidden_state
        return self.dna_proj(h.float())

    def _prot_hidden(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.prot_lm(input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        return out.hidden_states[-1]

    def _cross_attn_strand(self, prot_h, dna_h, dna_am):
        out, _ = self.attn(
            query=prot_h.float(), key=dna_h, value=dna_h,
            key_padding_mask=~dna_am.bool(),
        )
        return self.post_attn_norm(out) + prot_h.float()

    def forward(self, batch):
        prot_h = self._prot_hidden(batch["prot_input_ids"],
                                   batch["prot_attention_mask"])
        dna1_h = self._dna_hidden(batch["dna1_input_ids"],
                                  batch["dna1_attention_mask"])
        dna2_h = self._dna_hidden(batch["dna2_input_ids"],
                                  batch["dna2_attention_mask"])

        ctx1 = self._cross_attn_strand(prot_h, dna1_h,
                                       batch["dna1_attention_mask"])
        ctx2 = self._cross_attn_strand(prot_h, dna2_h,
                                       batch["dna2_attention_mask"])

        cm1 = self.contact_head(ctx1, dna1_h)
        cm2 = self.contact_head(ctx2, dna2_h)
        return torch.stack([cm1, cm2], dim=1)


def build_model(dna_lm, prot_lm, dna_info: dict, prot_info: dict,
                arch_cfg: dict, device: str = "cpu"):
    """Construct an :class:`AttentionContactMapModel` from registry info and
    architecture config."""
    return AttentionContactMapModel(
        dna_lm, prot_lm,
        dna_emb_dim=dna_info["emb_dim"],
        prot_emb_dim=prot_info["emb_dim"],
        head_dim=arch_cfg.get("head_dim", 64),
    ).to(device)
