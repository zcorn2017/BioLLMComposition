"""Composition (LM-interleaved cross-attention) contact-map model.

DNA context from both strands is concatenated and injected into protein
transformer layers via cross-attention at selected layers.  The final
per-token protein representations are paired with per-strand DNA hidden
states through a bilinear head to produce ``(B, 2, R, L)`` logits.

Supports **ESM2** and **ESM-C** protein families through lightweight
encoder adapters that normalise the layer-stepping interface.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Encoder adapters — hide family-specific attribute paths
# ---------------------------------------------------------------------------

class _Esm2Encoder:
    """Adapter for HuggingFace ESM2 (``model.esm.encoder.layer[i]``)."""

    def __init__(self, prot_lm):
        self.prot_lm = prot_lm

    @property
    def num_layers(self) -> int:
        return len(self.prot_lm.esm.encoder.layer)

    def get_embeddings(self, input_ids, attention_mask):
        return self.prot_lm.esm.embeddings(input_ids, attention_mask)

    def prepare_mask(self, attention_mask, input_ids):
        return self.prot_lm.get_extended_attention_mask(
            attention_mask, input_ids.size(),
        )

    def apply_layer(self, i, hidden_states, mask):
        return self.prot_lm.esm.encoder.layer[i](hidden_states, mask)[0]

    def final_norm(self, hidden_states):
        return self.prot_lm.esm.encoder.emb_layer_norm_after(hidden_states)


class _EsmcEncoder:
    """Adapter for the ``esm`` package ESMC model.

    ``model.embed`` is a plain ``nn.Embedding``.
    ``model.transformer.blocks[i]`` are ``UnifiedTransformerBlock`` layers.
    ``model.transformer.norm`` is the final ``LayerNorm``.
    """

    def __init__(self, prot_lm):
        self.prot_lm = prot_lm

    @property
    def num_layers(self) -> int:
        return len(self.prot_lm.transformer.blocks)

    def get_embeddings(self, input_ids, attention_mask):
        return self.prot_lm.embed(input_ids)

    def prepare_mask(self, attention_mask, input_ids):
        return attention_mask.bool()

    def apply_layer(self, i, hidden_states, mask):
        chain_id = torch.ones(
            hidden_states.shape[:2], dtype=torch.int64,
            device=hidden_states.device,
        )
        return self.prot_lm.transformer.blocks[i](
            hidden_states, mask, None, None, chain_id,
        )

    def final_norm(self, hidden_states):
        return self.prot_lm.transformer.norm(hidden_states)


_ENCODER_ADAPTERS: dict[str, type] = {
    "esm2": _Esm2Encoder,
    "esmc": _EsmcEncoder,
}


# ---------------------------------------------------------------------------
# Contact-map head & composition model
# ---------------------------------------------------------------------------

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


class CompositionContactMapModel(nn.Module):
    """Composition of LMs with per-residue contact-map output.

    DNA context (both strands concatenated) is injected into protein
    encoder layers via cross-attention at ``target_layers``.

    Works with any protein family that has an entry in
    ``_ENCODER_ADAPTERS`` (currently ESM2 and ESM-C).
    """

    def __init__(self, dna_lm, prot_lm, dna_emb_dim: int, prot_emb_dim: int,
                 num_heads: int = 20, head_dim: int = 64,
                 target_layers: list[int] | None = None,
                 esm_layers: int = 6,
                 prot_family: str = "esm2",
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.dna_lm = dna_lm
        self.prot_lm = prot_lm
        self.target_layers = target_layers or [0, 3, 5]
        self.esm_layers = esm_layers
        self.gradient_checkpointing = gradient_checkpointing

        adapter_cls = _ENCODER_ADAPTERS.get(prot_family)
        if adapter_cls is None:
            raise ValueError(
                f"No encoder adapter for family '{prot_family}'. "
                f"Supported: {sorted(_ENCODER_ADAPTERS)}"
            )
        self._enc = adapter_cls(prot_lm)

        self.dna_proj = nn.Linear(dna_emb_dim, prot_emb_dim)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(prot_emb_dim, num_heads, batch_first=True)
            for _ in self.target_layers
        ])
        self.post_attn_norms = nn.ModuleList([
            nn.LayerNorm(prot_emb_dim) for _ in self.target_layers
        ])
        self.contact_head = ContactMapHead(prot_emb_dim, prot_emb_dim,
                                           head_dim)

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

    def forward(self, batch):
        dna1_h = self._dna_hidden(batch["dna1_input_ids"],
                                  batch["dna1_attention_mask"])
        dna2_h = self._dna_hidden(batch["dna2_input_ids"],
                                  batch["dna2_attention_mask"])

        dna_cat = torch.cat([dna1_h, dna2_h], dim=1)
        dna_cat_am = torch.cat([batch["dna1_attention_mask"],
                                batch["dna2_attention_mask"]], dim=1)
        dna_cat_kpm = ~dna_cat_am.bool()
        all_masked = dna_cat_kpm.all(dim=1)
        if all_masked.any():
            dna_cat_kpm = dna_cat_kpm.clone()
            dna_cat_kpm[all_masked, 0] = False

        prot_am = batch["prot_attention_mask"]
        prot_mask = self._enc.prepare_mask(prot_am, batch["prot_input_ids"])
        prot = self._enc.get_embeddings(batch["prot_input_ids"], prot_am)

        counter = 0
        for i in range(self.esm_layers):
            if self.gradient_checkpointing and self.training:
                prot = checkpoint(
                    lambda h, m, idx=i: self._enc.apply_layer(idx, h, m),
                    prot, prot_mask, use_reentrant=True,
                )
            else:
                prot = self._enc.apply_layer(i, prot, prot_mask)
            if i in self.target_layers:
                # Cross-attention is float32; cast inputs to match (encoder may be bf16)
                q = prot.float()
                kv = dna_cat.float()
                attn_out, _ = self.cross_attn_layers[counter](
                    query=q, key=kv, value=kv,
                    key_padding_mask=dna_cat_kpm,
                )
                prot = prot + self.post_attn_norms[counter](attn_out).to(prot.dtype)
                counter += 1

        prot = self._enc.final_norm(prot)
        prot_f = prot.float()

        cm1 = self.contact_head(prot_f, dna1_h)
        cm2 = self.contact_head(prot_f, dna2_h)
        return torch.stack([cm1, cm2], dim=1)


def build_model(dna_lm, prot_lm, dna_info: dict, prot_info: dict,
                arch_cfg: dict, device: str = "cpu"):
    """Construct a :class:`CompositionContactMapModel`.

    Validates that the protein model family is supported and that
    ``target_layers`` indices are within the model's layer count.
    """
    prot_family = prot_info.get("family")
    if prot_family not in _ENCODER_ADAPTERS:
        raise ValueError(
            f"Composition framework requires a supported protein model "
            f"(got family='{prot_family}'). "
            f"Supported families: {sorted(_ENCODER_ADAPTERS)}. "
            f"Use the attention framework for other protein LMs."
        )

    esm_layers = prot_info["layers"]
    prot_emb_dim = prot_info["emb_dim"]
    target_layers = arch_cfg.get("target_layers", [0, 3, 5])

    if max(target_layers) >= esm_layers:
        raise ValueError(
            f"target_layers {target_layers} has index >= esm_layers "
            f"({esm_layers}). Adjust target_layers for this model."
        )

    num_heads = arch_cfg.get("num_heads", 20)
    if prot_emb_dim % num_heads != 0:
        # Find the largest divisor of prot_emb_dim that is <= num_heads
        compatible = max(h for h in range(1, num_heads + 1)
                         if prot_emb_dim % h == 0)
        raise ValueError(
            f"num_heads={num_heads} does not divide prot_emb_dim={prot_emb_dim} "
            f"(protein model: {prot_info.get('hf_name', prot_family)}). "
            f"Set num_heads to a divisor of {prot_emb_dim}, "
            f"e.g. num_heads={compatible}."
        )

    return CompositionContactMapModel(
        dna_lm, prot_lm,
        dna_emb_dim=dna_info["emb_dim"],
        prot_emb_dim=prot_emb_dim,
        num_heads=num_heads,
        head_dim=arch_cfg.get("head_dim", 64),
        target_layers=target_layers,
        esm_layers=esm_layers,
        prot_family=prot_family,
        gradient_checkpointing=arch_cfg.get("gradient_checkpointing", False),
    ).to(device)
