"""Shared utilities for protein-DNA contact-map prediction training."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

_DEFAULT_DATA_ROOT = "/home/zcorn/Projects/proteinDNA_data"


def resolve_data_path(path: str) -> str:
    """Replace the default data root with ``$DATA_ROOT`` if set.

    Allows the same YAML configs to work on both local and remote machines
    by overriding the data prefix via the ``DATA_ROOT`` environment variable.
    """
    data_root = os.environ.get("DATA_ROOT")
    if data_root and path.startswith(_DEFAULT_DATA_ROOT):
        return path.replace(_DEFAULT_DATA_ROOT, data_root, 1)
    return path


def load_split_and_resolve_data(cfg: dict) -> tuple:
    """Load split file and resolve data_pt from its ``source_pt`` metadata.

    If ``data.data_pt`` is present in *cfg* it is used directly; otherwise
    the split file's ``source_pt`` key provides the path.  Both paths go
    through :func:`resolve_data_path` for ``$DATA_ROOT`` portability.

    Returns ``(data, data_spec, split, split_spec)``.
    """
    import torch

    split_path = resolve_data_path(cfg["data"]["split_pt"])
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    split_spec = split["spec"]

    data_pt = cfg["data"].get("data_pt")
    if not data_pt:
        data_pt = split.get("source_pt")
        if not data_pt:
            raise ValueError(
                "data_pt not in config and split file has no 'source_pt' key"
            )
    data_path = resolve_data_path(data_pt)

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    data_spec = data["spec"]

    print(f"Split:  {split_path}")
    print(f"Data:   {data_path}")
    return data, data_spec, split, split_spec


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def mask_special_tokens(attention_mask: torch.Tensor) -> torch.Tensor:
    """Zero out position 0 (BOS/CLS) and the last non-pad position (EOS/SEP).

    Works for both 1-D ``(L,)`` and 2-D ``(N, L)`` attention masks.
    Returns a **new** tensor (no in-place modification).
    """
    mask = attention_mask.clone()
    squeezed = mask.dim() == 1
    if squeezed:
        mask = mask.unsqueeze(0)
    mask[:, 0] = 0
    orig = attention_mask if attention_mask.dim() == 2 else attention_mask.unsqueeze(0)
    last_idx = orig.sum(dim=1).long() - 1
    rows = torch.arange(mask.size(0), device=mask.device)
    mask[rows, last_idx.clamp(min=0)] = 0
    return mask.squeeze(0) if squeezed else mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_bce_loss(logits, targets, prot_mask, dna_mask):
    """Masked BCE-with-logits loss over contact-map logits.

    Parameters
    ----------
    logits    : (B, 2, R, L)
    targets   : (B, 2, R, L)
    prot_mask : (B, R)   -- 1 at valid protein positions
    dna_mask  : (B, 2, L) -- 1 at valid DNA positions per strand
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )
    mask = prot_mask[:, None, :, None].float() * dna_mask[:, :, None, :].float()
    return (bce * mask).sum() / mask.sum().clamp(min=1.0)


def masked_focal_loss(logits, targets, prot_mask, dna_mask,
                      alpha: float = 0.95, gamma: float = 2.0):
    """Masked focal loss over contact-map logits.

    Focal loss down-weights well-classified (easy) negatives so the model
    focuses on hard positives, counteracting extreme class imbalance.

    Parameters
    ----------
    logits    : (B, 2, R, L)
    targets   : (B, 2, R, L)
    prot_mask : (B, R)   -- 1 at valid protein positions
    dna_mask  : (B, 2, L) -- 1 at valid DNA positions per strand
    alpha     : Balance weight for the positive class (1-alpha for negatives).
    gamma     : Focusing exponent; higher values down-weight easy examples more.
    """
    # Numerically stable BCE from logits
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma

    mask = prot_mask[:, None, :, None].float() * dna_mask[:, :, None, :].float()
    return (focal_weight * bce * mask).sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def flatten_valid(logits, targets, prot_mask, dna_mask):
    """Flatten logits / targets at valid mask positions to 1-D numpy arrays.

    Returns ``(y_true_1d, y_score_1d)`` as float32 numpy arrays.
    """
    mask = (
        prot_mask[:, None, :, None].float() * dna_mask[:, :, None, :].float()
    ).bool()
    scores = torch.sigmoid(logits)[mask].detach().cpu().numpy()
    labels = targets[mask].detach().cpu().numpy()
    return labels.astype(np.float32), scores.astype(np.float32)


def compute_contactmap_metrics(y_true_1d, y_score_1d, thresh=0.5,
                               n_positives_for_topL=None):
    """Compute metrics for contact-map predictions.

    Returns dict with keys: ``pr_auc``, ``roc_auc``, ``mcc``,
    ``balanced_accuracy``, ``precision``, ``recall``, ``f1``,
    ``top_L_precision``.
    """
    y_score_1d = np.asarray(y_score_1d, dtype=np.float64)
    y_score_1d = np.nan_to_num(y_score_1d, nan=0.0, posinf=1.0, neginf=0.0)
    y_score_1d = np.clip(y_score_1d, 0.0, 1.0)

    y_int = y_true_1d.astype(int)
    y_pred = (y_score_1d >= thresh).astype(int)
    both_classes = len(np.unique(y_int)) > 1

    pr_auc = average_precision_score(y_int, y_score_1d) if both_classes else 0.0
    roc_auc = roc_auc_score(y_int, y_score_1d) if both_classes else 0.0
    mcc = matthews_corrcoef(y_int, y_pred)

    L = n_positives_for_topL if n_positives_for_topL else int(y_int.sum())
    if L > 0 and len(y_score_1d) > 0:
        top_idx = np.argsort(y_score_1d)[-L:]
        top_L_prec = y_int[top_idx].sum() / L
    else:
        top_L_prec = 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy_score(y_int, y_pred),
        "precision": precision_score(y_int, y_pred, zero_division=0),
        "recall": recall_score(y_int, y_pred, zero_division=0),
        "f1": f1_score(y_int, y_pred, zero_division=0),
        "top_L_precision": float(top_L_prec),
    }


# ---------------------------------------------------------------------------
# Data subsetting
# ---------------------------------------------------------------------------

def subset_data(data: dict, indices) -> dict:
    """Subset a precomputed data dict by sample indices.

    Compatible with :class:`ContactMapDataset` input format.
    ``data`` must have top-level keys ``dna1``, ``dna2``, ``prot``
    (each a dict of tensors) and ``contact_maps`` (a list).
    """
    idx = list(indices)
    return {
        "dna1": {k: v[idx] for k, v in data["dna1"].items()},
        "dna2": {k: v[idx] for k, v in data["dna2"].items()},
        "prot": {k: v[idx] for k, v in data["prot"].items()},
        "contact_maps": [data["contact_maps"][i] for i in idx],
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContactMapDataset(Dataset):
    """PyTorch dataset that builds padded Y / masks on the fly.

    Raw contact maps are stored as compact int8 arrays (variable size per
    sample).  Each ``__getitem__`` call pads the map into a dense
    ``(2, R, L)`` float32 tensor, which keeps total memory proportional to
    the actual data rather than ``N * 2 * R * L``.

    Expects a data dict with keys ``dna1``, ``dna2``, ``prot`` (each
    containing ``input_ids`` and ``attention_mask`` tensors) and
    ``contact_maps`` (a list of ``{"cm1": np.array, "cm2": ...}``).

    Parameters
    ----------
    split_data : dict
        Subset of a precomputed data dict (see :func:`subset_data`).
    dna_has_special_tokens : bool
        If ``True`` (default, DNABERT-2), the DNA tokenizer prepends
        CLS and appends SEP, so contact-map labels start at position 1
        and DNA masks go through :func:`mask_special_tokens`.
        If ``False`` (NTv3), the DNA tokens are plain nucleotides with
        no CLS/SEP, so labels start at position 0 and the raw
        ``attention_mask`` is used directly.
    """

    def __init__(self, split_data: dict,
                 dna_has_special_tokens: bool = True):
        self.dna1_ids = split_data["dna1"]["input_ids"]
        self.dna2_ids = split_data["dna2"]["input_ids"]
        self.prot_ids = split_data["prot"]["input_ids"]
        self.prot_am = split_data["prot"]["attention_mask"]

        # Some tokenizers (e.g. NTv3) don't produce attention_mask;
        # fall back to deriving it from the pad_token_id (usually 1).
        def _get_am(d: dict) -> torch.Tensor:
            if "attention_mask" in d:
                return d["attention_mask"]
            pad_id = 1  # NTv3 pad token id
            return (d["input_ids"] != pad_id).long()

        self.dna1_am = _get_am(split_data["dna1"])
        self.dna2_am = _get_am(split_data["dna2"])
        self.contact_maps = split_data["contact_maps"]
        self.R = self.prot_ids.size(1)
        self.L = self.dna1_ids.size(1)
        self.dna_has_special_tokens = dna_has_special_tokens

    def __len__(self):
        return len(self.contact_maps)

    def __getitem__(self, idx):
        R, L = self.R, self.L
        dna_off = 1 if self.dna_has_special_tokens else 0

        Y = torch.zeros(2, R, L)
        cm = self.contact_maps[idx]
        cm1 = cm["cm1"]
        rend = min(1 + cm1.shape[0], R)
        lend1 = min(dna_off + cm1.shape[1], L)
        Y[0, 1:rend, dna_off:lend1] = torch.from_numpy(
            cm1[: rend - 1, : lend1 - dna_off].astype(np.float32),
        )
        cm2 = cm["cm2"]
        if cm2 is not None:
            lend2 = min(dna_off + cm2.shape[1], L)
            Y[1, 1:rend, dna_off:lend2] = torch.from_numpy(
                cm2[: rend - 1, : lend2 - dna_off].astype(np.float32),
            )

        prot_mask = mask_special_tokens(self.prot_am[idx])
        if self.dna_has_special_tokens:
            dna_mask = torch.stack([
                mask_special_tokens(self.dna1_am[idx]),
                mask_special_tokens(self.dna2_am[idx]),
            ])
        else:
            dna_mask = torch.stack([
                self.dna1_am[idx],
                self.dna2_am[idx],
            ])

        return {
            "dna1_input_ids": self.dna1_ids[idx],
            "dna1_attention_mask": self.dna1_am[idx],
            "dna2_input_ids": self.dna2_ids[idx],
            "dna2_attention_mask": self.dna2_am[idx],
            "prot_input_ids": self.prot_ids[idx],
            "prot_attention_mask": self.prot_am[idx],
            "Y": Y,
            "dna_mask": dna_mask,
            "prot_mask": prot_mask,
        }
