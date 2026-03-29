#!/usr/bin/env python3
"""Diagnose trained composition contact-map models.

Loads a checkpoint and runs inference on the validation set, extracting:
  - Cross-attention weight distributions (padding fraction, entropy,
    correlation with true contacts)
  - Per-sample metrics (PR-AUC vs sequence length, contact count)
  - Score calibration (histogram of predictions for pos/neg cells)
  - Failure-mode characterisation of worst-performing samples

Usage
-----
    python scripts/analysis/attention_diagnostics.py \
        --config configs/training/composition_contactmap_v2.yaml \
        --checkpoint results/my_model_best.pth \
        --output_dir analysis_output/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from biollmcomposition.frameworks.composition import build_model
from biollmcomposition.models import get_model_info, load_model
from biollmcomposition.utils.contact_map import (
    ContactMapDataset,
    compute_contactmap_metrics,
    flatten_valid,
    load_split_and_resolve_data,
    mask_special_tokens,
    set_seed,
    subset_data,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./analysis_output")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of val samples (for quick testing)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Attention extraction via hooks
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Register forward hooks on cross-attention layers to capture weights."""

    def __init__(self, model):
        self.weights: dict[int, torch.Tensor] = {}
        self._hooks = []
        for idx, layer in enumerate(model.cross_attn_layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # when average_attn_weights=True (default): (B, T_q, T_k)
            if isinstance(output, tuple) and len(output) >= 2:
                w = output[1]
                if w is not None:
                    self.weights[layer_idx] = w.detach().cpu()
        return hook_fn

    def clear(self):
        self.weights.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _enable_attn_weights(model):
    """Temporarily set need_weights=True on all cross-attention layers."""
    originals = {}
    for idx, layer in enumerate(model.cross_attn_layers):
        # PyTorch MHA forward signature checks the `need_weights` kwarg;
        # we patch forward to always pass need_weights=True.
        original_forward = layer.forward
        originals[idx] = original_forward

        def patched_forward(query, key, value, *args,
                            _orig=original_forward, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return _orig(query, key, value, *args, **kwargs)

        layer.forward = patched_forward
    return originals


def _restore_attn_weights(model, originals):
    for idx, layer in enumerate(model.cross_attn_layers):
        if idx in originals:
            layer.forward = originals[idx]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_attn_stats(attn_weights: torch.Tensor,
                       dna_cat_mask: torch.Tensor) -> dict:
    """Compute attention statistics for one batch.

    Parameters
    ----------
    attn_weights : (B, R, 2L) — averaged attention weights.
    dna_cat_mask : (B, 2L) — 1 at real DNA positions, 0 at padding.
    """
    B = attn_weights.size(0)
    stats = defaultdict(list)

    for i in range(B):
        w = attn_weights[i]  # (R, 2L)
        m = dna_cat_mask[i].bool()  # (2L,)
        n_real = m.sum().item()
        n_total = m.size(0)

        if n_real == 0 or n_total == 0:
            continue

        mass_on_real = w[:, m].sum(dim=-1).mean().item()
        stats["attn_mass_on_real"].append(mass_on_real)

        # Attention entropy (over key dimension)
        w_clamped = w.clamp(min=1e-10)
        entropy = -(w_clamped * w_clamped.log()).sum(dim=-1).mean().item()
        max_entropy = np.log(n_real)
        stats["attn_entropy"].append(entropy)
        stats["attn_entropy_normalised"].append(
            entropy / max_entropy if max_entropy > 0 else 0.0)
        stats["n_real_dna_positions"].append(n_real)

    return dict(stats)


def per_sample_metrics(logits, targets, prot_mask, dna_mask) -> list[dict]:
    """Compute metrics for each sample individually."""
    B = logits.size(0)
    results = []
    for i in range(B):
        yt, ys = flatten_valid(
            logits[i:i+1], targets[i:i+1],
            prot_mask[i:i+1], dna_mask[i:i+1],
        )
        n_pos = int(yt.sum())
        n_total = len(yt)
        if n_pos == 0 or n_pos == n_total:
            m = {"pr_auc": 0.0, "n_pos": n_pos, "n_total": n_total}
        else:
            m = compute_contactmap_metrics(yt, ys)
            m["n_pos"] = n_pos
            m["n_total"] = n_total
        results.append(m)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histograms(pos_scores, neg_scores, out_path):
    """Score distribution for positive vs negative cells."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(neg_scores, bins=100, alpha=0.6, label=f"Neg (n={len(neg_scores):,})",
            density=True, color="steelblue")
    ax.hist(pos_scores, bins=100, alpha=0.7, label=f"Pos (n={len(pos_scores):,})",
            density=True, color="coral")
    ax.set_xlabel("Predicted score (sigmoid)")
    ax.set_ylabel("Density")
    ax.set_title("Score calibration: positive vs negative cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_auc_vs_feature(sample_metrics, feature_key, xlabel, out_path):
    """Scatter plot of per-sample PR-AUC vs a feature."""
    xs = [m[feature_key] for m in sample_metrics if m.get("pr_auc", 0) > 0]
    ys = [m["pr_auc"] for m in sample_metrics if m.get("pr_auc", 0) > 0]
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.3, s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Per-sample PR-AUC vs {xlabel}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attn_stats(all_stats, out_dir):
    """Plot attention diagnostic histograms."""
    for key in ("attn_mass_on_real", "attn_entropy_normalised"):
        vals = all_stats.get(key, [])
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(vals, bins=50, alpha=0.7, color="teal")
        ax.set_xlabel(key.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {key}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Load data ---------------------------------------------------------
    data, data_spec, split, split_spec = load_split_and_resolve_data(cfg)
    dna_short = data_spec["dna_model_short"]
    prot_short = data_spec["prot_model_short"]
    dna_info = get_model_info(dna_short)
    prot_info = get_model_info(prot_short)

    val_data = subset_data(data, split["val_idx"])
    if args.max_samples and args.max_samples < len(split["val_idx"]):
        val_data = subset_data(data, split["val_idx"][:args.max_samples])

    dna_special = dna_info.get("has_special_tokens", True)
    val_ds = ContactMapDataset(val_data, dna_has_special_tokens=dna_special)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    # -- Load model --------------------------------------------------------
    print(f"Loading DNA model: {dna_info['hf_name']}")
    dna_lm = load_model(dna_short, device=str(device))
    print(f"Loading Protein model: {prot_info['hf_name']}")
    prot_lm = load_model(prot_short, device=str(device))

    a = cfg.get("architecture", {})
    model = build_model(dna_lm, prot_lm, dna_info, prot_info, a,
                        device=str(device))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # -- Set up attention capture ------------------------------------------
    originals = _enable_attn_weights(model)
    capture = AttentionCapture(model)

    # -- Run inference -----------------------------------------------------
    all_sample_metrics = []
    all_attn_stats = defaultdict(list)
    pos_scores_all, neg_scores_all = [], []

    print(f"Running inference on {len(val_ds)} samples...")
    for batch in val_loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        capture.clear()
        logits = model(batch_dev)

        # -- Per-sample metrics
        sm = per_sample_metrics(logits, batch_dev["Y"],
                                batch_dev["prot_mask"], batch_dev["dna_mask"])
        prot_lens = batch["prot_attention_mask"].sum(dim=1).tolist()
        dna1_lens = batch["dna1_attention_mask"].sum(dim=1).tolist()
        dna2_lens = batch["dna2_attention_mask"].sum(dim=1).tolist()
        for j, m in enumerate(sm):
            m["prot_len"] = prot_lens[j]
            m["dna_len"] = max(dna1_lens[j], dna2_lens[j])
            m["padding_ratio"] = (
                (val_ds.R * val_ds.L * 2) /
                max(m["prot_len"] * m["dna_len"] * 2, 1)
            )
        all_sample_metrics.extend(sm)

        # -- Score distributions
        mask = (
            batch_dev["prot_mask"][:, None, :, None].float()
            * batch_dev["dna_mask"][:, :, None, :].float()
        ).bool()
        scores = torch.sigmoid(logits)[mask].cpu().numpy()
        labels = batch_dev["Y"][mask].cpu().numpy()
        pos_scores_all.append(scores[labels > 0.5])
        neg_scores_all.append(scores[labels <= 0.5])

        # -- Attention stats (use last cross-attention layer)
        if capture.weights:
            last_layer = max(capture.weights.keys())
            w = capture.weights[last_layer]
            dna_cat_am = torch.cat([batch["dna1_attention_mask"],
                                    batch["dna2_attention_mask"]], dim=1)
            stats = compute_attn_stats(w, dna_cat_am)
            for k, v in stats.items():
                all_attn_stats[k].extend(v)

    capture.remove()
    _restore_attn_weights(model, originals)

    # -- Aggregate results -------------------------------------------------
    pos_scores = np.concatenate(pos_scores_all) if pos_scores_all else np.array([])
    neg_scores = np.concatenate(neg_scores_all) if neg_scores_all else np.array([])

    # -- Plots -------------------------------------------------------------
    print("Generating plots...")
    plot_histograms(pos_scores, neg_scores, out_dir / "score_calibration.png")
    plot_pr_auc_vs_feature(all_sample_metrics, "prot_len",
                           "Protein length", out_dir / "prauc_vs_prot_len.png")
    plot_pr_auc_vs_feature(all_sample_metrics, "dna_len",
                           "DNA length (max strand)",
                           out_dir / "prauc_vs_dna_len.png")
    plot_pr_auc_vs_feature(all_sample_metrics, "n_pos",
                           "Number of contacts",
                           out_dir / "prauc_vs_n_contacts.png")
    plot_pr_auc_vs_feature(all_sample_metrics, "padding_ratio",
                           "Padding ratio (padded/actual)",
                           out_dir / "prauc_vs_padding_ratio.png")
    plot_attn_stats(dict(all_attn_stats), out_dir)

    # -- Failure mode summary ----------------------------------------------
    valid_samples = [m for m in all_sample_metrics if m.get("pr_auc", 0) > 0]
    if valid_samples:
        valid_samples.sort(key=lambda m: m["pr_auc"])
        n_worst = min(20, len(valid_samples))
        worst = valid_samples[:n_worst]

        print(f"\n--- Worst {n_worst} samples by PR-AUC ---")
        for i, m in enumerate(worst):
            print(f"  {i+1}. PR-AUC={m['pr_auc']:.4f}  "
                  f"prot_len={m['prot_len']}  dna_len={m['dna_len']}  "
                  f"n_contacts={m['n_pos']}  n_cells={m['n_total']}")

        summary = {
            "n_samples": len(all_sample_metrics),
            "n_with_contacts": len(valid_samples),
            "median_pr_auc": float(np.median([m["pr_auc"] for m in valid_samples])),
            "mean_pr_auc": float(np.mean([m["pr_auc"] for m in valid_samples])),
            "worst_20_median_prot_len": float(np.median([m["prot_len"] for m in worst])),
            "worst_20_median_dna_len": float(np.median([m["dna_len"] for m in worst])),
            "worst_20_median_n_contacts": float(np.median([m["n_pos"] for m in worst])),
        }
    else:
        summary = {"n_samples": len(all_sample_metrics), "n_with_contacts": 0}

    # Attention summary
    for k in ("attn_mass_on_real", "attn_entropy_normalised"):
        vals = all_attn_stats.get(k, [])
        if vals:
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"median_{k}"] = float(np.median(vals))

    # -- Save summary ------------------------------------------------------
    summary_path = out_dir / "diagnostics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save per-sample metrics
    sample_path = out_dir / "per_sample_metrics.json"
    with open(sample_path, "w") as f:
        json.dump(all_sample_metrics, f, indent=2)
    print(f"Per-sample metrics saved to {sample_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
