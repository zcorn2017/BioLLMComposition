#!/usr/bin/env python3
"""Unified training entry point for contact-map prediction models.

Loads precomputed tokens and a split file, then trains the model
specified by the ``framework`` key in the config.  Base model
information (DNA / Protein LM names and dims) is read from the data
file's metadata — **not** hardcoded in the training config — so there
is never a mismatch between tokenized data and the LMs used.

Supports both BCE and focal loss (via the ``loss:`` config section),
and optional cosine-annealing LR with linear warmup.

Usage
-----
    python scripts/02_run_training.py \
        --config configs/training/attention_contactmap.yaml

    # Override any value via CLI:
    python scripts/02_run_training.py \
        --config configs/training/composition_contactmap_focal_loss.yaml \
        --lr 1e-4 --batch_size 8 --focal_alpha 0.9
"""

from __future__ import annotations

import argparse
import datetime
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from biollmcomposition.frameworks import get_framework
from biollmcomposition.models import get_model_info, load_model
from biollmcomposition.utils.wandb_logger import (
    init_run, log_scalars, log_best_metrics,
    log_source_artifacts, log_checkpoint, finish,
)
from biollmcomposition.utils.contact_map import (
    ContactMapDataset,
    compute_contactmap_metrics,
    flatten_valid,
    masked_bce_loss,
    masked_focal_loss,
    load_split_and_resolve_data,
    set_seed,
    subset_data,
)


# ── CLI / config ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--data_pt", type=str, default=None)
    p.add_argument("--split_pt", type=str, default=None)
    p.add_argument("--framework", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--focal_alpha", type=float, default=None)
    p.add_argument("--focal_gamma", type=float, default=None)
    p.add_argument("--description", type=str, default=None,
                   help="Free-text run description logged to W&B")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    return p.parse_args()


def load_config(args) -> dict:
    """Merge YAML config with CLI overrides (CLI wins)."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cli = vars(args)
    if cli.get("framework"):
        cfg["framework"] = cli["framework"]
    for key in ("data_pt", "split_pt"):
        if cli.get(key):
            cfg.setdefault("data", {})[key] = cli[key]
    for key in ("lr", "epochs", "batch_size", "num_workers", "n_runs", "seed",
                "warmup_epochs"):
        if cli.get(key) is not None:
            cfg.setdefault("training", {})[key] = cli[key]
    for key in ("focal_alpha", "focal_gamma"):
        if cli.get(key) is not None:
            cfg.setdefault("loss", {})[key] = cli[key]
    if cli.get("description"):
        cfg["description"] = cli["description"]
    for key in ("log_dir", "save_dir"):
        if cli.get(key):
            cfg.setdefault("output", {})[key] = cli[key]
    return cfg


# ── Loss ──────────────────────────────────────────────────────────────────

def build_loss_fn(loss_cfg: dict):
    """Return a loss callable and a short tag string from the config.

    If the config has a ``loss:`` section with ``focal_alpha`` or
    ``focal_gamma``, returns :func:`masked_focal_loss` with those params.
    Otherwise returns :func:`masked_bce_loss`.
    """
    if loss_cfg.get("focal_alpha") is not None or loss_cfg.get("focal_gamma") is not None:
        alpha = loss_cfg.get("focal_alpha", 0.95)
        gamma = loss_cfg.get("focal_gamma", 2.0)
        fn = partial(masked_focal_loss, alpha=alpha, gamma=gamma)
        tag = f"focal_a{alpha}_g{gamma}"
        return fn, tag
    return masked_bce_loss, "bce"


# ── LR schedule ──────────────────────────────────────────────────────────

def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int,
                           min_lr_ratio: float = 0.01):
    """Linear warmup for ``warmup_steps``, then cosine decay to
    ``min_lr_ratio * base_lr`` over the remaining steps."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
    return LambdaLR(optimizer, lr_lambda)


# ── Training / evaluation ────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, loss_fn,
                    scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = loss_fn(logits, batch["Y"],
                       batch["prot_mask"], batch["dna_mask"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * batch["Y"].size(0)
        n += batch["Y"].size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_true, all_score = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        total_loss += (
            loss_fn(logits, batch["Y"],
                    batch["prot_mask"], batch["dna_mask"]).item()
            * batch["Y"].size(0)
        )
        n += batch["Y"].size(0)
        yt, ys = flatten_valid(logits, batch["Y"],
                               batch["prot_mask"], batch["dna_mask"])
        all_true.append(yt)
        all_score.append(ys)
    metrics = compute_contactmap_metrics(
        np.concatenate(all_true), np.concatenate(all_score),
    )
    metrics["val_loss"] = total_loss / max(n, 1)
    return metrics


# ── Run name builder ──────────────────────────────────────────────────────

_FW_SHORT = {
    "attention": "attn_cm",
    "composition": "comp_cm",
}


def build_run_tag(framework_name: str, dna_short: str, prot_short: str,
                  split_spec: dict, training_cfg: dict,
                  arch_cfg: dict, loss_tag: str, timestamp: str) -> str:
    """Build a descriptive, filesystem-safe W&B run name."""
    fw = _FW_SHORT.get(framework_name, framework_name)
    split_tag = split_spec.get("strategy", "unk").replace("_", "")

    parts = [
        fw,
        dna_short,
        prot_short,
        split_tag,
        f"lr{training_cfg.get('lr', 5e-5)}",
        f"bs{training_cfg.get('batch_size', 16)}",
        f"hd{arch_cfg.get('head_dim', 64)}",
    ]
    if framework_name == "composition":
        parts.append(f"nh{arch_cfg.get('num_heads', 20)}")
        tl = arch_cfg.get("target_layers", [0, 3, 5])
        parts.append(f"tl{''.join(str(x) for x in tl)}")

    parts.append(loss_tag)
    parts.append(timestamp)
    return "_".join(str(p) for p in parts)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args)

    framework_name = cfg.get("framework", "attention")
    t = cfg.get("training", {})
    a = cfg.get("architecture", {})
    o = cfg.get("output", {})
    loss_cfg = cfg.get("loss", {})

    seed = t.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Loss function (BCE or focal) ──
    loss_fn, loss_tag = build_loss_fn(loss_cfg)

    print(f"Device:    {device}")
    print(f"Framework: {framework_name}")
    print(f"Loss:      {loss_tag}")
    print(f"Config:    {cfg}")

    # ── Load precomputed tokens ──
    data, data_spec, split, split_spec = load_split_and_resolve_data(cfg)

    # ── Model info (read from data, NOT from config) ──
    dna_short = data_spec["dna_model_short"]
    prot_short = data_spec["prot_model_short"]
    dna_info = get_model_info(dna_short)
    prot_info = get_model_info(prot_short)

    print(f"\nBase models (from data metadata):")
    print(f"  DNA:     {dna_info['hf_name']}  ({dna_short})")
    print(f"  Protein: {prot_info['hf_name']}  ({prot_short})")
    print(f"Split:     {split_spec['strategy']}, "
          f"train={split_spec['n_train']}, val={split_spec['n_val']}, "
          f"test={split_spec['n_test']}")

    # ── Subset by split indices ──
    train_data = subset_data(data, split["train_idx"])
    val_data = subset_data(data, split["val_idx"])

    dna_special = dna_info.get("has_special_tokens", True)
    nw = t.get("num_workers", 4)
    bs = t.get("batch_size", 16)
    train_loader = DataLoader(
        ContactMapDataset(train_data, dna_has_special_tokens=dna_special),
        batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        ContactMapDataset(val_data, dna_has_special_tokens=dna_special),
        batch_size=bs,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )

    # ── Frozen LMs ──
    print(f"\nLoading DNA model:     {dna_info['hf_name']}")
    dna_lm = load_model(dna_short, device=str(device))
    print(f"Loading Protein model: {prot_info['hf_name']}")
    prot_lm = load_model(prot_short, device=str(device))

    # ── Framework ──
    framework = get_framework(framework_name)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = build_run_tag(framework_name, dna_short, prot_short,
                            split_spec, t, a, loss_tag, ts)

    save_dir = Path(o.get("save_dir", "./results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    lr = t.get("lr", 5e-5)
    epochs = t.get("epochs", 100)
    n_runs = t.get("n_runs", 1)

    # ── LR scheduler config ──
    warmup_epochs = t.get("warmup_epochs", 0)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    use_scheduler = warmup_epochs > 0

    if use_scheduler:
        print(f"LR schedule: warmup={warmup_epochs} epochs ({warmup_steps} steps), "
              f"total={epochs} epochs ({total_steps} steps)")

    # ── Training runs ──
    for run in range(n_runs):
        print(f"\n{'=' * 70}")
        print(f"Run {run + 1}/{n_runs}  [{run_tag}]")
        print("=" * 70)

        init_run(cfg, run_tag, run, framework_name,
                 dna_short=dna_short, prot_short=prot_short,
                 loss_tag=loss_tag)
        log_source_artifacts(framework, dna_info, prot_info,
                             __file__, args.config)

        model = framework.build_model(
            dna_lm, prot_lm, dna_info, prot_info, a, device=str(device),
        )
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr,
        )
        scheduler = (warmup_cosine_schedule(optimizer, warmup_steps, total_steps)
                     if use_scheduler else None)

        best_pr_auc = -1.0
        best_metrics: dict = {}
        ckpt_path = save_dir / f"{run_tag}_run{run}_best.pth"

        for epoch in tqdm(range(epochs), desc=f"Run {run}"):
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         device, loss_fn, scheduler)
            metrics = evaluate(model, val_loader, device, loss_fn)

            cur_lr = optimizer.param_groups[0]["lr"] if scheduler else None
            log_scalars(epoch, train_loss, metrics, lr=cur_lr)

            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_metrics = metrics
                torch.save(model.state_dict(), ckpt_path)

        log_best_metrics(best_metrics)
        log_checkpoint(ckpt_path)
        finish()
        print(
            f"  Best: PR-AUC={best_metrics.get('pr_auc', 0):.4f}, "
            f"ROC-AUC={best_metrics.get('roc_auc', 0):.4f}, "
            f"MCC={best_metrics.get('mcc', 0):.4f}, "
            f"P={best_metrics.get('precision', 0):.4f}, "
            f"R={best_metrics.get('recall', 0):.4f}, "
            f"F1={best_metrics.get('f1', 0):.4f}, "
            f"Top-L={best_metrics.get('top_L_precision', 0):.4f}"
        )


if __name__ == "__main__":
    main()
