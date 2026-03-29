#!/usr/bin/env python3
"""Composition contact-map trainer v2: dynamic batching + refined head.

Extends the focal-loss trainer with:
  - Length-bucketed dynamic batching (reduces ~33x padding overhead)
  - Optional 2D-conv refined contact head
  - Optional positive-enriched batch sampling

The original ``composition_contactmap_focal_loss.py`` is left untouched
as the stable baseline.

Usage
-----
    python scripts/frameworks/composition_contactmap_v2.py \
        --config configs/training/composition_contactmap_v2.yaml
"""

from __future__ import annotations

import argparse
import datetime
import functools
import importlib
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from biollmcomposition.frameworks.composition import (
    CompositionContactMapModel,
    build_model,
)
from biollmcomposition.models import get_model_info, load_model
from biollmcomposition.utils.wandb_logger import (
    init_run, log_scalars, log_best_metrics,
    log_source_artifacts, log_checkpoint, finish,
)
from biollmcomposition.utils.contact_map import (
    ContactMapDataset,
    VariableLengthContactMapDataset,
    BucketBatchSampler,
    collate_contactmap_batch,
    compute_contactmap_metrics,
    flatten_valid,
    masked_focal_loss,
    load_split_and_resolve_data,
    set_seed,
    subset_data,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data_pt", type=str, default=None)
    p.add_argument("--split_pt", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--focal_alpha", type=float, default=None)
    p.add_argument("--focal_gamma", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--contact_head", type=str, default=None,
                   choices=["bilinear", "refined"])
    p.add_argument("--dynamic_batching", action="store_true", default=None)
    p.add_argument("--no_dynamic_batching", dest="dynamic_batching",
                   action="store_false")
    p.add_argument("--description", type=str, default=None,
                   help="Free-text run description logged to W&B")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    return p.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cli = vars(args)
    for key in ("data_pt", "split_pt"):
        if cli.get(key):
            cfg.setdefault("data", {})[key] = cli[key]
    for key in ("lr", "epochs", "batch_size", "num_workers", "n_runs", "seed",
                "warmup_epochs"):
        if cli.get(key) is not None:
            cfg.setdefault("training", {})[key] = cli[key]
    for key in ("head_dim", "num_heads", "contact_head"):
        if cli.get(key) is not None:
            cfg.setdefault("architecture", {})[key] = cli[key]
    for key in ("focal_alpha", "focal_gamma"):
        if cli.get(key) is not None:
            cfg.setdefault("loss", {})[key] = cli[key]
    if cli.get("dynamic_batching") is not None:
        cfg.setdefault("data_loading", {})["dynamic_batching"] = cli["dynamic_batching"]
    if cli.get("description"):
        cfg["description"] = cli["description"]
    for key in ("log_dir", "save_dir"):
        if cli.get(key):
            cfg.setdefault("output", {})[key] = cli[key]
    return cfg


# -- LR schedule: linear warmup -> cosine decay --------------------------------

def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int,
                           min_lr_ratio: float = 0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
    return LambdaLR(optimizer, lr_lambda)


# -- Data loaders --------------------------------------------------------------

def build_loaders(train_data, val_data, cfg, dna_special, device):
    """Build train/val DataLoaders according to config.

    Returns ``(train_loader, val_loader, uses_dynamic_batching)``.
    """
    t = cfg.get("training", {})
    dl_cfg = cfg.get("data_loading", {})
    bs = t.get("batch_size", 16)
    nw = t.get("num_workers", 4)
    pin = device.type == "cuda"
    persistent = nw > 0

    use_dynamic = dl_cfg.get("dynamic_batching", True)

    if use_dynamic:
        train_ds = VariableLengthContactMapDataset(
            train_data, dna_has_special_tokens=dna_special)
        val_ds = VariableLengthContactMapDataset(
            val_data, dna_has_special_tokens=dna_special)

        pos_enrich = dl_cfg.get("positive_enrichment", False)
        min_pos_frac = dl_cfg.get("min_positive_fraction", 0.5)
        bucket_mult = dl_cfg.get("bucket_size_multiplier", 10)

        train_sampler = BucketBatchSampler(
            train_ds, batch_size=bs,
            bucket_size_multiplier=bucket_mult,
            shuffle=True,
            positive_enrichment=pos_enrich,
            min_positive_fraction=min_pos_frac,
            seed=t.get("seed", 42),
        )

        collate_fn = functools.partial(
            collate_contactmap_batch,
            dna_has_special_tokens=dna_special,
        )

        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=nw, pin_memory=pin,
            persistent_workers=persistent,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs,
            collate_fn=collate_fn,
            num_workers=nw, pin_memory=pin,
            persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            ContactMapDataset(train_data, dna_has_special_tokens=dna_special),
            batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin,
            persistent_workers=persistent,
        )
        val_loader = DataLoader(
            ContactMapDataset(val_data, dna_has_special_tokens=dna_special),
            batch_size=bs,
            num_workers=nw, pin_memory=pin,
            persistent_workers=persistent,
        )

    return train_loader, val_loader, use_dynamic


# -- Training / eval -----------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    focal_alpha: float, focal_gamma: float,
                    sampler=None, epoch: int = 0):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = masked_focal_loss(logits, batch["Y"],
                                 batch["prot_mask"], batch["dna_mask"],
                                 alpha=focal_alpha, gamma=focal_gamma)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * batch["Y"].size(0)
        n += batch["Y"].size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, focal_alpha: float, focal_gamma: float):
    model.eval()
    all_true, all_score = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        total_loss += (
            masked_focal_loss(logits, batch["Y"],
                              batch["prot_mask"], batch["dna_mask"],
                              alpha=focal_alpha, gamma=focal_gamma).item()
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


# -- Main ----------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args)

    t = cfg.get("training", {})
    a = cfg.get("architecture", {})
    o = cfg.get("output", {})
    loss_cfg = cfg.get("loss", {})
    dl_cfg = cfg.get("data_loading", {})

    focal_alpha = loss_cfg.get("focal_alpha", 0.8)
    focal_gamma = loss_cfg.get("focal_gamma", 1.0)

    set_seed(t.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data, data_spec, split, split_spec = load_split_and_resolve_data(cfg)

    dna_short = data_spec["dna_model_short"]
    prot_short = data_spec["prot_model_short"]
    dna_info = get_model_info(dna_short)
    prot_info = get_model_info(prot_short)

    train_data = subset_data(data, split["train_idx"])
    val_data = subset_data(data, split["val_idx"])

    dna_special = dna_info.get("has_special_tokens", True)
    train_loader, val_loader, uses_dynamic = build_loaders(
        train_data, val_data, cfg, dna_special, device,
    )

    print(f"Loading DNA model: {dna_info['hf_name']}")
    dna_lm = load_model(dna_short, device=str(device))
    print(f"Loading Protein model: {prot_info['hf_name']}")
    prot_lm = load_model(prot_short, device=str(device))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(o.get("save_dir", "./results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    bs = t.get("batch_size", 16)
    epochs = t.get("epochs", 100)
    warmup_epochs = t.get("warmup_epochs", 5)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    split_tag = split_spec.get("strategy", "unk").replace("_", "")
    tl = a.get("target_layers", [0, 3, 5])
    head_tag = a.get("contact_head", "bilinear")
    dyn_tag = "dyn" if uses_dynamic else "fix"
    run_tag = (f"comp_v2_{dna_short}_{prot_short}_{split_tag}"
               f"_lr{t.get('lr', 5e-5)}_bs{bs}_hd{a.get('head_dim', 64)}"
               f"_nh{a.get('num_heads', 20)}"
               f"_tl{''.join(str(x) for x in tl)}"
               f"_a{focal_alpha}_g{focal_gamma}"
               f"_{head_tag}_{dyn_tag}"
               f"_{ts}")

    print(f"Focal loss: alpha={focal_alpha}, gamma={focal_gamma}")
    print(f"Contact head: {head_tag}")
    print(f"Dynamic batching: {uses_dynamic}")
    if dl_cfg.get("positive_enrichment"):
        print(f"Positive enrichment: min_fraction={dl_cfg.get('min_positive_fraction', 0.5)}")
    print(f"LR schedule: warmup={warmup_epochs} epochs ({warmup_steps} steps), "
          f"total={epochs} epochs ({total_steps} steps)")

    for run in range(t.get("n_runs", 1)):
        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{t.get('n_runs', 1)}  [{run_tag}]")
        print("=" * 60)

        loss_tag = f"focal_a{focal_alpha}_g{focal_gamma}"
        init_run(cfg, run_tag, run, "composition",
                 dna_short=dna_short, prot_short=prot_short,
                 loss_tag=loss_tag)
        log_source_artifacts(importlib.import_module(build_model.__module__),
                             dna_info, prot_info, __file__, args.config)

        model = build_model(dna_lm, prot_lm, dna_info, prot_info, a,
                            device=str(device))
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=t.get("lr", 5e-5),
        )
        scheduler = warmup_cosine_schedule(optimizer, warmup_steps, total_steps)

        train_sampler = (train_loader.batch_sampler
                         if uses_dynamic else None)

        best_pr_auc = -1.0
        best_metrics: dict = {}
        ckpt_path = save_dir / f"{run_tag}_run{run}_best.pth"

        for epoch in tqdm(range(epochs), desc=f"Run {run}"):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                focal_alpha, focal_gamma,
                sampler=train_sampler, epoch=epoch,
            )
            metrics = evaluate(model, val_loader, device,
                               focal_alpha, focal_gamma)

            cur_lr = optimizer.param_groups[0]["lr"]
            log_scalars(epoch, train_loss, metrics, lr=cur_lr)
            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_metrics = metrics
                torch.save(model.state_dict(), ckpt_path)

        log_best_metrics(best_metrics)
        log_checkpoint(ckpt_path)
        finish()
        print(f"  Best: PR-AUC={best_metrics.get('pr_auc', 0):.4f}, "
              f"ROC-AUC={best_metrics.get('roc_auc', 0):.4f}, "
              f"MCC={best_metrics.get('mcc', 0):.4f}, "
              f"P={best_metrics.get('precision', 0):.4f}, "
              f"R={best_metrics.get('recall', 0):.4f}, "
              f"F1={best_metrics.get('f1', 0):.4f}, "
              f"Top-L={best_metrics.get('top_L_precision', 0):.4f}")


if __name__ == "__main__":
    main()
