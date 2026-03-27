#!/usr/bin/env python3
"""Standalone training script for the attention contact-map model.

For the unified config-driven entry point, prefer:
    python scripts/02_run_training.py --config configs/training/attention_contactmap.yaml
"""

from __future__ import annotations

import argparse
import datetime
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from biollmcomposition.frameworks.attention import (
    AttentionContactMapModel,
    build_model,
)
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
    for key in ("lr", "epochs", "batch_size", "num_workers", "n_runs", "seed"):
        if cli.get(key) is not None:
            cfg.setdefault("training", {})[key] = cli[key]
    if cli.get("head_dim") is not None:
        cfg.setdefault("architecture", {})["head_dim"] = cli["head_dim"]
    for key in ("log_dir", "save_dir"):
        if cli.get(key):
            cfg.setdefault("output", {})[key] = cli[key]
    return cfg


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = masked_bce_loss(logits, batch["Y"],
                               batch["prot_mask"], batch["dna_mask"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["Y"].size(0)
        n += batch["Y"].size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_score = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        total_loss += (
            masked_bce_loss(logits, batch["Y"],
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


def main():
    args = parse_args()
    cfg = load_config(args)

    t = cfg.get("training", {})
    a = cfg.get("architecture", {})
    o = cfg.get("output", {})

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

    print(f"Loading DNA model: {dna_info['hf_name']}")
    dna_lm = load_model(dna_short, device=str(device))
    print(f"Loading Protein model: {prot_info['hf_name']}")
    prot_lm = load_model(prot_short, device=str(device))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(o.get("save_dir", "./results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    split_tag = split_spec.get("strategy", "unk").replace("_", "")
    run_tag = (f"attn_cm_{dna_short}_{prot_short}_{split_tag}"
               f"_lr{t.get('lr', 5e-5)}_bs{bs}_hd{a.get('head_dim', 64)}"
               f"_{ts}")

    for run in range(t.get("n_runs", 1)):
        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{t.get('n_runs', 1)}  [{run_tag}]")
        print("=" * 60)

        init_run(cfg, run_tag, run, "attention",
                 dna_short=dna_short, prot_short=prot_short, loss_tag="bce")
        log_source_artifacts(importlib.import_module(build_model.__module__),
                             dna_info, prot_info, __file__, args.config)

        model = build_model(dna_lm, prot_lm, dna_info, prot_info, a,
                            device=str(device))
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=t.get("lr", 5e-5),
        )

        best_pr_auc = -1.0
        best_metrics: dict = {}
        ckpt_path = save_dir / f"{run_tag}_run{run}_best.pth"

        for epoch in tqdm(range(t.get("epochs", 100)), desc=f"Run {run}"):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            metrics = evaluate(model, val_loader, device)
            log_scalars(epoch, train_loss, metrics)
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
