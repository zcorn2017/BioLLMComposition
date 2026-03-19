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
import inspect
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from biollmcomposition.frameworks import get_framework
from biollmcomposition.models import get_model_info, load_model
from biollmcomposition.utils.contact_map import (
    ContactMapDataset,
    compute_contactmap_metrics,
    flatten_valid,
    masked_bce_loss,
    masked_focal_loss,
    load_split_and_resolve_data,
    resolve_data_path,
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
def evaluate(model, loader, device, loss_fn, _eval_call_count=[0]):
    model.eval()
    all_true, all_score = [], []
    total_loss, n = 0.0, 0
    _eval_call_count[0] += 1
    _should_log = _eval_call_count[0] <= 2  # log first 2 eval calls
    for _bi, batch in enumerate(loader):
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

        # #region agent log — H3/H5: check logit and score distribution for first batch
        if _should_log and _bi == 0:
            import json as _json, time as _time
            _LOGPATH = "/home/zcorn/Projects/BioLLMComposition/.cursor/debug-9cd513.log"
            _probs = torch.sigmoid(logits)
            _mask = (batch["prot_mask"][:, None, :, None].float()
                     * batch["dna_mask"][:, :, None, :].float()).bool()
            _valid_logits = logits[_mask]
            _valid_probs = _probs[_mask]
            _valid_labels = batch["Y"][_mask]
            with open(_LOGPATH, "a") as _f:
                _f.write(_json.dumps({"sessionId":"9cd513",
                    "hypothesisId":"H3_H5",
                    "location":"evaluate:first_batch",
                    "message":f"logit/score stats (eval call {_eval_call_count[0]})",
                    "data":{
                        "batch_size": logits.shape[0],
                        "logit_shape": list(logits.shape),
                        "valid_positions": int(_mask.sum()),
                        "valid_positives": int(_valid_labels.sum()),
                        "positive_rate": float(_valid_labels.sum() / _mask.sum()) if _mask.sum() > 0 else 0,
                        "logit_mean": float(_valid_logits.mean()),
                        "logit_std": float(_valid_logits.std()),
                        "logit_min": float(_valid_logits.min()),
                        "logit_max": float(_valid_logits.max()),
                        "prob_mean": float(_valid_probs.mean()),
                        "prob_gt_05": int((_valid_probs > 0.5).sum()),
                        "prob_gt_03": int((_valid_probs > 0.3).sum()),
                        "prob_gt_01": int((_valid_probs > 0.1).sum()),
                        "labels_sum_in_batch": float(batch["Y"].sum()),
                        "yt_len": len(yt), "yt_sum": float(yt.sum()),
                        "ys_mean": float(ys.mean()), "ys_max": float(ys.max()),
                    },
                    "timestamp":int(_time.time()*1000)}) + "\n")
        # #endregion

    # #region agent log — H5: overall eval metrics before aggregation
    _all_t = np.concatenate(all_true)
    _all_s = np.concatenate(all_score)
    if _should_log:
        import json as _json, time as _time
        _LOGPATH = "/home/zcorn/Projects/BioLLMComposition/.cursor/debug-9cd513.log"
        with open(_LOGPATH, "a") as _f:
            _f.write(_json.dumps({"sessionId":"9cd513",
                "hypothesisId":"H5",
                "location":"evaluate:aggregate",
                "message":f"full eval stats (eval call {_eval_call_count[0]})",
                "data":{
                    "total_valid_positions": len(_all_t),
                    "total_positives": int(_all_t.sum()),
                    "positive_rate": float(_all_t.sum() / len(_all_t)) if len(_all_t) > 0 else 0,
                    "score_mean": float(_all_s.mean()),
                    "score_std": float(_all_s.std()),
                    "preds_gt_05": int((_all_s >= 0.5).sum()),
                    "preds_gt_03": int((_all_s >= 0.3).sum()),
                },
                "timestamp":int(_time.time()*1000)}) + "\n")
    # #endregion

    metrics = compute_contactmap_metrics(_all_t, _all_s)
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
    """Build a descriptive, filesystem-safe TensorBoard run name."""
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
    log_dir = o.get("log_dir", "./runs")

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

    # #region agent log — debug instrumentation (data pipeline diagnostics)
    import json as _json, time as _time
    _LOGPATH = "/home/zcorn/Projects/BioLLMComposition/.cursor/debug-9cd513.log"
    def _dlog(hyp, loc, msg, data):
        with open(_LOGPATH, "a") as _f:
            _f.write(_json.dumps({"sessionId":"9cd513","hypothesisId":hyp,
                "location":loc,"message":msg,"data":data,
                "timestamp":int(_time.time()*1000)}) + "\n")

    # H1/H2/H4: Check mask and label statistics from the dataset
    _ds = train_loader.dataset
    _n_check = min(50, len(_ds))
    _valid_counts, _pos_counts, _total_poss = [], [], []
    _dna_lens, _cm_shapes = [], []
    for _i in range(_n_check):
        _b = _ds[_i]
        _pm = _b["prot_mask"]   # (R,)
        _dm = _b["dna_mask"]    # (2, L)
        _Y = _b["Y"]           # (2, R, L)
        _mask2d = _pm[None, :, None].float() * _dm[:, None, :].float()  # (2,R,L)
        _nvalid = _mask2d.sum().item()
        _npos = (_Y * _mask2d).sum().item()
        _npos_raw = _Y.sum().item()
        _valid_counts.append(_nvalid)
        _pos_counts.append(_npos)
        _total_poss.append(_npos_raw)
        _dna_lens.append(_dm[0].sum().item())
        _cm = _ds.contact_maps[_i]
        _cm_shapes.append(list(_cm["cm1"].shape))
    _dlog("H1", "main:data_stats", "mask and label statistics over first N samples", {
        "n_checked": _n_check,
        "R": _ds.R, "L": _ds.L,
        "dna_seq_lens_min_max_mean": [min(_dna_lens), max(_dna_lens), sum(_dna_lens)/len(_dna_lens)],
        "valid_positions_min_max_mean": [min(_valid_counts), max(_valid_counts), sum(_valid_counts)/len(_valid_counts)],
        "positives_in_valid_min_max_mean": [min(_pos_counts), max(_pos_counts), sum(_pos_counts)/len(_pos_counts)],
        "positives_raw_min_max_mean": [min(_total_poss), max(_total_poss), sum(_total_poss)/len(_total_poss)],
    })

    # H2: Verify contact map alignment — compare raw cm shape vs placed in Y
    _s0 = _ds[0]
    _cm0 = _ds.contact_maps[0]
    _cm1_raw = _cm0["cm1"]
    _dlog("H2", "main:alignment", "sample 0 alignment check", {
        "cm1_raw_shape": list(_cm1_raw.shape),
        "cm1_raw_positives": int(_cm1_raw.sum()),
        "Y_shape": list(_s0["Y"].shape),
        "Y_total_positives": float(_s0["Y"].sum()),
        "Y_strand0_positives": float(_s0["Y"][0].sum()),
        "prot_mask_sum": float(_s0["prot_mask"].sum()),
        "dna_mask_strand0_sum": float(_s0["dna_mask"][0].sum()),
        "dna_off": 0 if not dna_special else 1,
        "dna_has_special_tokens": dna_special,
        "first_10_dna1_ids": _ds.dna1_ids[0, :10].tolist(),
    })

    # H4: Check that mask doesn't accidentally zero out real tokens
    _am_raw = _ds.dna1_am[0]
    _ids_raw = _ds.dna1_ids[0]
    _nonpad_count = (_ids_raw != 1).sum().item()  # NTv3 pad=1
    _mask_count = _am_raw.sum().item()
    _dlog("H4", "main:mask_vs_ids", "mask vs raw nonpad token count (sample 0)", {
        "nonpad_tokens": _nonpad_count,
        "mask_1s": _mask_count,
        "match": _nonpad_count == _mask_count,
        "first_20_ids": _ids_raw[:20].tolist(),
        "first_20_mask": _am_raw[:20].tolist(),
    })
    # #endregion

    # ── Training runs ──
    for run in range(n_runs):
        print(f"\n{'=' * 70}")
        print(f"Run {run + 1}/{n_runs}  [{run_tag}]")
        print("=" * 70)

        writer = SummaryWriter(f"{log_dir}/{run_tag}/run{run}")
        writer.add_text("training_config", str(cfg))
        writer.add_text("data_spec", str(data_spec))
        writer.add_text("split_spec", str(split_spec))
        writer.add_text("base_models",
                        f"DNA: {dna_info['hf_name']} ({dna_short})\n"
                        f"Protein: {prot_info['hf_name']} ({prot_short})")
        writer.add_text("loss", loss_tag)

        # Snapshot source code for full reproducibility
        writer.add_text("source/training_script",
                        f"```python\n{Path(__file__).read_text()}\n```")
        writer.add_text("source/framework",
                        f"```python\n{inspect.getsource(type(framework))}\n```"
                        if hasattr(framework, '__module__') and not inspect.ismodule(framework)
                        else f"```python\n{inspect.getsource(framework)}\n```")
        from biollmcomposition.utils import contact_map as _cm_mod
        writer.add_text("source/contact_map_utils",
                        f"```python\n{inspect.getsource(_cm_mod)}\n```")
        writer.add_text("source/config_yaml",
                        f"```yaml\n{Path(args.config).read_text()}\n```")

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

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", metrics["val_loss"], epoch)
            if scheduler is not None:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            for k, v in metrics.items():
                if k != "val_loss":
                    writer.add_scalar(f"metric/{k}", v, epoch)

            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_metrics = metrics
                torch.save(model.state_dict(), ckpt_path)

        writer.close()
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
