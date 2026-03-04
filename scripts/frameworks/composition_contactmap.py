#!/usr/bin/env python3
"""Train composition (LM-interleaved cross-attention) contact-map model.

DNA context from both strands is concatenated and injected into ESM2
transformer layers via cross-attention at selected layers.  The final
per-token protein representations are paired with per-strand DNA hidden
states through a bilinear head to produce ``(B, 2, R, L)`` logits.

Usage
-----
    python scripts/frameworks/composition_contactmap.py \
        --config scripts/configs/composition_contactmap.yaml

    # Override any config value via CLI:
    python scripts/frameworks/composition_contactmap.py \
        --config scripts/configs/composition_contactmap.yaml \
        --lr 1e-4 --batch_size 8
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, BertConfig

from biollmcomposition.utils.contact_map import (
    set_seed,
    masked_bce_loss,
    flatten_valid,
    compute_contactmap_metrics,
    ContactMapDataset,
)


# ── Config loading ────────────────────────────────────────────────────────

def load_config(args):
    """Merge YAML config with CLI overrides. CLI wins."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cli = vars(args)
    for key in ["lr", "epochs", "batch_size", "num_workers", "n_runs",
                "seed", "head_dim", "num_heads", "log_dir", "save_dir",
                "data_pt"]:
        if cli.get(key) is not None:
            if key in ("lr", "epochs", "batch_size", "num_workers",
                       "n_runs", "seed"):
                cfg.setdefault("training", {})[key] = cli[key]
            elif key in ("head_dim", "num_heads"):
                cfg.setdefault("architecture", {})[key] = cli[key]
            elif key == "data_pt":
                cfg.setdefault("data", {})["data_pt"] = cli[key]
            elif key in ("log_dir", "save_dir"):
                cfg.setdefault("output", {})[key] = cli[key]
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--data_pt", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────

class ContactMapHead(nn.Module):
    """Bilinear contact head: score(r, l) = prot_proj[r] · dna_proj[l]."""

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

    DNA context (both strands concatenated) is injected into ESM2 layers
    via cross-attention at ``target_layers``.  The final per-token protein
    representations are paired with per-strand DNA hidden states through a
    bilinear contact-map head.
    """

    def __init__(self, dna_lm, prot_lm, dna_emb_dim, prot_emb_dim,
                 num_heads: int = 20, head_dim: int = 64,
                 target_layers=None, esm_layers: int = 6):
        super().__init__()
        self.dna_lm = dna_lm
        self.prot_lm = prot_lm
        self.num_heads = num_heads
        self.target_layers = target_layers or [0, 3, 5]
        self.esm_layers = esm_layers

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
        return self.dna_proj(h)

    def forward(self, batch):
        dna1_h = self._dna_hidden(batch["dna1_input_ids"],
                                  batch["dna1_attention_mask"])
        dna2_h = self._dna_hidden(batch["dna2_input_ids"],
                                  batch["dna2_attention_mask"])

        dna_cat = torch.cat([dna1_h, dna2_h], dim=1)
        dna_cat_am = torch.cat([batch["dna1_attention_mask"],
                                batch["dna2_attention_mask"]], dim=1)
        dna_cat_kpm = ~dna_cat_am.bool()

        prot_am = batch["prot_attention_mask"]
        prot_self_mask = self.prot_lm.get_extended_attention_mask(
            prot_am, batch["prot_input_ids"].size(),
        )

        prot = self.prot_lm.esm.embeddings(
            batch["prot_input_ids"], prot_am,
        )

        counter = 0
        for i in range(self.esm_layers):
            prot = self.prot_lm.esm.encoder.layer[i](
                prot, prot_self_mask,
            )[0]
            if i in self.target_layers:
                attn_out, _ = self.cross_attn_layers[counter](
                    query=prot, key=dna_cat, value=dna_cat,
                    key_padding_mask=dna_cat_kpm,
                )
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
                prot = prot + self.post_attn_norms[counter](attn_out)
                counter += 1

        prot = self.prot_lm.esm.encoder.emb_layer_norm_after(prot)

        cm1 = self.contact_head(prot, dna1_h)
        cm2 = self.contact_head(prot, dna2_h)
        return torch.stack([cm1, cm2], dim=1)


# ── Training / eval loops ─────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args)

    m = cfg["models"]
    t = cfg["training"]
    a = cfg["architecture"]
    o = cfg["output"]

    set_seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    dna_short = m["dna"]["short"]
    prot_short = m["protein"]["short"]
    dna_emb_dim = m["dna"]["emb_dim"]
    prot_emb_dim = m["protein"]["emb_dim"]
    target_layers = a.get("target_layers", [0, 3, 5])
    esm_layers = a.get("esm_layers", 6)

    # ── data ──
    data = torch.load(cfg["data"]["data_pt"], map_location="cpu",
                      weights_only=False)
    spec = data["spec"]
    print(f"Spec: {spec}")

    nw = t["num_workers"]
    train_loader = DataLoader(
        ContactMapDataset(data["train"]),
        batch_size=t["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        ContactMapDataset(data["val"]),
        batch_size=t["batch_size"],
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )

    # ── LMs (frozen) ──
    print(f"Loading DNA model: {m['dna']['name']}")
    dna_config = BertConfig.from_pretrained(m["dna"]["name"])
    dna_lm = (AutoModel
              .from_pretrained(m["dna"]["name"], trust_remote_code=True,
                               config=dna_config)
              .to(device).eval())
    print(f"Loading Protein model: {m['protein']['name']}")
    prot_lm = (AutoModelForMaskedLM
               .from_pretrained(m["protein"]["name"])
               .to(device).eval())

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(o["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    tl_str = "".join(str(x) for x in target_layers)
    run_tag = (f"comp_cm_{dna_short}_{prot_short}"
               f"_lr{t['lr']}_bs{t['batch_size']}"
               f"_hd{a['head_dim']}_nh{a['num_heads']}_tl{tl_str}"
               f"_{ts}")

    # ── runs ──
    for run in range(t["n_runs"]):
        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{t['n_runs']}  [{run_tag}]")
        print("=" * 60)

        writer = SummaryWriter(f"{o['log_dir']}/{run_tag}/run{run}")
        writer.add_text("config", str(cfg))
        writer.add_text("base_models",
                        f"DNA: {m['dna']['name']}\nProtein: {m['protein']['name']}")

        model = CompositionContactMapModel(
            dna_lm, prot_lm,
            dna_emb_dim=dna_emb_dim, prot_emb_dim=prot_emb_dim,
            num_heads=a["num_heads"], head_dim=a["head_dim"],
            target_layers=target_layers, esm_layers=esm_layers,
        ).to(device)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=t["lr"],
        )

        best_pr_auc = -1.0
        best_metrics = {}
        ckpt_path = save_dir / f"{run_tag}_best.pth"

        for epoch in tqdm(range(t["epochs"]), desc=f"Run {run}"):
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         device)
            metrics = evaluate(model, val_loader, device)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", metrics["val_loss"], epoch)
            for k, v in metrics.items():
                if k != "val_loss":
                    writer.add_scalar(f"metric/{k}", v, epoch)

            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_metrics = metrics
                torch.save(model.state_dict(), ckpt_path)

        writer.close()
        print(f"  Best: PR-AUC={best_metrics.get('pr_auc', 0):.4f}, "
              f"ROC-AUC={best_metrics.get('roc_auc', 0):.4f}, "
              f"MCC={best_metrics.get('mcc', 0):.4f}, "
              f"P={best_metrics.get('precision', 0):.4f}, "
              f"R={best_metrics.get('recall', 0):.4f}, "
              f"F1={best_metrics.get('f1', 0):.4f}, "
              f"Top-L={best_metrics.get('top_L_precision', 0):.4f}")


if __name__ == "__main__":
    main()
