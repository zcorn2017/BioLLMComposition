#!/usr/bin/env python3
"""Tokenize sequences and pack raw contact maps for all samples.

Produces a single ``.pt`` file containing tokenised inputs for both DNA
strands and the protein, plus compact (variable-size) contact maps.
**No splitting** is performed here; see ``01_split_datasets.py``.

Usage
-----
    python scripts/00_precompute_tokens_and_labels.py \
        --config configs/precompute/default.yaml

    # Override any value via CLI:
    python scripts/00_precompute_tokens_and_labels.py \
        --config configs/precompute/default.yaml \
        --dna_model dnabert2-117M --prot_model esm2-650M
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from biollmcomposition.models import get_model_info, load_tokenizer
from biollmcomposition.utils.contact_map import resolve_data_path, set_seed


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--dna_model", type=str, default=None,
                   help="Short name override (e.g. dnabert2-117M)")
    p.add_argument("--prot_model", type=str, default=None,
                   help="Short name override (e.g. esm2-650M)")
    p.add_argument("--max_dna_len", type=int, default=None)
    p.add_argument("--max_prot_len", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_config(args) -> dict:
    """Merge YAML config with CLI overrides (CLI wins)."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cli = vars(args)
    if cli.get("data_path"):
        cfg["data_path"] = cli["data_path"]
    if cli.get("out_dir"):
        cfg["out_dir"] = cli["out_dir"]
    if cli.get("dna_model"):
        cfg.setdefault("models", {})["dna"] = cli["dna_model"]
    if cli.get("prot_model"):
        cfg.setdefault("models", {})["protein"] = cli["prot_model"]
    if cli.get("max_dna_len") is not None:
        cfg["max_dna_len"] = cli["max_dna_len"]
    if cli.get("max_prot_len") is not None:
        cfg["max_prot_len"] = cli["max_prot_len"]
    if cli.get("seed") is not None:
        cfg["seed"] = cli["seed"]
    return cfg


# ── Helpers ───────────────────────────────────────────────────────────────

def load_data(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_parquet(path)


def auto_max_len(seqs, cap: int, buffer: int = 10) -> int:
    """max(seq_lengths) + buffer, capped."""
    return min(cap, max(len(s) for s in seqs if s) + buffer)


# ── Core tokenization ────────────────────────────────────────────────────

def tokenize_and_pack(df: pd.DataFrame, dna_tok, prot_tok,
                      R: int, L: int,
                      dna_has_special_tokens: bool = True) -> dict:
    """Tokenize all samples and pack raw contact maps.

    Contact maps are stored as compact int8 numpy arrays (variable size)
    instead of a dense ``(N, 2, R, L)`` tensor to avoid OOM.
    """
    n = len(df)
    prot_seqs = df["prot_seq"].tolist()
    dna1_seqs = df["dna_seq_1"].tolist()
    dna2_raw = df["dna_seq_2"].tolist()
    dna2_seqs = [s if isinstance(s, str) and s else "" for s in dna2_raw]

    dna_tok_kwargs: dict = dict(
        return_tensors="pt", padding="max_length",
        max_length=L, truncation=True,
    )
    if not dna_has_special_tokens:
        dna_tok_kwargs["add_special_tokens"] = False

    print(f"  Tokenizing protein ({n}) …")
    prot_tokens = prot_tok(
        prot_seqs, return_tensors="pt",
        padding="max_length", max_length=R, truncation=True,
    )
    print(f"  Tokenizing dna_strand_1 ({n}) …")
    dna1_tokens = dict(dna_tok(dna1_seqs, **dna_tok_kwargs))
    print(f"  Tokenizing dna_strand_2 ({n}) …")
    dna2_tokens = dict(dna_tok(dna2_seqs, **dna_tok_kwargs))

    # Some tokenizers (e.g. NTv3) omit attention_mask; derive from pad tokens
    pad_id = dna_tok.pad_token_id
    for tok_dict in (dna1_tokens, dna2_tokens):
        if "attention_mask" not in tok_dict and pad_id is not None:
            tok_dict["attention_mask"] = (
                tok_dict["input_ids"] != pad_id
            ).long()

    print("  Packing raw contact maps …")
    contact_maps: list[dict] = []
    total_pos = 0
    for _, row in tqdm(df.iterrows(), total=n, desc="  contact_maps"):
        cm1 = np.asarray(row["contact_map_1"], dtype=np.int8)
        total_pos += cm1.sum()
        cm2_raw = row.get("contact_map_2")
        cm2 = (np.asarray(cm2_raw, dtype=np.int8)
               if cm2_raw is not None else None)
        if cm2 is not None:
            total_pos += cm2.sum()
        contact_maps.append({"cm1": cm1, "cm2": cm2})

    cm_bytes = sum(
        c["cm1"].nbytes + (c["cm2"].nbytes if c["cm2"] is not None else 0)
        for c in contact_maps
    )
    print(f"  {n} samples, cm storage ≈ {cm_bytes / 1e6:.1f} MB, "
          f"total positives = {total_pos}")

    # Sanity spot-check
    row0 = df.iloc[0]
    if np.asarray(row0["contact_map_1"]).sum() != contact_maps[0]["cm1"].sum():
        print("  [WARN] sample-0 contact_map_1 mismatch after packing")

    meta: dict = {
        "pdb_ids": df["pdb_id"].tolist(),
        "prot_chain_ids": df["prot_chain_id"].tolist(),
        "dna_entity_types": df["dna_entity_type"].tolist(),
    }
    if "cluster_id" in df.columns:
        meta["cluster_ids"] = df["cluster_id"].tolist()

    return {
        "dna1": {k: v for k, v in dna1_tokens.items()},
        "dna2": {k: v for k, v in dna2_tokens.items()},
        "prot": {k: v for k, v in prot_tokens.items()},
        "contact_maps": contact_maps,
        "meta": meta,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    dna_short = cfg["models"]["dna"]
    prot_short = cfg["models"]["protein"]
    dna_info = get_model_info(dna_short)
    prot_info = get_model_info(prot_short)

    out_dir = Path(resolve_data_path(cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    data_path = resolve_data_path(cfg["data_path"])
    print(f"Loading data from {data_path} …")
    df = load_data(data_path)
    print(f"  {len(df)} samples, columns: {df.columns.tolist()}")

    # ── max lengths ──
    all_prot = df["prot_seq"].tolist()
    all_dna = (
        df["dna_seq_1"].tolist()
        + [s for s in df["dna_seq_2"].tolist()
           if isinstance(s, str) and s]
    )
    dna_has_special_tokens = dna_info.get("has_special_tokens", True)

    max_prot = cfg.get("max_prot_len", 0)
    max_dna = cfg.get("max_dna_len", 0)
    R = max_prot if max_prot and max_prot > 0 else auto_max_len(all_prot, 1024)
    L = max_dna if max_dna and max_dna > 0 else auto_max_len(all_dna, 512)

    # NTv3 full U-Net requires sequence length divisible by 2^num_downsamples.
    # In stem_only mode the U-Net is bypassed, so no alignment is needed.
    num_ds = dna_info.get("num_downsamples", 0)
    stem_only = dna_info.get("stem_only", False)
    if num_ds > 0 and not stem_only:
        alignment = 2 ** num_ds
        L = ((L + alignment - 1) // alignment) * alignment
        print(f"  DNA length rounded to multiple of {alignment} for U-Net")

    print(f"  max_prot_len (R) = {R},  max_dna_len (L) = {L}")

    # ── tokenizers ──
    print(f"Loading tokenizers: {dna_info['hf_name']} + {prot_info['hf_name']}")
    dna_tok = load_tokenizer(dna_short)
    prot_tok = load_tokenizer(prot_short)

    # ── tokenize all samples ──
    data = tokenize_and_pack(df, dna_tok, prot_tok, R, L,
                             dna_has_special_tokens=dna_has_special_tokens)

    # ── spec ──
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    spec = {
        "dna_model": dna_info["hf_name"],
        "dna_model_short": dna_short,
        "prot_model": prot_info["hf_name"],
        "prot_model_short": prot_short,
        "max_prot_len": R,
        "max_dna_len": L,
        "n_samples": len(df),
        "seed": seed,
        "timestamp": ts,
        "source_data": str(cfg["data_path"]),
    }

    out_name = (f"tokens_{dna_short}_{prot_short}"
                f"_dna{L}_prot{R}_{ts}.pt")
    out_path = out_dir / out_name

    print(f"\nSaving → {out_path}")
    torch.save({"spec": spec, **data}, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
