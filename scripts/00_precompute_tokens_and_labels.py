#!/usr/bin/env python3
"""Tokenize sequences and precompute padded contact-map labels / masks.

Produces a single ``.pt`` file containing tokenised inputs for both DNA
strands and protein, plus the raw (compact) contact maps.  Padded Y
tensors and masks are built on-the-fly per sample in the Dataset to
avoid materialising a dense (N, 2, R, L) array that would OOM.

Usage
-----
    python scripts/00_precompute_tokens_and_labels.py \
        --data_path /path/to/residue_wise.pkl \
        --out_dir   /path/to/processed/embeddings
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from transformers import AutoTokenizer

from biollmcomposition.utils.contact_map import set_seed


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", type=str,
                   default="/home/zcorn/Projects/proteinDNA_data/processed/"
                           "dna_protein_residue_wise_fullseq_20260219.pkl")
    p.add_argument("--out_dir", type=str,
                   default="/home/zcorn/Projects/proteinDNA_data/processed/embeddings")
    p.add_argument("--max_dna_len", type=int, default=0,
                   help="0 = auto-detect from data, capped at 512")
    p.add_argument("--max_prot_len", type=int, default=0,
                   help="0 = auto-detect from data, capped at 1024")
    p.add_argument("--train_size", type=float, default=0.8)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────

DNA_MODEL = "zhihan1996/DNABERT-2-117M"
PROT_MODEL = "facebook/esm2_t6_8M_UR50D"


def load_data(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_parquet(path)


def auto_max_len(seqs, cap, buffer=10):
    """max(seq_lengths) + buffer, capped."""
    return min(cap, max(len(s) for s in seqs if s) + buffer)


# ── Per-split processing ─────────────────────────────────────────────────

def process_split(split_df, dna_tok, prot_tok, R, L, split_name):
    """Tokenize sequences and pack raw contact maps for one data split.

    Contact maps are stored as compact int8 numpy arrays (variable size)
    instead of a dense (N, 2, R, L) tensor to avoid OOM.  Padding is done
    on-the-fly inside ``ContactMapDataset.__getitem__``.
    """
    n = len(split_df)
    prot_seqs = split_df["prot_seq"].tolist()
    dna1_seqs = split_df["dna_seq_1"].tolist()
    dna2_raw = split_df["dna_seq_2"].tolist()
    dna2_seqs = [s if isinstance(s, str) and s else "" for s in dna2_raw]

    print(f"  Tokenizing {split_name} protein ({n}) …")
    prot_tokens = prot_tok(
        prot_seqs, return_tensors="pt",
        padding="max_length", max_length=R, truncation=True,
    )
    print(f"  Tokenizing {split_name} dna_strand_1 ({n}) …")
    dna1_tokens = dna_tok(
        dna1_seqs, return_tensors="pt",
        padding="max_length", max_length=L, truncation=True,
    )
    print(f"  Tokenizing {split_name} dna_strand_2 ({n}) …")
    dna2_tokens = dna_tok(
        dna2_seqs, return_tensors="pt",
        padding="max_length", max_length=L, truncation=True,
    )

    # ── pack raw contact maps (compact) ──
    print(f"  Packing raw contact maps for {split_name} …")
    contact_maps = []
    total_pos = 0
    for _, row in tqdm(split_df.iterrows(), total=n,
                       desc=f"  cm_{split_name}"):
        cm1 = np.asarray(row["contact_map_1"], dtype=np.int8)
        total_pos += cm1.sum()
        cm2_raw = row.get("contact_map_2")
        cm2 = np.asarray(cm2_raw, dtype=np.int8) if cm2_raw is not None else None
        if cm2 is not None:
            total_pos += cm2.sum()
        contact_maps.append({"cm1": cm1, "cm2": cm2})

    cm_bytes = sum(
        c["cm1"].nbytes + (c["cm2"].nbytes if c["cm2"] is not None else 0)
        for c in contact_maps
    )
    print(f"  {split_name}: {n} samples, raw cm storage ≈ {cm_bytes / 1e6:.1f} MB, "
          f"total positives = {total_pos}")

    # ── sanity spot-check on first sample ──
    row0 = split_df.iloc[0]
    orig_pos = np.asarray(row0["contact_map_1"]).sum()
    packed_pos = contact_maps[0]["cm1"].sum()
    if orig_pos != packed_pos:
        print(f"  [WARN] sample-0 strand-1: orig={orig_pos}, packed={packed_pos}")

    return {
        "dna1": {k: v for k, v in dna1_tokens.items()},
        "dna2": {k: v for k, v in dna2_tokens.items()},
        "prot": {k: v for k, v in prot_tokens.items()},
        "contact_maps": contact_maps,
        "meta": {
            "pdb_ids": split_df["pdb_id"].tolist(),
            "prot_chain_ids": split_df["prot_chain_id"].tolist(),
            "dna_entity_types": split_df["dna_entity_type"].tolist(),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──
    print(f"Loading data from {args.data_path} …")
    df = load_data(args.data_path)
    print(f"  {len(df)} samples, columns: {df.columns.tolist()}")

    # ── split ──
    group_col = "cluster_id" if "cluster_id" in df.columns else "pdb_id"
    print(f"Splitting by '{group_col}' "
          f"(train={args.train_size:.0%}, val={args.val_size:.0%})")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size,
                           random_state=args.seed)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df[group_col]) & set(val_df[group_col])
    assert len(overlap) == 0, f"Group overlap between splits: {overlap}"
    print(f"  train={len(train_df)}, val={len(val_df)}, "
          f"groups: train={train_df[group_col].nunique()}, "
          f"val={val_df[group_col].nunique()}")

    # ── max lengths ──
    all_prot = train_df["prot_seq"].tolist() + val_df["prot_seq"].tolist()
    all_dna = (
        train_df["dna_seq_1"].tolist() + val_df["dna_seq_1"].tolist()
        + [s for s in (train_df["dna_seq_2"].tolist()
                       + val_df["dna_seq_2"].tolist())
           if isinstance(s, str) and s]
    )
    R = args.max_prot_len if args.max_prot_len > 0 else auto_max_len(all_prot, 1024)
    L = args.max_dna_len if args.max_dna_len > 0 else auto_max_len(all_dna, 512)
    print(f"  max_prot_len (R) = {R},  max_dna_len (L) = {L}")

    # ── tokenizers ──
    print(f"Loading tokenizers: {DNA_MODEL} + {PROT_MODEL}")
    dna_tok = AutoTokenizer.from_pretrained(DNA_MODEL, trust_remote_code=True)
    prot_tok = AutoTokenizer.from_pretrained(PROT_MODEL)

    # ── process ──
    train_data = process_split(train_df, dna_tok, prot_tok, R, L, "train")
    val_data = process_split(val_df, dna_tok, prot_tok, R, L, "val")

    # ── save ──
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    spec = {
        "max_prot_len": R,
        "max_dna_len": L,
        "dna_model": DNA_MODEL,
        "prot_model": PROT_MODEL,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "group_col": group_col,
        "seed": args.seed,
        "timestamp": ts,
        "data_path": str(args.data_path),
    }
    out_path = out_dir / f"contactmap_tokens_labels_dna{L}_prot{R}_{ts}.pt"
    print(f"\nSaving → {out_path}")
    torch.save({"spec": spec, "train": train_data, "val": val_data}, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
