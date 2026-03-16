#!/usr/bin/env python3
"""Split a precomputed token dataset into train / val / test subsets.

Produces a lightweight ``.pt`` file containing only the split indices
and metadata.  The training script (``02_run_training.py``) loads *both*
the token file and the split file and combines them at runtime.

Usage
-----
    python scripts/01_split_datasets.py \
        --config configs/splitting/default.yaml

    # Override via CLI:
    python scripts/01_split_datasets.py \
        --config configs/splitting/default.yaml \
        --data_pt /path/to/tokens.pt --seed 123
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit

from biollmcomposition.utils.contact_map import resolve_data_path, set_seed


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--data_pt", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_config(args) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_pt:
        cfg["data_pt"] = args.data_pt
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


# ── Splitting logic ──────────────────────────────────────────────────────

def _resolve_groups(meta: dict, group_col: str) -> tuple[np.ndarray, str]:
    """Find a valid group array from the data's meta dict.

    Returns (groups_array, actual_column_name).
    """
    candidates = {
        "cluster_id": "cluster_ids",
        "cluster_ids": "cluster_ids",
        "pdb_id": "pdb_ids",
        "pdb_ids": "pdb_ids",
    }
    key = candidates.get(group_col, group_col)
    if key in meta:
        return np.array(meta[key]), key.rstrip("s")

    for fallback_key, fallback_col_name in [
        ("cluster_ids", "cluster_id"),
        ("pdb_ids", "pdb_id"),
    ]:
        if fallback_key in meta:
            print(f"  [INFO] '{group_col}' not found, falling back to "
                  f"'{fallback_col_name}'")
            return np.array(meta[fallback_key]), fallback_col_name

    raise ValueError(
        f"No valid group column for '{group_col}'. "
        f"Available meta keys: {list(meta)}"
    )


def split_group_shuffle(n: int, groups: np.ndarray,
                        train_r: float, val_r: float, test_r: float,
                        seed: int):
    """Two-stage GroupShuffleSplit → (train_idx, val_idx, test_idx)."""
    indices = np.arange(n)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_r,
                            random_state=seed)
    trainval_idx, test_idx = next(gss1.split(indices, groups=groups))

    val_frac = val_r / (train_r + val_r)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac,
                            random_state=seed)
    sub_tr, sub_va = next(
        gss2.split(trainval_idx, groups=groups[trainval_idx]),
    )
    train_idx = trainval_idx[sub_tr]
    val_idx = trainval_idx[sub_va]
    return train_idx, val_idx, test_idx


def split_random(n: int, train_r: float, val_r: float, seed: int):
    """Simple random split → (train_idx, val_idx, test_idx)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_pt_path = resolve_data_path(cfg["data_pt"])
    print(f"Loading precomputed data from {data_pt_path} …")
    data = torch.load(data_pt_path, map_location="cpu", weights_only=False)
    data_spec = data["spec"]
    meta = data.get("meta", {})
    n = data_spec["n_samples"]
    print(f"  {n} samples, DNA={data_spec['dna_model_short']}, "
          f"Prot={data_spec['prot_model_short']}")

    strategy = cfg.get("strategy", "group_shuffle")
    group_col = cfg.get("group_col", "cluster_id")
    ratios = cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    train_r = ratios["train"]
    val_r = ratios["val"]
    test_r = ratios.get("test", round(1.0 - train_r - val_r, 6))

    assert abs(train_r + val_r + test_r - 1.0) < 1e-4, (
        f"Ratios must sum to 1.0, got {train_r + val_r + test_r:.6f}"
    )

    actual_group_col: str | None = None

    if strategy == "group_shuffle":
        groups, actual_group_col = _resolve_groups(meta, group_col)
        print(f"Strategy: {strategy}, group_col: {actual_group_col}, "
              f"unique groups: {len(np.unique(groups))}")
        train_idx, val_idx, test_idx = split_group_shuffle(
            n, groups, train_r, val_r, test_r, seed,
        )

        # Verify no group leakage
        tg = set(groups[train_idx])
        vg = set(groups[val_idx])
        teg = set(groups[test_idx])
        assert not (tg & vg), "Group leakage: train ∩ val"
        assert not (tg & teg), "Group leakage: train ∩ test"
        assert not (vg & teg), "Group leakage: val ∩ test"
        print(f"  Groups — train: {len(tg)}, val: {len(vg)}, test: {len(teg)}")

    elif strategy == "random":
        print(f"Strategy: {strategy}")
        train_idx, val_idx, test_idx = split_random(
            n, train_r, val_r, seed,
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    print(f"  Samples — train: {len(train_idx)}, val: {len(val_idx)}, "
          f"test: {len(test_idx)}")

    # ── Save ──
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(resolve_data_path(cfg.get("out_dir", str(Path(data_pt_path).parent))))
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_str = f"{int(train_r * 100)}-{int(val_r * 100)}-{int(test_r * 100)}"
    gc_tag = actual_group_col or "random"
    out_name = f"split_{strategy}_{gc_tag}_s{seed}_{ratio_str}_{ts}.pt"
    out_path = out_dir / out_name

    split_spec = {
        "strategy": strategy,
        "group_col": actual_group_col,
        "seed": seed,
        "ratios": {"train": train_r, "val": val_r, "test": test_r},
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "timestamp": ts,
    }

    payload = {
        "spec": split_spec,
        "source_pt": str(data_pt_path),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

    print(f"\nSaving → {out_path}")
    torch.save(payload, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
