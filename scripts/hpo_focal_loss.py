#!/usr/bin/env python3
"""Optuna hyperparameter optimisation for focal-loss parameters.

Wraps the existing focal-loss training scripts and tunes
``focal_alpha``, ``focal_gamma``, ``lr``, and ``warmup_epochs``
using Optuna.  Each trial runs the full training script as a
subprocess so no code is duplicated.

Supports parallel execution via SLURM job arrays.  When
``SLURM_ARRAY_TASK_COUNT`` is set, each array task runs its share
of trials.  All workers coordinate through Optuna's
``JournalFileStorage`` (NFS-safe, no SQLite locking issues).

HPO-level metrics (trial params + PR-AUC) are logged to a dedicated
W&B run via Optuna's ``WeightsAndBiasesCallback``, giving you
parallel-coordinate plots and parameter-importance charts alongside
the per-trial training runs.

Usage
-----
    # Local (sequential)
    python scripts/hpo_focal_loss.py \
        --config configs/training/composition_contactmap_focal_loss.yaml \
        --n_trials 40 --epochs 30

    # SLURM array (parallel) -- submitted via Makefile
    make submit-hpo ARRAY=8 EXTRA="--n_trials 40 --epochs 30"
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna_integration.wandb import WeightsAndBiasesCallback
import wandb
import yaml

SCRIPT_MAP = {
    "composition": "scripts/frameworks/composition_contactmap_focal_loss.py",
    "attention": "scripts/frameworks/attention_contactmap_focal_loss.py",
}

PR_AUC_RE = re.compile(r"PR-AUC=([\d.]+)")


def parse_pr_auc(stdout: str) -> float | None:
    """Extract the last PR-AUC value printed by the training script."""
    matches = PR_AUC_RE.findall(stdout)
    if matches:
        return float(matches[-1])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config (determines framework)")
    p.add_argument("--n_trials", type=int, default=30,
                   help="Total trials across all workers")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs per trial (shorter for HPO)")
    p.add_argument("--storage", type=str, default="optuna_focal.log",
                   help="Journal file for Optuna study persistence (NFS-safe)")
    p.add_argument("--study_name", type=str, default="focal_hpo")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    framework = cfg.get("framework", "composition")
    script = SCRIPT_MAP.get(framework)
    if script is None:
        sys.exit(f"Unknown framework '{framework}'. Expected: {list(SCRIPT_MAP)}")

    n_workers = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    worker_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    trials_this_worker = math.ceil(args.n_trials / n_workers)

    print(f"Worker {worker_id}/{n_workers}  |  "
          f"{trials_this_worker} trials (of {args.n_trials} total)  |  "
          f"framework={framework}")

    journal_path = Path(args.storage).resolve()
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    wandb_callback = WeightsAndBiasesCallback(
        metric_name="pr_auc",
        wandb_kwargs={
            "project": "BioLLMComposition",
            "group": "hpo",
            "name": f"hpo_{args.study_name}_w{worker_id}",
            "tags": ["hpo", framework],
            "config": {
                "study_name": args.study_name,
                "n_trials": args.n_trials,
                "n_workers": n_workers,
                "worker_id": worker_id,
                "framework": framework,
                "epochs": args.epochs,
            },
            "finish_previous": True,
        },
    )

    def objective(trial: optuna.Trial) -> float:
        # Previous grid search found optimum at alpha=0.80, gamma=1.0
        # (lower bounds of sweep). Ranges extend well below to let
        # Optuna explore whether less aggressive focal loss helps.
        alpha = trial.suggest_float("focal_alpha", 0.6, 0.99)
        gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        warmup = trial.suggest_int("warmup_epochs", 2, 15)

        cmd = [
            sys.executable, "-u", script, "--config", args.config,
            "--focal_alpha", str(alpha),
            "--focal_gamma", str(gamma),
            "--lr", str(lr),
            "--warmup_epochs", str(warmup),
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[Worker {worker_id} | Trial {trial.number}] "
                  f"FAILED (exit {result.returncode})")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            raise optuna.TrialPruned()

        pr_auc = parse_pr_auc(result.stdout)
        if pr_auc is None:
            print(f"[Worker {worker_id} | Trial {trial.number}] "
                  f"Could not parse PR-AUC from stdout")
            raise optuna.TrialPruned()

        print(f"[Worker {worker_id} | Trial {trial.number}] "
              f"alpha={alpha:.3f} gamma={gamma:.2f} "
              f"lr={lr:.2e} warmup={warmup} -> PR-AUC={pr_auc:.4f}")
        return pr_auc

    study.optimize(objective, n_trials=trials_this_worker,
                   callbacks=[wandb_callback])

    if worker_id == 0:
        print("\n" + "=" * 60)
        print(f"Study '{args.study_name}' — {len(study.trials)} total trials completed")
        try:
            best = study.best_trial
            print(f"Best trial (#{best.number}):")
            print(f"  PR-AUC: {best.value:.4f}")
            for k, v in best.params.items():
                print(f"  {k}: {v}")
        except ValueError:
            print("No completed trials yet.")
        print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    main()
