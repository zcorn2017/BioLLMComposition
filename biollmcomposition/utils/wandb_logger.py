"""Centralised W&B experiment-tracking helpers.

Every training script calls :func:`init_run` once per run.  Helper
functions handle scalar logging, artifact uploads, and best-metric
summaries so that the training scripts stay concise.

Offline mode (``WANDB_MODE=offline``) is set in the sbatch scripts for
SLURM clusters that lack internet.  After pulling results, run
``wandb sync`` locally to upload.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import wandb


def init_run(
    cfg: dict,
    run_tag: str,
    run_idx: int,
    framework_name: str,
    *,
    dna_short: str = "",
    prot_short: str = "",
    loss_tag: str = "bce",
) -> Any:
    """Start a W&B run with structured project/group/tags."""
    loss_short = "focal" if loss_tag.startswith("focal") else loss_tag
    tags = [framework_name, loss_short]
    if dna_short:
        tags.append(dna_short)
    if prot_short:
        tags.append(prot_short)

    run = wandb.init(
        project="BioLLMComposition",
        group=framework_name,
        name=f"{run_tag}/run{run_idx}",
        tags=tags,
        config=cfg,
        reinit="finish_previous",
    )
    return run


def log_scalars(
    epoch: int,
    train_loss: float,
    metrics: dict,
    *,
    lr: float | None = None,
) -> None:
    """Log per-epoch scalars."""
    data: dict[str, Any] = {
        "epoch": epoch,
        "loss/train": train_loss,
        "loss/val": metrics["val_loss"],
    }
    if lr is not None:
        data["lr"] = lr
    for k, v in metrics.items():
        if k != "val_loss":
            data[f"metric/{k}"] = v
    wandb.log(data, step=epoch)


def log_best_metrics(best_metrics: dict) -> None:
    """Write best-run metrics to the run summary for easy comparison."""
    for k in ("pr_auc", "roc_auc", "mcc", "precision",
              "recall", "f1", "top_L_precision"):
        wandb.summary[f"best/{k}"] = best_metrics.get(k, 0)


def log_source_artifacts(
    framework_module: Any,
    dna_info: dict,
    prot_info: dict,
    training_script: str | Path,
    config_path: str | Path,
) -> None:
    """Save source files as W&B artifacts for reproducibility."""
    artifact = wandb.Artifact("source_code", type="code")

    artifact.add_file(str(Path(framework_module.__file__).resolve()),
                      name="architecture.py")

    for key, info in (("dna_lm", dna_info), ("prot_lm", prot_info)):
        family = info["family"]
        mod = importlib.import_module(f"biollmcomposition.models.{family}")
        artifact.add_file(str(Path(mod.__file__).resolve()),
                          name=f"{key}.py")

    artifact.add_file(str(Path(training_script).resolve()),
                      name="training_script.py")
    artifact.add_file(str(Path(config_path).resolve()),
                      name="config.yaml")

    from biollmcomposition.utils import contact_map as _cm
    artifact.add_file(str(Path(_cm.__file__).resolve()),
                      name="contact_map.py")

    wandb.log_artifact(artifact)


def log_checkpoint(ckpt_path: str | Path) -> None:
    """Upload best checkpoint as a W&B artifact."""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)


def finish() -> None:
    """End the current W&B run."""
    wandb.finish()
