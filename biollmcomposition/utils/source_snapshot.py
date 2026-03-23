"""On-disk source snapshots for TensorBoard reproducibility."""

from __future__ import annotations

import importlib
from pathlib import Path


def _read_py(path: Path) -> str:
    path = path.resolve()
    return f"```python\n# {path}\n{path.read_text(encoding='utf-8')}\n```"


def log_architecture_sources(writer, framework_module, dna_info: dict, prot_info: dict) -> None:
    """Log framework + DNA/protein registry modules (embed_mode, layer defs, etc.)."""
    fw_path = Path(framework_module.__file__)
    writer.add_text("source/architecture", _read_py(fw_path))

    for key, info in (("dna_lm", dna_info), ("prot_lm", prot_info)):
        family = info["family"]
        mod = importlib.import_module(f"biollmcomposition.models.{family}")
        writer.add_text(f"source/{key}", _read_py(Path(mod.__file__)))
