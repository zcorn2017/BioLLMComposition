"""Contact-map prediction framework registry.

Each framework module exposes a ``build_model`` function with a uniform
signature so the training script can dispatch by name.
"""

from __future__ import annotations

import importlib

FRAMEWORK_REGISTRY: dict[str, str] = {
    "attention": "biollmcomposition.frameworks.attention",
    "composition": "biollmcomposition.frameworks.composition",
}


def get_framework(name: str):
    """Import and return the framework module by name."""
    if name not in FRAMEWORK_REGISTRY:
        raise ValueError(
            f"Unknown framework '{name}'. Available: {sorted(FRAMEWORK_REGISTRY)}"
        )
    return importlib.import_module(FRAMEWORK_REGISTRY[name])
