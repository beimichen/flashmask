"""Project-root-relative paths and reproducibility helpers.

Every path in the project is derived from :data:`PROJECT_ROOT` so the code runs
unchanged on any machine — no hardcoded ``/Users/...`` paths. Override any
location with an environment variable (e.g. ``FLASHMASK_DATA_DIR``) when needed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# src/flashmask/config.py -> parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, default: Path) -> Path:
    """Return an env-var override (if set) else the default, as an absolute Path."""
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Paths:
    """Canonical project directories, all relative to :data:`PROJECT_ROOT`."""

    root: Path = PROJECT_ROOT
    data: Path = field(
        default_factory=lambda: _resolve("FLASHMASK_DATA_DIR", PROJECT_ROOT / "data")
    )
    models: Path = field(
        default_factory=lambda: _resolve("FLASHMASK_MODELS_DIR", PROJECT_ROOT / "models")
    )
    reports: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    configs: Path = field(default_factory=lambda: PROJECT_ROOT / "configs")

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def data_interim(self) -> Path:
        return self.data / "interim"

    @property
    def data_processed(self) -> Path:
        return self.data / "processed"

    @property
    def data_external(self) -> Path:
        return self.data / "external"

    @property
    def data_sample(self) -> Path:
        return self.data / "sample"

    @property
    def fonts(self) -> Path:
        return self.data_external / "fonts"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"


paths = Paths()


def set_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs.

    Returns the seed so callers can log it. ``torch`` is imported lazily so this
    works without the optional ``[train]`` extras.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep, defensive only
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass  # torch is an optional extra; seeding the rest is still useful.

    return seed
