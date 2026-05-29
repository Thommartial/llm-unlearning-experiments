"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (the latter two if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
