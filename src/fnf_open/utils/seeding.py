"""Seeding helpers used across training scripts and examples."""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["set_random_seed"]


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on environment
        torch.cuda.manual_seed_all(seed)
