"""Utility helpers specific to detection training."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["seed_everything", "get_device", "EarlyStopping", "timestamp"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "max"
    min_delta: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStopping.mode must be 'min' or 'max'")
        self.best = -float("inf") if self.mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
