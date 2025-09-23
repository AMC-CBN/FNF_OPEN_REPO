"""Configuration dataclasses used by the training routines."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

__all__ = ["TrainingConfig", "FoldReport"]

@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42
    model_name: str = "efficientnet_b0"
    num_classes: int = 1
    task: str = "g12_vs_g34"
    use_pretrained: bool = True

@dataclass
class FoldReport:
    fold_index: int
    best_epoch: int
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2,2), dtype=int))

    def to_dict(self) -> Dict[str, object]:
        return {
            "fold": self.fold_index + 1,
            "best_epoch": self.best_epoch,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix.astype(int).tolist(),
        }
