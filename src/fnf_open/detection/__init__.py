"""Detection utilities for hip-joint localisation."""

from .dataset import (
    PickleHipDetectionDataset,
    build_detection_transform,
    detection_collate,
)
from .metrics import evaluate_detection
from .models import build_faster_rcnn, load_faster_rcnn_checkpoint
from .training import TrainConfig, train_detection, evaluate_detection_model, load_detection_checkpoint

__all__ = [
    "PickleHipDetectionDataset",
    "build_detection_transform",
    "detection_collate",
    "evaluate_detection",
    "build_faster_rcnn",
    "load_faster_rcnn_checkpoint",
    "load_detection_checkpoint",
    "TrainConfig",
    "train_detection",
    "evaluate_detection_model",
]
