"""Albumentations pipelines tailored for the Garden classification tasks."""

from __future__ import annotations
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = ["create_transforms"]

def create_transforms(task: str, augment: bool = True) -> Tuple[A.Compose, A.Compose]:
    """Create training and evaluation transforms for the classification tasks."""
    additional_targets = {"image_lat": "image", "image_ap_right": "image"}
    if augment:
        if "ap" in task.lower() and "lat" in task.lower():
            train_transforms = A.Compose(
                [A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.5), ToTensorV2()],
                additional_targets=additional_targets,
            )
        else:
            train_transforms = A.Compose([A.HorizontalFlip(p=0.5), ToTensorV2()], additional_targets=additional_targets)
    else:
        train_transforms = A.Compose([ToTensorV2()], additional_targets=additional_targets)
    eval_transforms = A.Compose([ToTensorV2()], additional_targets=additional_targets)
    return train_transforms, eval_transforms
