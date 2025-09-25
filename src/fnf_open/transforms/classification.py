"""Albumentations pipelines tailored for the Garden classification tasks."""

from __future__ import annotations
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = ["create_transforms"]

def create_transforms(
    task: str,
    augment: bool = True,
    *,
    include_lat: bool = True,
) -> Tuple[A.Compose, A.Compose]:
    """Create training and evaluation transforms for the classification tasks."""

    additional_targets = {"image_ap_right": "image"}
    if include_lat:
        additional_targets["image_lat"] = "image"

    train_ops = [A.HorizontalFlip(p=0.5)] if augment else []
    if augment and include_lat:
        train_ops.append(
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5,
            )
        )
    train_ops.append(ToTensorV2())

    train_transforms = A.Compose(train_ops, additional_targets=additional_targets)
    eval_transforms = A.Compose([ToTensorV2()], additional_targets=additional_targets)
    return train_transforms, eval_transforms
