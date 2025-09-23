"""Model builders for detection."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..utils.paths import detection_checkpoint_path

__all__ = ["build_faster_rcnn", "load_faster_rcnn_checkpoint"]


def build_faster_rcnn(num_classes: int, weights: Optional[str] = "DEFAULT") -> torch.nn.Module:
    """Create a Faster R-CNN (ResNet50-FPN) with the correct classifier head."""

    if weights == "DEFAULT":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif weights is None:
        model = fasterrcnn_resnet50_fpn(weights=None)
    else:
        model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_faster_rcnn_checkpoint(
    checkpoint_name: str,
    *,
    num_classes: int,
    weights: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.nn.Module, Dict[str, object] | None]:
    """Build and load a Faster R-CNN model from a saved checkpoint."""

    model = build_faster_rcnn(num_classes=num_classes, weights=weights)
    ckpt_path = detection_checkpoint_path(checkpoint_name, directory=checkpoint_dir)
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model"])
    return model, state.get("cfg")
