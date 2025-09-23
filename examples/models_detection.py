\
from typing import Dict, Optional, Tuple, Union

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .train_detection import load_detection_checkpoint


def build_faster_rcnn(num_classes: int, weights: Optional[str] = "DEFAULT") -> torch.nn.Module:
    """
    Create a Faster R-CNN (ResNet50-FPN) with correct classifier head for `num_classes` (including background).
    """
    if weights == "DEFAULT":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif weights is None:
        model = fasterrcnn_resnet50_fpn(weights=None)
    else:
        # pass-through: if user passes a valid weight enum name they can map before calling us.
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
    model, cfg = load_detection_checkpoint(
        model,
        checkpoint_name,
        checkpoint_dir=checkpoint_dir,
        map_location=map_location,
    )
    return model, cfg
