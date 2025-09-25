"""Model definitions for multi‑view fracture classification."""

from __future__ import annotations
from typing import List
import timm
import torch
from torch import nn

__all__ = ["MultiBranchClassifier", "LegacyMultiBranchClassifier"]

class MultiBranchClassifier(nn.Module):
    """Apply the same timm backbone independently to three image inputs.

    Each branch shares weights (created via :func:`timm.create_model`). The
    penultimate feature tensors are concatenated and fed to a small MLP head.

    Inputs are expected as:
      - `ap_right`: Tensor[N, C, H, W]
      - `ap_left`:  Tensor[N, C, H, W]
      - `lat`:      Tensor[N, C, H, W]
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 1,
        dropout: float = 0.0,
        pretrained: bool = True,
        num_views: int = 3,
    ) -> None:
        super().__init__()
        # Create a feature extractor that returns a pooled embedding
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.feature_extractor.num_features
        if num_views not in (2, 3):
            raise ValueError(f"Unsupported number of views: {num_views}. Expected 2 or 3.")
        self.num_views = num_views

        # Weight-shared branches: forward reuses the same module (weight sharing)
        self.branches: List[nn.Module] = nn.ModuleList(
            [self.feature_extractor for _ in range(num_views)]
        )

        head: List[nn.Module] = []
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(feature_dim * num_views, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(
        self,
        ap_right: torch.Tensor,
        ap_left: torch.Tensor,
        lat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs: List[torch.Tensor] = [ap_right, ap_left]
        if self.num_views == 3:
            if lat is None:
                raise ValueError("LAT tensor is required when num_views is 3")
            inputs.append(lat)
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        combined = torch.cat(features, dim=1)
        return self.classifier(combined)


class LegacyMultiBranchClassifier(nn.Module):
    """Legacy architecture with independent backbones and a final fusion layer."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    global_pool="avg",
                )
                for _ in range(3)
            ]
        )
        self.fc = nn.Linear(num_classes * 3, num_classes)

    def forward(self, ap_right: torch.Tensor, ap_left: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        logits = [
            branch(x)
            for branch, x in zip(self.branches, (ap_right, ap_left, lat))
        ]
        combined = torch.cat(logits, dim=1)
        return self.fc(combined)
