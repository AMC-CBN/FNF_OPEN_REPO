"""CLI for training hip-joint Faster R-CNN detectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from ..detection import (
    PickleHipDetectionDataset,
    TrainConfig,
    build_detection_transform,
    detection_collate,
    train_detection,
)

ALL_FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
VIEW_CLASSES = {
    "AP": ["Left", "Right"],
    "LAT": ["LAT_Neck"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the FNF *_Detection_data.pkl file")
    parser.add_argument("--view", choices=["AP", "LAT"], default="AP", help="View to train on")
    parser.add_argument(
        "--fold",
        default="all",
        help="Fold to hold out for validation (1..5 or 'all')",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--image-size", type=int, help="Optional resize/pad square edge length")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold for evaluation metrics")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for evaluation metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to store checkpoints (defaults to models/detection)",
    )
    parser.add_argument(
        "--save-best-metric",
        choices=["mean_iou", "precision"],
        default="mean_iou",
        help="Metric used to track the best checkpoint",
    )
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision training")
    parser.set_defaults(amp=True)
    return parser


def _parse_folds(value: str) -> List[int]:
    if value.lower() == "all":
        return [1, 2, 3, 4, 5]
    try:
        index = int(value)
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise ValueError("--fold must be an integer 1..5 or 'all'") from exc
    if not 1 <= index <= 5:
        raise ValueError("--fold must be between 1 and 5")
    return [index]


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    requested = _parse_folds(args.fold)
    transform = build_detection_transform(args.image_size)

    view = args.view.upper()
    view_lower = view.lower()
    classes = VIEW_CLASSES[view]

    base_cfg = TrainConfig(
        num_classes=len(classes) + 1,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        amp=args.amp,
        patience=args.patience,
        save_best_metric=args.save_best_metric,
        out_dir=str(args.out) if args.out else str(Path("models/detection")),
        name=f"detection_{view_lower}",
    )

    summaries: List[Dict[str, object]] = []
    for fold_index in requested:
        val_fold = ALL_FOLDS[fold_index - 1]
        train_folds = [fold for fold in ALL_FOLDS if fold != val_fold]

        cfg = replace(base_cfg, name=f"detection_{view_lower}_fold{fold_index}")

        train_ds = PickleHipDetectionDataset(
            args.dataset,
            train_folds,
            classes=classes,
            transform=transform,
            view_key=view,
        )
        val_ds = PickleHipDetectionDataset(
            args.dataset,
            [val_fold],
            classes=classes,
            transform=transform,
            view_key=view,
        )

        result = train_detection(train_ds, val_ds, cfg)
        best_metrics = result.get("best_metrics", {})
        print(json.dumps({
            "fold": fold_index,
            "best_epoch": result.get("best_epoch"),
            "best_metrics": best_metrics,
            "checkpoint": result.get("ckpt_path"),
        }, indent=2))
        summaries.append(result)

    if len(summaries) > 1:
        aggregated = {}
        for metric in ("precision", "recall", "mean_iou", "mAP"):
            values = [s.get("best_metrics", {}).get(metric) for s in summaries]
            values = [v for v in values if isinstance(v, (int, float))]
            if values:
                aggregated[metric] = sum(values) / len(values)
        if aggregated:
            print("Averaged best metrics:", json.dumps(aggregated, indent=2))

    return 0


__all__ = ["build_parser", "main"]
