"""CLI for evaluating hip-joint Faster R-CNN detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..detection import (
    PickleHipDetectionDataset,
    TrainConfig,
    build_detection_transform,
    build_faster_rcnn,
    evaluate_detection_model,
    load_detection_checkpoint,
)

ALL_FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
CLASSES = ["Left", "Right"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the FNF *_Detection_data.pkl file")
    parser.add_argument("--view", choices=["AP", "LAT"], default="AP", help="View to evaluate")
    parser.add_argument(
        "--fold",
        default="all",
        help="Fold to evaluate (1..5 or 'all')",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--image-size", type=int, help="Optional resize/pad square edge length")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold for metrics")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to evaluate (use when --fold targets a single split)",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        help="Template containing {fold} placeholder for multi-fold evaluation",
    )
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

    folds = _parse_folds(args.fold)
    if len(folds) == 1 and args.checkpoint is None and args.checkpoint_template is None:
        parser.error("Provide --checkpoint or --checkpoint-template")
    if len(folds) > 1 and args.checkpoint_template is None:
        parser.error("--checkpoint-template is required when evaluating multiple folds")

    transform = build_detection_transform(args.image_size)
    cfg = TrainConfig(
        num_classes=len(CLASSES) + 1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )

    summaries: List[Dict[str, float]] = []
    for fold_index in folds:
        val_fold = ALL_FOLDS[fold_index - 1]
        dataset = PickleHipDetectionDataset(
            args.dataset,
            [val_fold],
            classes=CLASSES,
            transform=transform,
            view_key=args.view,
        )

        if args.checkpoint_template:
            ckpt_path = Path(args.checkpoint_template.format(fold=fold_index))
        else:
            ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for fold {fold_index}: {ckpt_path}")

        model = build_faster_rcnn(cfg.num_classes)
        load_detection_checkpoint(
            model,
            ckpt_path.name,
            checkpoint_dir=str(ckpt_path.parent),
            map_location="cpu",
        )
        metrics = evaluate_detection_model(model, dataset, cfg)
        print(json.dumps({"fold": fold_index, "checkpoint": str(ckpt_path), "metrics": metrics}, indent=2))
        summaries.append(metrics)

    if len(summaries) > 1:
        aggregated = {}
        for metric in ("precision", "recall", "mean_iou", "mAP"):
            values = [m.get(metric) for m in summaries if isinstance(m.get(metric), (int, float))]
            if values:
                aggregated[metric] = sum(values) / len(values)
        if aggregated:
            print("Average metrics:", json.dumps(aggregated, indent=2))

    return 0


__all__ = ["build_parser", "main"]
