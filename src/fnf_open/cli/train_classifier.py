"""Command line helpers for training Garden classification models."""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Sequence
import torch

from ..classification.detection_crops import build_detection_crops
from ..data.loading import load_pickle_dataset
from ..training.config import TrainingConfig
from ..training.cross_validation import aggregate_reports, train_model_cross_validation
from ..utils.seeding import set_random_seed
from ..utils.paths import DEFAULT_CLASSIFICATION_MODEL_DIR


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, help="Path to a pickle built by the preprocessing scripts")
    p.add_argument("--backbone", default="efficientnet_b0", help="timm backbone name (e.g. rexnet_100, resnet50)")
    p.add_argument("--epochs", type=int, default=15, help="Training epochs for each fold")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument(
        "--task",
        choices=("g12_vs_g34", "g3_vs_g4"),
        default="g12_vs_g34",
        help="Garden classification task to train",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=DEFAULT_CLASSIFICATION_MODEL_DIR,
        help="Directory for outputs",
    )
    p.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable ImageNet pretrained weights (useful for fully offline runs)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    p.add_argument(
        "--ap-detection-pickle",
        type=Path,
        help="Path to FNF_AP_Detection_data.pkl for detector-derived crops",
    )
    p.add_argument(
        "--lat-detection-pickle",
        type=Path,
        help="Path to FNF_LAT_Detection_data.pkl for detector-derived crops",
    )
    p.add_argument(
        "--ap-detection-checkpoint",
        type=Path,
        help="Checkpoint for the AP Faster R-CNN detector",
    )
    p.add_argument(
        "--lat-detection-checkpoint",
        type=Path,
        help="Checkpoint for the LAT Faster R-CNN detector",
    )
    p.add_argument(
        "--ap-detection-checkpoint-template",
        help="String template for AP detector checkpoints (use {fold})",
    )
    p.add_argument(
        "--lat-detection-checkpoint-template",
        help="String template for LAT detector checkpoints (use {fold})",
    )
    p.add_argument(
        "--detection-score-thr",
        type=float,
        default=0.3,
        help="Minimum detector score when selecting hip-joint boxes",
    )
    p.add_argument(
        "--detection-device",
        help="Device string for detector inference (defaults to --device)",
    )
    p.add_argument(
        "--use-ground-truth-crops",
        action="store_true",
        help="Reuse stored ground-truth crops instead of detector outputs",
    )
    p.set_defaults(pretrained=True)
    return p

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    set_random_seed(args.seed)
    raw = load_pickle_dataset(args.dataset)

    detected_crops = None
    if not args.use_ground_truth_crops:
        if args.ap_detection_pickle is None or args.lat_detection_pickle is None:
            parser.error(
                "Detector crops requested but detection pickles are missing. "
                "Provide both AP and LAT detection pickles or use --use-ground-truth-crops."
            )
        if args.ap_detection_checkpoint is None and args.ap_detection_checkpoint_template is None:
            parser.error(
                "Provide either --ap-detection-checkpoint or --ap-detection-checkpoint-template."
            )
        if args.lat_detection_checkpoint is None and args.lat_detection_checkpoint_template is None:
            parser.error(
                "Provide either --lat-detection-checkpoint or --lat-detection-checkpoint-template."
            )

        detection_device = str(args.detection_device or args.device)
        if detection_device.lower().startswith("cuda"):
            if not torch.cuda.is_available():
                logging.warning(
                    "CUDA requested for detector inference via '%s' but is unavailable; falling back to CPU",
                    detection_device,
                )
                detection_device = "cpu"
            else:
                _, _, index_str = detection_device.partition(":")
                if index_str:
                    try:
                        index = int(index_str)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid CUDA device '{detection_device}' for detection inference"
                        ) from exc
                    device_count = torch.cuda.device_count()
                    if index < 0 or index >= device_count:
                        raise ValueError(
                            f"Invalid CUDA device '{detection_device}' for detection inference. Available indices: 0..{device_count - 1}"
                        )
        detected_crops = build_detection_crops(
            ap_detection_pickle=args.ap_detection_pickle,
            lat_detection_pickle=args.lat_detection_pickle,
            ap_checkpoint=args.ap_detection_checkpoint,
            lat_checkpoint=args.lat_detection_checkpoint,
            ap_checkpoint_template=args.ap_detection_checkpoint_template,
            lat_checkpoint_template=args.lat_detection_checkpoint_template,
            score_thr=args.detection_score_thr,
            device=detection_device,
        )

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.wd,
        seed=args.seed,
        model_name=args.backbone,
        num_classes=1,
        task=args.task,
        use_pretrained=args.pretrained,
    )

    out_dir = args.out / args.backbone / args.task

    fold_reports = train_model_cross_validation(
        raw,
        cfg,
        num_folds=args.folds,
        device=args.device,
        checkpoint_dir=out_dir,
        detected_crops=detected_crops,
    )
    aggregate = aggregate_reports(fold_reports)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "backbone": args.backbone,
        "task": args.task,
        "folds": [fr.to_dict() for fr in fold_reports],
        "aggregate": aggregate,
    }, indent=2))
    print(json.dumps({"model": args.backbone, "task": args.task, **aggregate}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
