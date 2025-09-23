"""Command line helpers for training Garden classification models."""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Sequence
import torch

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
    p.set_defaults(pretrained=True)
    return p

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    set_random_seed(args.seed)
    raw = load_pickle_dataset(args.dataset)
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
