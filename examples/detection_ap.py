\
"""
Launcher for Hip Joint Detection — AP View (Faster R-CNN).
This script uses the refactored `fnf_open` package to train or evaluate Faster R-CNN on AP/LAT views.
"""
import argparse
from pathlib import Path
from fnf_open.detection import (
    PickleHipDetectionDataset,
    TrainConfig,
    build_detection_transform,
    build_faster_rcnn,
    detection_collate,
    evaluate_detection_model,
    load_detection_checkpoint,
    train_detection,
)
from fnf_open.utils.paths import DEFAULT_DETECTION_MODEL_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to the FNF *_Detection_data.pkl")
    parser.add_argument("--view", type=str, default="AP", choices=["AP","LAT"], help="View to use")
    parser.add_argument(
        "--fold",
        type=str,
        default="1",
        help="Fold to hold out for validation (1..5 or 'all' to iterate across every fold)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path (supports {fold}); skips retraining and only evaluates",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs (ignored with --checkpoint)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_DETECTION_MODEL_DIR))
    parser.add_argument("--name", type=str, default="detection_ap")
    parser.add_argument("--image_size", type=int, default=None, help="Square size for resize/pad (optional)")
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    args = parser.parse_args()

    classes = ["Left","Right"]
    all_folds = ["fold1","fold2","fold3","fold4","fold5"]

    evaluate_only = args.checkpoint is not None
    if args.fold.lower() == "all" and evaluate_only and "{fold}" not in args.checkpoint:
        raise ValueError("When using --fold all with --checkpoint, include a {fold} placeholder in the checkpoint path")

    if evaluate_only:
        print("Evaluation-only mode detected; ignoring --epochs and training-specific settings.")

    if args.fold.lower() == "all":
        requested_folds = [1, 2, 3, 4, 5]
    else:
        try:
            fold_idx = int(args.fold)
        except ValueError as exc:  # pragma: no cover - CLI guard
            raise ValueError("--fold must be an integer 1..5 or 'all'") from exc
        if not 1 <= fold_idx <= 5:
            raise ValueError("--fold must be between 1 and 5 or 'all'")
        requested_folds = [fold_idx]

    def run_single_fold(fold_idx: int) -> dict:
        val_fold = all_folds[fold_idx - 1]
        tfm = build_detection_transform(args.image_size)
        val_ds = PickleHipDetectionDataset(
            args.pickle_path,
            [val_fold],
            classes=classes,
            transform=tfm,
            view_key=args.view,
        )
        val_ds.collate_fn = detection_collate

        model = build_faster_rcnn(num_classes=3, weights="DEFAULT")
        cfg_name = args.name if len(requested_folds) == 1 else f"{args.name}_fold{fold_idx}"
        cfg = TrainConfig(
            num_classes=3,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            out_dir=args.out_dir,
            name=cfg_name,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
        )
        if evaluate_only:
            checkpoint_template = args.checkpoint
            ckpt_path = Path(checkpoint_template.format(fold=fold_idx) if "{fold}" in checkpoint_template else checkpoint_template)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx}: {ckpt_path}")
            model, _ = load_detection_checkpoint(
                model,
                ckpt_path.name,
                checkpoint_dir=str(ckpt_path.parent),
                map_location="cpu",
            )
            metrics = evaluate_detection_model(model, val_ds, cfg)
            best_val = metrics.get(cfg.save_best_metric, 0.0)
            print(f"Fold {fold_idx} metrics (checkpoint {ckpt_path.name}): {metrics}")
            return {
                "best_metrics": metrics,
                "best_val": best_val,
                "best_epoch": None,
                "ckpt_path": str(ckpt_path),
            }

        train_folds = [f for f in all_folds if f != val_fold]
        train_ds = PickleHipDetectionDataset(
            args.pickle_path,
            train_folds,
            classes=classes,
            transform=tfm,
            view_key=args.view,
        )
        train_ds.collate_fn = detection_collate

        result = train_detection(train_ds, val_ds, cfg)
        best_metrics = result.get("best_metrics", {})
        best_epoch = result.get("best_epoch")
        print(f"Fold {fold_idx} best (epoch {best_epoch}): {best_metrics}")
        return result

    summaries = []
    for idx in requested_folds:
        summaries.append(run_single_fold(idx))

    if len(requested_folds) > 1:
        best_vals = [s.get("best_val") for s in summaries if isinstance(s.get("best_val"), (int, float))]
        metric_names = ("precision", "recall", "mean_iou", "mAP")
        aggregated = {}
        for name in metric_names:
            values = [s.get("best_metrics", {}).get(name) for s in summaries]
            values = [v for v in values if isinstance(v, (int, float))]
            if values:
                aggregated[name] = sum(values) / len(values)
        if best_vals:
            mean_best = sum(best_vals) / len(best_vals)
            print(f"Mean monitored metric across {len(best_vals)} folds: {mean_best:.4f}")
        if aggregated:
            print("Mean best metrics:", aggregated)

if __name__ == "__main__":
    main()
