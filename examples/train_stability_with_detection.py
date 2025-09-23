\
"""
Launcher for Training Stability Study With Detection.
This script uses the refactored `fnf_open` package to train Faster R-CNN on AP/LAT views.
"""
import argparse
from fnf_open.data_detection import PickleFNFAPDataset, build_default_transform, detection_collate
from fnf_open.models_detection import build_faster_rcnn
from fnf_open.train_detection import TrainConfig, run_training
from fnf_open.utils.paths import DEFAULT_DETECTION_MODEL_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to the FNF *_Detection_data.pkl")
    parser.add_argument("--view", type=str, default="AP", choices=["AP","LAT"], help="View to use")
    parser.add_argument("--fold", type=int, default=1, help="Fold to hold out for validation (1..5)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_DETECTION_MODEL_DIR))
    parser.add_argument("--name", type=str, default="train_stability_with_detection")
    parser.add_argument("--image_size", type=int, default=None, help="Square size for resize/pad (optional)")
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    args = parser.parse_args()

    classes = ["Left","Right"]
    all_folds = ["fold1","fold2","fold3","fold4","fold5"]
    assert 1 <= args.fold <= 5, "fold must be 1..5"
    val_fold = all_folds[args.fold-1]
    train_folds = [f for f in all_folds if f != val_fold]

    tfm = build_default_transform(image_size=args.image_size)
    train_ds = PickleFNFAPDataset(args.pickle_path, train_folds, classes=classes, transform=tfm, view_key=args.view)
    val_ds = PickleFNFAPDataset(args.pickle_path, [val_fold], classes=classes, transform=tfm, view_key=args.view)
    # attach collate
    train_ds.collate_fn = detection_collate
    val_ds.collate_fn = detection_collate

    model = build_faster_rcnn(num_classes=3, weights="DEFAULT")

    cfg = TrainConfig(
        num_classes=3,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        name=args.name,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
    )

    result = run_training(model, train_ds, val_ds, cfg)
    print("Best:", result)

if __name__ == "__main__":
    main()
