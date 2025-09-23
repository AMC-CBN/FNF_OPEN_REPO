\
"""
Launcher for Classification: Garden III vs IV (classification).
Minimal ResNet50 training on a folder structure:
  data_dir/train/<class>/*.jpg
  data_dir/val/<class>/*.jpg
"""
import argparse
from fnf_open.train_classification import ClsConfig, run_classification
from fnf_open.utils.paths import DEFAULT_CLASSIFICATION_MODEL_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_CLASSIFICATION_MODEL_DIR))
    parser.add_argument("--name", type=str, default="train_garden_iii_vs_iv")
    args = parser.parse_args()

    cfg = ClsConfig(
        num_classes=args.num_classes,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        name=args.name,
    )
    result = run_classification(cfg)
    print("Best:", result)

if __name__ == "__main__":
    main()
