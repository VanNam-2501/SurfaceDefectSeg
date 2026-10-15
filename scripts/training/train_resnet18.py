"""Train ResNet-18 for Straw Defect Classification."""
import sys
import argparse
from pathlib import Path

# Add src to python path
_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from straw_defect_cls.train_engine import TrainConfig, run_training


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 classifier")
    parser.add_argument("--data-dir", type=str, default="", help="Dataset root with class subfolders")
    parser.add_argument("--train-csv", type=str, default="", help="Train split CSV")
    parser.add_argument("--val-csv", type=str, default="", help="Val split CSV")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default="resnet18_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name="resnet18",
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
