"""Train and evaluate all models (ResNet18, EfficientNet-B0, MobileNetV3) sequentially.

This is the master launcher — equivalent to the old train_on_kaggle.py but for
the new classification task. Works both locally and on Kaggle.

Usage (Kaggle):
    python train_all_models.py \\
        --data-dir /kaggle/input/juice-box-straw-defects \\
        --output-dir /kaggle/working/results

Usage (local, CSV-based):
    python train_all_models.py \\
        --train-csv data/splits/train.csv \\
        --val-csv   data/splits/val.csv \\
        --test-csv  data/splits/test.csv \\
        --output-dir results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from straw_defect_cls.train_engine import TrainConfig, run_training, validate
from straw_defect_cls.dataset import build_dataloaders, CLASS_NAMES, NUM_CLASSES
from straw_defect_cls.models import build_model
from straw_defect_cls.losses import build_criterion
from straw_defect_cls.metrics import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_classification_report,
)

ALL_MODELS = ["resnet18", "efficientnet_b0", "mobilenet_v3"]


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    split_name: str,
    loader,
    device: torch.device,
    output_dir: Path,
    cfg: TrainConfig,
) -> dict:
    """Load checkpoint and evaluate on a given split."""
    model = build_model(model_name, num_classes=cfg.num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device)

    criterion = build_criterion(cfg.loss_name)
    metrics = validate(model, loader, criterion, device, cfg)

    eval_dir = output_dir / model_name / cfg.run_name / f"eval_{split_name}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    serializable = {k: v for k, v in metrics.items() if not k.startswith("_")}
    (eval_dir / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, eval_dir / "confusion_matrix.png")
    if metrics.get("_y_prob") is not None:
        plot_roc_curves(metrics["_y_true"], metrics["_y_prob"], CLASS_NAMES, eval_dir / "roc_curves.png")

    print(f"  [{model_name}] {split_name}: acc={metrics['accuracy']:.4f}  "
          f"f1={metrics['weighted_f1']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    return serializable


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate all straw defect classifiers")
    parser.add_argument("--model", choices=["all"] + ALL_MODELS, default="all")
    parser.add_argument("--data-dir", type=str, default="", help="Dataset root (folder-based)")
    parser.add_argument("--train-csv", type=str, default="")
    parser.add_argument("--val-csv", type=str, default="")
    parser.add_argument("--test-csv", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default="kaggle_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    models_to_train = ALL_MODELS if args.model == "all" else [args.model]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    summary_rows = []
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {model_name}")
        print(f"{'='*60}")

        cfg = TrainConfig(
            model_name=model_name,
            data_dir=args.data_dir,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            run_name=args.run_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            seed=args.seed,
            device=args.device,
        )
        run_dir = run_training(cfg)
        best_ckpt = run_dir / "checkpoints" / "best.pt"

        if not best_ckpt.exists():
            print(f"  ⚠ No best checkpoint found for {model_name}, skipping evaluation.")
            continue

        # Evaluate on val
        loaders = build_dataloaders(
            data_dir=cfg.data_dir or None,
            train_csv=cfg.train_csv or None,
            val_csv=cfg.val_csv or None,
            test_csv=cfg.test_csv or None,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        row = {"model": model_name}
        if "val" in loaders:
            val_metrics = evaluate_checkpoint(
                model_name, best_ckpt, "val", loaders["val"], device,
                Path(args.output_dir), cfg,
            )
            row["val_acc"] = val_metrics["accuracy"]
            row["val_f1"] = val_metrics["weighted_f1"]

        if not args.skip_test and "test" in loaders:
            test_metrics = evaluate_checkpoint(
                model_name, best_ckpt, "test", loaders["test"], device,
                Path(args.output_dir), cfg,
            )
            row["test_acc"] = test_metrics["accuracy"]
            row["test_f1"] = test_metrics["weighted_f1"]

        summary_rows.append(row)

    # Print final summary
    if summary_rows:
        print(f"\n{'='*60}")
        print("  FINAL SUMMARY")
        print(f"{'='*60}")
        df = pd.DataFrame(summary_rows)
        print(df.to_string(index=False))
        df.to_csv(Path(args.output_dir) / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
