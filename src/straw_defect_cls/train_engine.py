"""Training engine for Straw Defect Classification (Phase 2).
# -*- coding: utf-8 -*-

Supports hierarchical two-stage classification and 5-class baseline:
- Backbone: resnet18, swin_t, vmamba_t
- Stage: 1 (Good vs Defect), 2 (4 Defect classes), full5 (5-class baseline)
- Resolution: 224, 384
- Best weight selection:
  * Stage 1: Defect recall at FPR <= 5% (penalizing false negatives)
  * Stage 2 & Full5: Macro recall across defect / all classes
- Losses & Scheduling:
  * Stage 1: Class-weighted CrossEntropy (penalizing FN)
  * Stage 2 & Full5: Label smoothing (0.1 / 0.05) + CosineAnnealingLR
- Logging: experiments/runs/<tag>/history.csv

Usage:
    .venv/Scripts/python.exe -m src.straw_defect_cls.train_engine --backbone resnet18 --stage 1 --resolution 224 --seed 42
    .venv/Scripts/python.exe -m src.straw_defect_cls.train_engine --backbone resnet18 --stage 2 --resolution 224 --seed 42
    .venv/Scripts/python.exe -m src.straw_defect_cls.train_engine --backbone resnet18 --stage full5 --resolution 224 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_curve
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    STAGE1_CLASS_NAMES,
    STAGE2_CLASS_NAMES,
    FULL_CLASS_NAMES,
    StrawDefectDataset,
    build_dataloaders,
    get_train_transform,
    get_val_transform,
)
from .losses import CrossEntropyWithLabelSmoothing, build_criterion
from .metrics import compute_metrics, plot_confusion_matrix, plot_roc_curves
from .models import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """All hyper-parameters and paths for a training run."""
    # Task / Stage: 'stage1' (Good vs Defect), 'stage2' (4 Defect classes), 'full5' (5-class baseline)
    stage: str = "full5"
    task: str = "full"
    # Data paths & parameters
    data_dir: str = ""
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    masks_csv: str = "data/masks/masks.csv"
    image_size: int = 224
    # Architecture
    model_name: str = "resnet18"
    num_classes: int = 5
    pretrained: bool = True
    # Training hyperparameters
    epochs: int = 25
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-3
    encoder_lr: float = 1e-4
    classifier_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_accum: int = 1
    grad_clip: float = 1.0
    use_amp: bool = True
    loss_name: str = "ce"
    label_smoothing: float = 0.0
    scheduler_type: str = "cosine"  # 'cosine' or 'plateau'
    seed: int = 42
    device: str = "cuda"
    # Early stopping & checkpoints
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4
    # Run tag and directories
    output_dir: str = "experiments/runs"
    tag: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def environment_info(device: torch.device) -> dict[str, Any]:
    """Capture runtime environment info for auditing."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / (1024**3), 2
        )
    return info


def compute_defect_recall_at_fpr5(y_true: list[int] | np.ndarray, y_prob: Optional[np.ndarray]) -> float:
    """Calculate Defect Recall at False Positive Rate (FPR) <= 5% for Stage 1.

    Args:
        y_true: Binary ground truth (0: Good, 1: Defect).
        y_prob: Array of predicted probabilities shape (N, 2).

    Returns:
        Defect recall (TPR) value achieved at the optimal operating point where FPR <= 5%.
    """
    if y_prob is None or len(y_prob) == 0:
        return 0.0
    y_true_arr = np.array(y_true)
    # Probability of defect (class 1)
    p_defect = y_prob[:, 1]
    fpr, tpr, _ = roc_curve(y_true_arr, p_defect)
    valid_mask = fpr <= 0.05
    if np.any(valid_mask):
        return float(np.max(tpr[valid_mask]))
    return float(tpr[0])


def compute_macro_recall(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    """Calculate macro-averaged recall across all classes."""
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


# ──────────────────────────────────────────────────────────────────────────────
# Training & Validation Loops
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Train for a single epoch."""
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train [{cfg.model_name}]", leave=False)
    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels) / cfg.grad_accum

        scaler.scale(loss).backward()

        if step % cfg.grad_accum == 0 or step == len(loader):
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * cfg.grad_accum
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{running_loss / step:.4f}")

    metrics = compute_metrics(all_labels, all_preds, None, getattr(cfg, "_class_names", CLASS_NAMES))
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Validate model and collect predictions and probabilities."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy())

    probs_arr = np.array(all_probs)
    metrics = compute_metrics(all_labels, all_preds, probs_arr, getattr(cfg, "_class_names", CLASS_NAMES))
    metrics["loss"] = running_loss / max(1, len(loader))
    metrics["_y_true"] = all_labels
    metrics["_y_pred"] = all_preds
    metrics["_y_prob"] = probs_arr
    return metrics


def plot_training_curves(history: list[dict], save_dir: Path) -> None:
    """Save training curves for loss, accuracy, and primary metric."""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [h["train_acc"] for h in history], "o-", label="Train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], "s-", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Primary Decision Metric
    metric_name = history[0].get("metric_name", "Primary Metric")
    axes[2].plot(epochs, [h["val_primary_metric"] for h in history], "o-", color="purple", label=f"Val {metric_name}")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel(metric_name)
    axes[2].set_title(f"Checkpoint Criterion: {metric_name}")
    axes[2].set_ylim(0, 1.02)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline Function
# ──────────────────────────────────────────────────────────────────────────────
def run_training(cfg: TrainConfig) -> Path:
    """Execute training run according to industrial specifications."""
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # Normalize stage and task
    stage_str = str(cfg.stage).lower().strip()
    if stage_str in ("1", "stage1"):
        task = "stage1"
        class_names = STAGE1_CLASS_NAMES
        cfg.num_classes = 2
        metric_name = "defect_recall_at_fpr5"
    elif stage_str in ("2", "stage2"):
        task = "stage2"
        class_names = STAGE2_CLASS_NAMES
        cfg.num_classes = 4
        metric_name = "macro_recall"
    else:
        task = "full"
        stage_str = "full5"
        class_names = FULL_CLASS_NAMES
        cfg.num_classes = 5
        metric_name = "macro_recall"

    cfg.task = task
    cfg.stage = stage_str
    cfg._class_names = class_names

    # Resolve Run Tag and Directory
    tag = cfg.tag or f"{cfg.model_name}_stage{stage_str}_r{cfg.image_size}_seed{cfg.seed}"
    run_dir = Path(cfg.output_dir) / tag
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resolve dataset CSVs if not provided
    splits_dir = REPO_ROOT / "data" / "straw_defects_splits"
    if not cfg.train_csv:
        if task == "stage1":
            cfg.train_csv = str(splits_dir / "train_stage1.csv")
            cfg.val_csv = str(splits_dir / "val_stage1.csv")
        elif task == "stage2":
            cfg.train_csv = str(splits_dir / "train_stage2.csv")
            cfg.val_csv = str(splits_dir / "val_stage2.csv")
        else:
            cfg.train_csv = str(splits_dir / "train.csv")
            cfg.val_csv = str(splits_dir / "val.csv")

    # Save initial config and environment
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(device), indent=2), encoding="utf-8"
    )

    # DataLoaders
    masks_path = REPO_ROOT / cfg.masks_csv
    loaders = build_dataloaders(
        train_csv=cfg.train_csv,
        val_csv=cfg.val_csv,
        masks_csv=masks_path if masks_path.exists() else None,
        task=task,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Model instantiation (with VMamba fallback protection)
    try:
        model = build_model(cfg.model_name, num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    except Exception as exc:
        if "vmamba" in cfg.model_name.lower():
            print(f"[WARN] Failed to instantiate VMamba ({exc}). Skipping gracefully.")
            raise RuntimeError(f"VMamba unavailable: {exc}")
        raise exc

    model = model.to(device)

    # Loss function configuration
    if task == "stage1":
        # Class-weighted CE penalizing False Negatives heavily (Good=0, Defect=1)
        weight = torch.tensor([1.0, 3.0], device=device, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif task == "stage2":
        # Stage 2: 4 balanced classes with label smoothing 0.1
        criterion = CrossEntropyWithLabelSmoothing(num_classes=4, smoothing=0.1).to(device)
    else:
        # Full5 baseline: label smoothing 0.05
        criterion = CrossEntropyWithLabelSmoothing(num_classes=5, smoothing=0.05).to(device)

    # Optimizer & Scheduler
    encoder_params = list(model.encoder_parameters()) if hasattr(model, "encoder_parameters") else []
    classifier_params = list(model.classifier_parameters()) if hasattr(model, "classifier_parameters") else []
    if encoder_params and classifier_params:
        param_groups = [
            {"params": encoder_params, "lr": cfg.encoder_lr, "name": "encoder"},
            {"params": classifier_params, "lr": cfg.classifier_lr, "name": "classifier"},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": cfg.learning_rate}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    # Training loop state
    best_metric = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"RUN: {tag} | Model: {cfg.model_name} | Stage: {stage_str} | Res: {cfg.image_size} | Seed: {cfg.seed}")
    print(f"Best weight tracked by: {metric_name}")
    print(f"{'='*70}")

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg)
        val_metrics = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start

        # Compute Primary Evaluation Metric for Checkpointing
        if task == "stage1":
            primary_metric = compute_defect_recall_at_fpr5(val_metrics["_y_true"], val_metrics["_y_prob"])
        else:
            primary_metric = compute_macro_recall(val_metrics["_y_true"], val_metrics["_y_pred"])

        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "train_f1": train_metrics["weighted_f1"],
            "val_f1": val_metrics["weighted_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_primary_metric": primary_metric,
            "metric_name": metric_name,
            "epoch_seconds": round(epoch_seconds, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)

        # Log history to CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(run_dir / "history.csv", index=False)
        history_df.to_csv(run_dir / "training_history.csv", index=False)
        plot_training_curves(history, run_dir / "curves")

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "primary_metric": primary_metric,
            "val_f1": val_metrics["weighted_f1"],
        }, ckpt_dir / "last.pt")

        # Checkpoint evaluation
        is_best = primary_metric > best_metric + cfg.early_stopping_min_delta
        if is_best:
            best_metric = primary_metric
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "config": asdict(cfg),
                "primary_metric": best_metric,
                "metric_name": metric_name,
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }, ckpt_dir / "best.pt")
            save_star = f"[BEST] ({metric_name}={best_metric:.4f})"

            # Save best confusion matrix and ROC
            plot_confusion_matrix(
                val_metrics["confusion_matrix"], class_names,
                run_dir / "best_confusion_matrix.png"
            )
            if val_metrics.get("_y_prob") is not None:
                plot_roc_curves(
                    val_metrics["_y_true"], val_metrics["_y_prob"],
                    class_names, run_dir / "best_roc_curves.png"
                )
        else:
            bad_epochs += 1
            save_star = ""

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} [{epoch_seconds:.1f}s] | "
            f"TrainLoss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.3f} | "
            f"ValLoss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.3f} "
            f"{metric_name}={primary_metric:.4f} {save_star}"
        )

        if bad_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    total_seconds = time.perf_counter() - total_start
    summary = {
        "tag": tag,
        "backbone": cfg.model_name,
        "stage": stage_str,
        "resolution": cfg.image_size,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "metric_name": metric_name,
        "best_metric": best_metric,
        "epochs_completed": len(history),
        "total_seconds": round(total_seconds, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] {tag} completed in {total_seconds:.1f}s | Best {metric_name}: {best_metric:.4f}")
    return run_dir


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical/baseline defect classifier.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "swin_t", "vmamba_t"],
        help="Model backbone architecture",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="full5",
        choices=["1", "2", "full5", "stage1", "stage2"],
        help="Classification stage: 1 (binary triage), 2 (4 defect classes), or full5 (5-class baseline)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        choices=[224, 384],
        help="Input image resolution",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--tag", type=str, default="", help="Custom tag for run folder")
    parser.add_argument("--output-dir", type=str, default="experiments/runs", help="Output directory")

    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.backbone,
        stage=args.stage,
        image_size=args.resolution,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        tag=args.tag,
        output_dir=args.output_dir,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
