"""Training engine for Straw Defect Classification.

Complete training pipeline with:
- Mixed-precision (AMP) training
- Gradient accumulation
- ReduceLROnPlateau scheduler
- Early stopping
- Checkpoint saving (best.pt / last.pt)
- Training curves and confusion matrix plotting
- CSV training history
"""
from __future__ import annotations

import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    StrawDefectDataset,
    build_dataloaders,
    get_train_transform,
    get_val_transform,
)
from .losses import build_criterion
from .metrics import compute_metrics, plot_confusion_matrix, plot_roc_curves
from .models import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """All hyper-parameters and paths for a training run."""
    # Data
    data_dir: str = ""
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    image_size: int = 224
    # Model
    model_name: str = "resnet18"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    # Training
    epochs: int = 30
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-3
    encoder_lr: float = 1e-4
    classifier_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_accum: int = 1
    grad_clip: float = 1.0
    use_amp: bool = True
    loss_name: str = "ce"
    seed: int = 42
    device: str = "cuda"
    # Scheduling & early stopping
    lr_patience: int = 3
    lr_factor: float = 0.5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    # Output
    output_dir: str = "results"
    run_name: str = "exp_v1"


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info(device: torch.device) -> dict[str, Any]:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_mem / (1024**3), 2
        )
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
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
    """Run one training epoch. Returns metrics dict."""
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train [{cfg.model_name}]")
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
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{running_loss / step:.4f}")

    metrics = compute_metrics(all_labels, all_preds, None, CLASS_NAMES)
    metrics["loss"] = running_loss / len(loader)
    return metrics


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Run validation. Returns metrics dict (including y_true and y_prob for plotting)."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc="Val"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs), CLASS_NAMES)
    metrics["loss"] = running_loss / max(len(loader), 1)
    # Keep raw arrays for ROC plotting (not serialised to JSON)
    metrics["_y_true"] = all_labels
    metrics["_y_prob"] = np.array(all_probs)
    return metrics


def plot_training_curves(history: list[dict], save_dir: Path) -> None:
    """Save loss and accuracy learning curves."""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [h["train_acc"] for h in history], "o-", label="Train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], "s-", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # F1
    axes[2].plot(epochs, [h["train_f1"] for h in history], "o-", label="Train Weighted-F1")
    axes[2].plot(epochs, [h["val_f1"] for h in history], "s-", label="Val Weighted-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Weighted F1")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def run_training(cfg: TrainConfig) -> Path:
    """Execute the full training pipeline.

    Returns the run directory path.
    """
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # Directories
    run_dir = Path(cfg.output_dir) / cfg.model_name / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config & environment
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(device), indent=2), encoding="utf-8"
    )

    # Data
    loaders = build_dataloaders(
        data_dir=cfg.data_dir or None,
        train_csv=cfg.train_csv or None,
        val_csv=cfg.val_csv or None,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Model, loss, optimizer
    model = build_model(cfg.model_name, num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    model = model.to(device)
    criterion = build_criterion(cfg.loss_name).to(device)

    # Differential learning rates
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    # Training loop
    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        print(f"\n{'='*60}\nEpoch {epoch}/{cfg.epochs} — {cfg.model_name}\n{'='*60}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg)
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, cfg)
        # Schedule
        scheduler.step(val_metrics["weighted_f1"])

        epoch_seconds = time.perf_counter() - epoch_start

        # History entry
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "train_f1": train_metrics["weighted_f1"],
            "val_f1": val_metrics["weighted_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        plot_training_curves(history, run_dir / "curves")

        # Print summary
        print(
            f"  Train — loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}  "
            f"f1={train_metrics['weighted_f1']:.4f}"
        )
        print(
            f"  Val   — loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}  "
            f"f1={val_metrics['weighted_f1']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}"
        )

        # Save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "val_f1": val_metrics["weighted_f1"],
        }, ckpt_dir / "last.pt")

        # Save best
        current_f1 = val_metrics["weighted_f1"]
        if current_f1 > best_f1 + cfg.early_stopping_min_delta:
            best_f1 = current_f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "config": asdict(cfg),
                "val_f1": best_f1,
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }, ckpt_dir / "best.pt")
            print(f"  ★ New best model saved! (F1={best_f1:.4f})")

            # Save best-epoch plots
            plot_confusion_matrix(
                val_metrics["confusion_matrix"], CLASS_NAMES,
                run_dir / "best_confusion_matrix.png"
            )
            if val_metrics.get("_y_prob") is not None:
                plot_roc_curves(
                    val_metrics["_y_true"], val_metrics["_y_prob"],
                    CLASS_NAMES, run_dir / "best_roc_curves.png"
                )
        else:
            bad_epochs += 1

        # Early stopping
        if bad_epochs >= cfg.early_stopping_patience:
            print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {bad_epochs} epochs)")
            break

    total_seconds = time.perf_counter() - total_start
    summary = {
        "model_name": cfg.model_name,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "epochs_completed": len(history),
        "total_seconds": round(total_seconds, 1),
        "early_stopped": bad_epochs >= cfg.early_stopping_patience,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n✓ Training complete. Results in: {run_dir}")
    print(f"  Best epoch: {best_epoch}  |  Best val F1: {best_f1:.4f}  |  Time: {total_seconds:.0f}s")
    return run_dir
