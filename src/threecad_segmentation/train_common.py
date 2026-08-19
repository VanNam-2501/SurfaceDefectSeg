from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from losses import BCEDiceLoss
from training_state import recover_best_epoch
from training_data import build_train_dataset
from project_paths import ARTIFACTS_ROOT, DATASET_ROOT
from fullres_eval import (
    predict_full_image,
    quick_validate_fullres,
    read_binary_mask,
    read_rgb,
    resolve_path,
    row_group,
    row_id,
)


@dataclass
class TrainConfig:
    dataset_root: str = str(DATASET_ROOT)
    train_csv: str = str(DATASET_ROOT / "dataset_audit" / "splits" / "train.csv")
    val_csv: str = str(DATASET_ROOT / "dataset_audit" / "splits" / "val.csv")
    test_csv: str = str(DATASET_ROOT / "dataset_audit" / "splits" / "test.csv")
    output_dir: str = str(ARTIFACTS_ROOT / "training")
    run_name: str = "main_seed42"
    data_mode: str = "patch"  # patch for E2-E6, resize only for E1 baseline
    augmentation: str = "photometric"
    patch_size: int = 512
    stride: int = 256
    val_tile_batch_size: int = 4
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 2
    seed: int = 42
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_accum: int = 2
    grad_clip: float = 1.0
    amp: bool = True
    device: str = "cuda"
    train_threshold: float = 0.5
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    lr_patience: int = 3
    lr_factor: float = 0.5
    val_every: int = 1
    monitor_every: int = 1
    monitor_count: int = 4
    val_max_images: int = 0
    resume: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def positive_dice_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, int]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets = masks >= 0.5
    positive = targets.flatten(1).sum(1) > 0
    if not positive.any():
        return 0.0, 0
    p = preds[positive].flatten(1).float()
    t = targets[positive].flatten(1).float()
    inter = (p * t).sum(1)
    dice = (2.0 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
    return dice.sum().item(), int(positive.sum().item())


def split_encoder_decoder_params(model: nn.Module):
    if hasattr(model, "encoder_parameters") and hasattr(model, "decoder_parameters"):
        return list(model.encoder_parameters()), list(model.decoder_parameters())
    encoder_names = ("stem", "maxpool", "encoder1", "encoder2", "encoder3", "encoder4")
    encoder_params, decoder_params, encoder_ids = [], [], set()
    for name in encoder_names:
        module = getattr(model, name, None)
        if module is not None:
            for p in module.parameters():
                encoder_params.append(p)
                encoder_ids.add(id(p))
    for p in model.parameters():
        if id(p) not in encoder_ids:
            decoder_params.append(p)
    if not encoder_params:
        return list(model.parameters()), []
    return encoder_params, decoder_params


def build_train_loader(cfg: TrainConfig) -> DataLoader:
    ds = build_train_dataset(
        csv_path=cfg.train_csv,
        dataset_root=cfg.dataset_root,
        patch_size=cfg.patch_size,
        seed=cfg.seed,
        data_mode=cfg.data_mode,
        augmentation=cfg.augmentation,
    )
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=cfg.num_workers > 0,
    )


def unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["image"], batch["mask"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def environment_info(device: torch.device) -> dict[str, Any]:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": torch.cuda.get_device_capability(device),
                "gpu_total_memory_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3),
            }
        )
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import importlib.metadata as metadata
        info["mamba_ssm"] = metadata.version("mamba-ssm")
    except Exception:
        pass
    vmamba_repo = Path(__file__).resolve().parent / "third_party" / "VMamba"
    if (vmamba_repo / ".git").exists():
        try:
            info["vmamba_git_commit"] = subprocess.check_output(
                ["git", "-C", str(vmamba_repo), "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            pass
    return info


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    cfg: TrainConfig,
    model_name: str,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    bad_epochs: int,
    interrupted: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(cfg),
            "model_name": model_name,
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "bad_epochs": int(bad_epochs),
            "interrupted": bool(interrupted),
        },
        path,
    )


def plot_training_curves(history: pd.DataFrame, out_dir: Path) -> None:
    """Save publication-friendly training diagnostics after every epoch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if history.empty:
        return

    # 1) Train vs full-resolution Validation loss.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    if "val_loss" in history and history["val_loss"].notna().any():
        ax.plot(history["epoch"], history["val_loss"], marker="s", label="Val full-res loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("0.5 BCE + 0.5 positive-only Dice loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve_loss.png", dpi=170)
    # Backward-compatible filename used by the notebooks.
    fig.savefig(out_dir / "learning_curve.png", dpi=170)
    plt.close(fig)

    # 2) Dice learning curve.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_positive_dice_0.5"], marker="o", label="Train Positive Dice@0.5")
    if "val_positive_dice_0.5" in history and history["val_positive_dice_0.5"].notna().any():
        ax.plot(history["epoch"], history["val_positive_dice_0.5"], marker="s", label="Val full-res Positive Dice@0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve_dice.png", dpi=170)
    plt.close(fig)

    # 3) Loss components. Dice is positive-only by protocol.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_bce"], label="Train BCE (all samples)")
    ax.plot(history["epoch"], history["train_dice_loss"], label="Train Dice loss (positive only)")
    if "val_bce" in history and history["val_bce"].notna().any():
        ax.plot(history["epoch"], history["val_bce"], linestyle="--", label="Val BCE")
    if "val_positive_dice_loss" in history and history["val_positive_dice_loss"].notna().any():
        ax.plot(history["epoch"], history["val_positive_dice_loss"], linestyle="--", label="Val positive Dice loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_components.png", dpi=170)
    plt.close(fig)

    # 4) Learning rates.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["encoder_lr"], label="Encoder LR")
    ax.plot(history["epoch"], history["decoder_lr"], label="Decoder LR")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_rate.png", dpi=170)
    plt.close(fig)

    # 5) Time per epoch.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_seconds"], label="Train seconds")
    if "val_seconds" in history and history["val_seconds"].notna().any():
        ax.plot(history["epoch"], history["val_seconds"], label="Val seconds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_time.png", dpi=170)
    plt.close(fig)

    # 6) Peak train VRAM.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["peak_vram_gb"], label="Peak train VRAM (GB)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GB")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "vram.png", dpi=170)
    plt.close(fig)

def select_monitor_rows(val_csv: str | Path, count: int) -> list[int]:
    df = pd.read_csv(val_csv)
    selected: list[int] = []
    good = df.index[df["label"].astype(int) == 0].tolist()
    if good:
        selected.append(good[0])
    defect = df[df["label"].astype(int) == 1].copy()
    seen = set()
    for idx, row in defect.iterrows():
        g = row_group(row)
        if g not in seen:
            selected.append(int(idx))
            seen.add(g)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for idx in defect.index:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= count:
                break
    return selected[:count]


def save_monitor_examples(
    model: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    row_indices: list[int],
) -> None:
    df = pd.read_csv(cfg.val_csv)
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for idx in row_indices:
        row = df.loc[idx]
        label = int(row["label"])
        image = read_rgb(resolve_path(cfg.dataset_root, row["image_path"]))
        if label == 1:
            mask = read_binary_mask(resolve_path(cfg.dataset_root, row["mask_path"]))
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        prob, _ = predict_full_image(
            model,
            image,
            device,
            cfg.data_mode,
            cfg.patch_size,
            cfg.stride,
            cfg.val_tile_batch_size,
            cfg.amp,
        )
        pred = prob >= cfg.train_threshold
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[1].imshow(mask, vmin=0, vmax=1)
        axes[1].set_title("GT")
        axes[2].imshow(prob, vmin=0, vmax=1)
        axes[2].set_title("Probability")
        axes[3].imshow(image)
        axes[3].imshow(pred, alpha=0.45)
        axes[3].set_title(f"Prediction @ {cfg.train_threshold:.2f}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"epoch={epoch} | id={row_id(row)} | group={row_group(row)}")
        fig.tight_layout()
        fig.savefig(epoch_dir / f"{row_id(row)}.png", dpi=140)
        plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_inputs(cfg: TrainConfig, run_dir: Path) -> None:
    snap = run_dir / "split_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, path in (("train.csv", cfg.train_csv), ("val.csv", cfg.val_csv), ("test.csv", cfg.test_csv)):
        p = Path(path)
        if p.exists():
            shutil.copy2(p, snap / name)
            hashes[name] = _sha256(p)
    (snap / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def run_train(model: nn.Module, cfg: TrainConfig, model_name: str) -> Path:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = build_train_loader(cfg)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    encoder_params, decoder_params = split_encoder_decoder_params(model)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": cfg.encoder_lr, "name": "encoder"})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": cfg.decoder_lr, "name": "decoder"})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    run_dir = Path(cfg.output_dir) / model_name / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    curves_dir = run_dir / "curves"
    monitor_dir = run_dir / "monitor"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_info(device), indent=2), encoding="utf-8")
    snapshot_inputs(cfg, run_dir)

    total_params, trainable_params = count_parameters(model)
    model_info = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "effective_batch_size": cfg.batch_size * cfg.grad_accum,
    }
    (run_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    history_rows: list[dict[str, Any]] = []
    start_epoch = 1
    best_metric = -math.inf
    best_epoch = 0
    bad_epochs = 0

    if cfg.resume:
        resume_path = Path(cfg.resume)
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best_metric = float(state.get("best_metric", -math.inf))
        bad_epochs = int(state.get("bad_epochs", 0))
        # Interrupted checkpoints restart the interrupted epoch; epoch checkpoints continue after it.
        start_epoch = int(state.get("epoch", 0)) + (0 if state.get("interrupted") else 1)
        hist_path = run_dir / "training_history.csv"
        if hist_path.exists():
            history_rows = pd.read_csv(hist_path).to_dict("records")
        best_epoch = recover_best_epoch(state, history_rows)
        print(f"Resumed from {resume_path} -> epoch {start_epoch}")

    monitor_rows = select_monitor_rows(cfg.val_csv, cfg.monitor_count)
    total_wall_start = time.perf_counter()
    stopped_early = False
    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            current_epoch = epoch
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            running_loss = running_bce = 0.0
            running_dice_loss_weighted = 0.0
            running_dice_loss_n = 0
            dice_sum = 0.0
            dice_n = 0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{cfg.epochs}")
            for step, batch in enumerate(pbar, start=1):
                images, masks = unpack_batch(batch)
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(cfg.amp and device.type == "cuda"),
                ):
                    logits = model(images)
                    if logits.shape != masks.shape:
                        logits = torch.nn.functional.interpolate(
                            logits,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    total_loss, bce, dice_loss, positive_count = criterion.components(logits, masks)
                    loss_for_backward = total_loss / cfg.grad_accum

                scaler.scale(loss_for_backward).backward()
                should_step = (step % cfg.grad_accum == 0) or (step == len(train_loader))
                if should_step:
                    if cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                dsum, dn = positive_dice_from_logits(logits.detach(), masks, cfg.train_threshold)
                running_loss += float(total_loss.item())
                running_bce += float(bce.item())
                if positive_count > 0:
                    running_dice_loss_weighted += float(dice_loss.item()) * positive_count
                    running_dice_loss_n += positive_count
                dice_sum += dsum
                dice_n += dn
                n_batches += 1
                pbar.set_postfix(
                    loss=f"{running_loss/n_batches:.4f}",
                    pos_dice=f"{dice_sum/max(dice_n,1):.4f}",
                )

            train_seconds = time.perf_counter() - train_t0
            peak_vram = (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": running_loss / max(n_batches, 1),
                "train_bce": running_bce / max(n_batches, 1),
                "train_dice_loss": running_dice_loss_weighted / max(running_dice_loss_n, 1),
                "train_positive_dice_0.5": dice_sum / max(dice_n, 1),
                "train_seconds": train_seconds,
                "train_images_per_second": len(train_loader.dataset) / max(train_seconds, 1e-9),
                "peak_vram_gb": peak_vram,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "decoder_lr": optimizer.param_groups[-1]["lr"],
            }

            did_validate = epoch % cfg.val_every == 0
            if did_validate:
                val_metrics = quick_validate_fullres(
                    model,
                    cfg.val_csv,
                    cfg.dataset_root,
                    device,
                    threshold=cfg.train_threshold,
                    data_mode=cfg.data_mode,
                    tile_size=cfg.patch_size,
                    stride=cfg.stride,
                    tile_batch_size=cfg.val_tile_batch_size,
                    amp=cfg.amp,
                    max_images=cfg.val_max_images,
                )
                row.update(val_metrics)
                current_metric = float(val_metrics["val_positive_dice_0.5"])
                scheduler.step(current_metric)

                if current_metric > best_metric + cfg.early_stopping_min_delta:
                    best_metric = current_metric
                    best_epoch = epoch
                    bad_epochs = 0
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        cfg,
                        model_name,
                        epoch,
                        best_metric,
                        best_epoch,
                        bad_epochs,
                    )
                else:
                    bad_epochs += 1

                if cfg.monitor_every > 0 and epoch % cfg.monitor_every == 0:
                    save_monitor_examples(model, cfg, device, epoch, monitor_dir, monitor_rows)
            else:
                for key in (
                    "val_loss",
                    "val_bce",
                    "val_positive_dice_loss",
                    "val_positive_dice_0.5",
                    "val_positive_iou_0.5",
                    "val_image_recall_0.5",
                    "val_image_fnr_0.5",
                    "val_image_fpr_0.5",
                    "val_seconds",
                    "val_model_forward_seconds",
                    "val_tiles",
                ):
                    row[key] = float("nan")

            history_rows.append(row)
            hist_df = pd.DataFrame(history_rows)
            hist_df.to_csv(run_dir / "training_history.csv", index=False)
            plot_training_curves(hist_df, curves_dir)

            save_checkpoint(
                ckpt_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                model_name,
                epoch,
                best_metric,
                best_epoch,
                bad_epochs,
            )

            print(
                f"[{model_name}] epoch={epoch} "
                f"train_loss={row['train_loss']:.6f} "
                f"train_dice={row['train_positive_dice_0.5']:.4f} "
                f"val_dice={row.get('val_positive_dice_0.5', float('nan')):.4f} "
                f"VRAM={peak_vram:.2f}GB "
                f"train={train_seconds:.1f}s "
                f"val={row.get('val_seconds', float('nan')):.1f}s"
            )

            if did_validate and bad_epochs >= cfg.early_stopping_patience:
                stopped_early = True
                print(
                    f"Early stopping at epoch {epoch}: no Val Positive Dice improvement "
                    f"for {bad_epochs} validation epochs. Best epoch={best_epoch}."
                )
                break

    except KeyboardInterrupt:
        print("Training interrupted. Saving interrupt checkpoint; resume restarts this epoch.")
        save_checkpoint(
            ckpt_dir / "interrupt.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            model_name,
            current_epoch,
            best_metric,
            best_epoch,
            bad_epochs,
            interrupted=True,
        )
        raise

    total_wall_seconds = time.perf_counter() - total_wall_start
    if best_epoch == 0 and history_rows:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            # Preserve an existing selection when resuming an older checkpoint
            # that did not yet serialize best_epoch.
            best_state = torch.load(best_path, map_location="cpu")
            best_epoch = int(best_state.get("best_epoch", best_state.get("epoch", 0)))
            best_metric = float(best_state.get("best_metric", best_metric))
        else:
            # Safety fallback only when validation produced no checkpoint.
            best_epoch = int(history_rows[-1]["epoch"])
            shutil.copy2(ckpt_dir / "last.pt", best_path)

    summary = {
        **model_info,
        "best_epoch": int(best_epoch),
        "best_val_positive_dice_0.5": float(best_metric),
        "stopped_early": bool(stopped_early),
        "epochs_completed": int(history_rows[-1]["epoch"]) if history_rows else 0,
        "total_train_run_wall_seconds": float(total_wall_seconds),
        "total_train_only_seconds": float(sum(float(r["train_seconds"]) for r in history_rows)),
        "total_validation_seconds": float(
            sum(float(r.get("val_seconds", 0.0)) for r in history_rows if not pd.isna(r.get("val_seconds", 0.0)))
        ),
        "max_peak_vram_gb": float(max((float(r["peak_vram_gb"]) for r in history_rows), default=0.0)),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def common_parser(
    default_batch_size: int = 2,
    default_grad_accum: int = 2,
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=str(DATASET_ROOT))
    p.add_argument("--train-csv", default=str(DATASET_ROOT / "dataset_audit" / "splits" / "train.csv"))
    p.add_argument("--val-csv", default=str(DATASET_ROOT / "dataset_audit" / "splits" / "val.csv"))
    p.add_argument("--test-csv", default=str(DATASET_ROOT / "dataset_audit" / "splits" / "test.csv"))
    p.add_argument("--output-dir", default="results")
    p.add_argument("--run-name", default="main_seed42")
    p.add_argument("--data-mode", choices=["patch", "resize"], default="patch")
    p.add_argument(
        "--augmentation",
        choices=["none", "photometric", "geometric", "photometric_geometric"],
        default="photometric",
        help="Final default is mild photometric. Geometric transforms are opt-in only.",
    )
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--val-tile-batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=default_batch_size)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--decoder-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=default_grad_accum)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--train-threshold", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--monitor-every", type=int, default=1)
    p.add_argument("--monitor-count", type=int, default=4)
    p.add_argument("--val-max-images", type=int, default=0)
    p.add_argument("--resume", default="")
    return p


def config_from_args(args) -> TrainConfig:
    return TrainConfig(
        dataset_root=args.dataset_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        run_name=args.run_name,
        data_mode=args.data_mode,
        augmentation=args.augmentation,
        patch_size=args.patch_size,
        stride=args.stride,
        val_tile_batch_size=args.val_tile_batch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        device=args.device,
        train_threshold=args.train_threshold,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        val_every=args.val_every,
        monitor_every=args.monitor_every,
        monitor_count=args.monitor_count,
        val_max_images=args.val_max_images,
        resume=args.resume,
    )
