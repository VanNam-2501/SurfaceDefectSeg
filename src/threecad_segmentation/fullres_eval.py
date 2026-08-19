"""Full-resolution inference, validation threshold selection, and test metrics.

Main protocol for patch-trained models:
full image -> 512x512 tiles -> stride 256 -> average overlap -> original HxW.
Validation/test never use GT to choose crops.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ani_dataset import IMAGENET_MEAN, IMAGENET_STD


MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def resolve_path(dataset_root: str | Path, value: object) -> Path:
    p = Path(_text(value))
    return p.resolve() if p.is_absolute() else (Path(dataset_root) / p).resolve()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 2:
        return (arr != 0).astype(np.uint8)
    if arr.ndim == 3:
        return np.any(arr != 0, axis=2).astype(np.uint8)
    raise ValueError(f"Unsupported mask shape {arr.shape} at {path}")


def row_group(row: pd.Series) -> str:
    candidates = (
        "defect_group",
        "defect_type",
        "group",
        "class",
        "category",
        "folder",
    )
    value = ""
    for key in candidates:
        if key in row.index and _text(row[key]):
            value = _text(row[key])
            break
    if not value:
        value = Path(_text(row.get("image_path", ""))).parent.name
    low = value.lower().strip()
    mapping = {
        "scratche": "scratches",
        "scratch": "scratches",
        "multiple-defects": "Multiple-defects",
        "multiple_defects": "Multiple-defects",
        "good": "Good",
    }
    return mapping.get(low, value if value else "unknown")


def row_id(row: pd.Series) -> str:
    for key in ("image_id", "id", "uid"):
        if key in row.index and _text(row[key]):
            return _text(row[key])
    return Path(_text(row.get("image_path", "image"))).stem


def normalize_tile(tile: np.ndarray) -> torch.Tensor:
    x = tile.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_to_minimum(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    dh = max(0, tile_size - h)
    dw = max(0, tile_size - w)
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    if dh or dw:
        image = np.pad(
            image,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return image, (top, bottom, left, right)


def predict_sliding_window(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    stride: int = 256,
    tile_batch_size: int = 4,
    amp: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    t_e2e = time.perf_counter()
    padded, pads = pad_to_minimum(image, tile_size)
    h, w = padded.shape[:2]
    ys = tile_starts(h, tile_size, stride)
    xs = tile_starts(w, tile_size, stride)
    coords = [(y, x) for y in ys for x in xs]

    prob_sum = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    model_seconds = 0.0

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coords), tile_batch_size):
            batch_coords = coords[start : start + tile_batch_size]
            tiles = [
                normalize_tile(padded[y : y + tile_size, x : x + tile_size])
                for y, x in batch_coords
            ]
            batch = torch.stack(tiles, dim=0).to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                logits = model(batch)
                if logits.shape[-2:] != (tile_size, tile_size):
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=(tile_size, tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                probs = torch.sigmoid(logits)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_seconds += time.perf_counter() - t0
            probs_np = probs[:, 0].float().cpu().numpy()

            for (y, x), p in zip(batch_coords, probs_np):
                prob_sum[y : y + tile_size, x : x + tile_size] += p
                counts[y : y + tile_size, x : x + tile_size] += 1.0

    prob = prob_sum / np.maximum(counts, 1.0)
    top, bottom, left, right = pads
    y1 = top
    y2 = h - bottom if bottom else h
    x1 = left
    x2 = w - right if right else w
    prob = prob[y1:y2, x1:x2]
    e2e_seconds = time.perf_counter() - t_e2e
    return prob, {
        "tiles": float(len(coords)),
        "model_seconds": float(model_seconds),
        "e2e_seconds": float(e2e_seconds),
    }


def resize_pad_preprocess(image: np.ndarray, size: int = 512):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.Resampling.BILINEAR))
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, (top, left, nh, nw, h, w)


def predict_resize_pad(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    size: int = 512,
    amp: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    t_e2e = time.perf_counter()
    canvas, meta = resize_pad_preprocess(image, size=size)
    batch = normalize_tile(canvas).unsqueeze(0).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, enabled=(amp and device.type == "cuda")
    ):
        logits = model(batch)
        if logits.shape[-2:] != (size, size):
            logits = torch.nn.functional.interpolate(
                logits, size=(size, size), mode="bilinear", align_corners=False
            )
        prob_canvas = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_seconds = time.perf_counter() - t0

    top, left, nh, nw, h, w = meta
    crop = prob_canvas[top : top + nh, left : left + nw]
    prob = np.asarray(
        Image.fromarray(crop.astype(np.float32), mode="F").resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return prob, {
        "tiles": 1.0,
        "model_seconds": float(model_seconds),
        "e2e_seconds": float(time.perf_counter() - t_e2e),
    }


def predict_full_image(
    model,
    image,
    device,
    data_mode: str,
    tile_size: int,
    stride: int,
    tile_batch_size: int,
    amp: bool,
):
    if data_mode == "patch":
        return predict_sliding_window(
            model,
            image,
            device,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=tile_batch_size,
            amp=amp,
        )
    if data_mode == "resize":
        return predict_resize_pad(model, image, device, size=tile_size, amp=amp)
    raise ValueError(f"Unsupported data_mode={data_mode!r}")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def warmup_model_inference(
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 512,
    tile_batch_size: int = 4,
    amp: bool = True,
    iterations: int = 5,
) -> None:
    """Warm CUDA/model kernels before any reported latency measurement."""
    if iterations <= 0:
        return
    batch_n = max(1, int(tile_batch_size))
    dummy = torch.zeros(batch_n, 3, tile_size, tile_size, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(iterations):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(amp and device.type == "cuda"),
            ):
                out = model(dummy)
                if out.shape[-2:] != (tile_size, tile_size):
                    out = torch.nn.functional.interpolate(
                        out,
                        size=(tile_size, tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                _ = torch.sigmoid(out)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fullres_loss_from_probability(
    prob: np.ndarray,
    mask: np.ndarray,
    is_positive: bool,
    eps: float = 1e-6,
) -> tuple[float, float | None]:
    """Return BCE(all) and positive-only soft Dice loss for one stitched image."""
    p = np.clip(prob.astype(np.float64, copy=False), eps, 1.0 - eps)
    y = mask.astype(np.float64, copy=False)
    bce = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())
    if not is_positive:
        return bce, None
    inter = float((p * y).sum())
    denom = float(p.sum() + y.sum())
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return bce, float(1.0 - dice)


def quick_validate_fullres(
    model: torch.nn.Module,
    csv_path: str | Path,
    dataset_root: str | Path,
    device: torch.device,
    threshold: float = 0.5,
    data_mode: str = "patch",
    tile_size: int = 512,
    stride: int = 256,
    tile_batch_size: int = 4,
    amp: bool = True,
    max_images: int = 0,
) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    if max_images and max_images > 0:
        df = df.iloc[:max_images].copy()

    dice_values: list[float] = []
    iou_values: list[float] = []
    bce_values: list[float] = []
    positive_dice_losses: list[float] = []
    tp_img = fp_img = tn_img = fn_img = 0
    e2e_total = model_total = tiles_total = 0.0

    for _, row in df.iterrows():
        label = int(row["label"])
        image = read_rgb(resolve_path(dataset_root, row["image_path"]))
        if label == 1:
            mask = read_binary_mask(resolve_path(dataset_root, row["mask_path"]))
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        prob, timing = predict_full_image(
            model, image, device, data_mode, tile_size, stride, tile_batch_size, amp
        )
        pred = prob >= threshold
        image_pred = bool(prob.max() >= threshold)
        val_bce, val_pos_dice_loss = fullres_loss_from_probability(
            prob, mask, is_positive=(label == 1)
        )
        bce_values.append(val_bce)
        if val_pos_dice_loss is not None:
            positive_dice_losses.append(val_pos_dice_loss)

        if label == 1:
            tp_img += int(image_pred)
            fn_img += int(not image_pred)
            gt = mask.astype(bool)
            inter = np.logical_and(pred, gt).sum()
            p_sum = pred.sum()
            g_sum = gt.sum()
            dice_values.append(_safe_div(2 * inter, p_sum + g_sum))
            iou_values.append(_safe_div(inter, np.logical_or(pred, gt).sum()))
        else:
            fp_img += int(image_pred)
            tn_img += int(not image_pred)

        e2e_total += timing["e2e_seconds"]
        model_total += timing["model_seconds"]
        tiles_total += timing["tiles"]

    mean_bce = float(np.mean(bce_values)) if bce_values else 0.0
    mean_pos_dice_loss = float(np.mean(positive_dice_losses)) if positive_dice_losses else 0.0
    return {
        "val_loss": 0.5 * mean_bce + 0.5 * mean_pos_dice_loss,
        "val_bce": mean_bce,
        "val_positive_dice_loss": mean_pos_dice_loss,
        "val_positive_dice_0.5": float(np.mean(dice_values)) if dice_values else 0.0,
        "val_positive_iou_0.5": float(np.mean(iou_values)) if iou_values else 0.0,
        "val_image_recall_0.5": _safe_div(tp_img, tp_img + fn_img),
        "val_image_fnr_0.5": _safe_div(fn_img, tp_img + fn_img),
        "val_image_fpr_0.5": _safe_div(fp_img, fp_img + tn_img),
        "val_seconds": float(e2e_total),
        "val_model_forward_seconds": float(model_total),
        "val_tiles": float(tiles_total),
    }


def histogram_counts(prob: np.ndarray, mask: np.ndarray, bins: int):
    idx = np.minimum((prob * bins).astype(np.int32), bins - 1)
    pos = np.bincount(idx[mask.astype(bool)].ravel(), minlength=bins).astype(np.int64)
    neg = np.bincount(idx[~mask.astype(bool)].ravel(), minlength=bins).astype(np.int64)
    return pos, neg


def counts_at_threshold(pos_hist: np.ndarray, neg_hist: np.ndarray, threshold: float):
    bins = len(pos_hist)
    idx = int(math.ceil(float(threshold) * bins))
    idx = min(max(idx, 0), bins - 1)
    tp = int(pos_hist[idx:].sum())
    fp = int(neg_hist[idx:].sum())
    fn = int(pos_hist.sum() - tp)
    tn = int(neg_hist.sum() - fp)
    return tp, fp, fn, tn


def pixel_average_precision_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    total_pos = int(pos_hist.sum())
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(pos_hist[::-1])
    fp = np.cumsum(neg_hist[::-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def threshold_scan_from_records(records: list[dict[str, Any]], thresholds: np.ndarray):
    rows = []
    positives = [r for r in records if r["label"] == 1]
    goods = [r for r in records if r["label"] == 0]
    for t in thresholds:
        dice_vals = []
        tp_img = fn_img = fp_img = tn_img = 0
        for r in positives:
            tp, fp, fn, _ = counts_at_threshold(r["pos_hist"], r["neg_hist"], float(t))
            dice_vals.append(_safe_div(2 * tp, 2 * tp + fp + fn))
            detected = r["image_score"] >= t
            tp_img += int(detected)
            fn_img += int(not detected)
        for r in goods:
            detected = r["image_score"] >= t
            fp_img += int(detected)
            tn_img += int(not detected)
        rows.append(
            {
                "threshold": float(t),
                "positive_dice": float(np.mean(dice_vals)) if dice_vals else 0.0,
                "image_fnr": _safe_div(fn_img, tp_img + fn_img),
                "image_fpr": _safe_div(fp_img, fp_img + tn_img),
                "image_recall": _safe_div(tp_img, tp_img + fn_img),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(scan: pd.DataFrame, fnr_limit: float = 0.10) -> tuple[float, str]:
    valid = scan[scan["image_fnr"] <= fnr_limit + 1e-12]
    if len(valid):
        best = valid.sort_values(
            ["positive_dice", "image_fpr", "threshold"],
            ascending=[False, True, True],
        ).iloc[0]
        return float(best["threshold"]), "fnr_constraint_satisfied"

    # Fallback is explicit and reproducible if the requested FNR is unattainable.
    best = scan.sort_values(
        ["image_fnr", "positive_dice", "image_fpr", "threshold"],
        ascending=[True, False, True, True],
    ).iloc[0]
    return float(best["threshold"]), "fnr_constraint_unattainable_fallback"


def derive_train_bins(
    train_csv: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    df = pd.read_csv(train_csv)
    areas: list[int] = []
    counts: list[int] = []
    for _, row in df.iterrows():
        if int(row["label"]) != 1:
            continue
        mask = read_binary_mask(resolve_path(dataset_root, row["mask_path"]))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        comp_count = max(0, n - 1)
        counts.append(comp_count)
        if n > 1:
            areas.extend([int(v) for v in stats[1:, cv2.CC_STAT_AREA]])

    if not areas:
        q25 = q50 = q75 = 0.0
    else:
        q25, q50, q75 = [float(x) for x in np.quantile(np.asarray(areas), [0.25, 0.50, 0.75])]

    q75_count = float(np.quantile(np.asarray(counts), 0.75)) if counts else 1.0
    many_threshold = max(3, int(math.ceil(q75_count)))
    return {
        "component_area_q25": q25,
        "component_area_q50": q50,
        "component_area_q75": q75,
        "multi_region_many_threshold": many_threshold,
        "train_component_count": len(areas),
        "train_defect_image_count": len(counts),
    }


def size_label(area: int, bins: dict[str, Any]) -> str:
    if area <= bins["component_area_q25"]:
        return "Tiny"
    if area <= bins["component_area_q50"]:
        return "Small"
    if area <= bins["component_area_q75"]:
        return "Medium"
    return "Large"


def multi_region_label(count: int, bins: dict[str, Any]) -> str:
    if count <= 1:
        return "single"
    many = int(bins["multi_region_many_threshold"])
    if count < many:
        return "few"
    return "many"


def _binary_image_metrics(labels: np.ndarray, preds: np.ndarray):
    labels = labels.astype(bool)
    preds = preds.astype(bool)
    tp = int(np.logical_and(labels, preds).sum())
    fp = int(np.logical_and(~labels, preds).sum())
    tn = int(np.logical_and(~labels, ~preds).sum())
    fn = int(np.logical_and(labels, ~preds).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "image_tp": tp,
        "image_fp": fp,
        "image_tn": tn,
        "image_fn": fn,
        "image_precision": precision,
        "image_recall": recall,
        "image_specificity": specificity,
        "image_f1": _safe_div(2 * precision * recall, precision + recall),
        "image_fnr": _safe_div(fn, tp + fn),
        "image_fpr": _safe_div(fp, fp + tn),
    }


def _plot_threshold_scan(scan: pd.DataFrame, selected: float, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scan["threshold"], scan["positive_dice"], label="Positive Dice")
    ax.plot(scan["threshold"], scan["image_fnr"], label="Image FNR")
    ax.plot(scan["threshold"], scan["image_fpr"], label="Image FPR")
    ax.axvline(selected, linestyle="--", label=f"Selected={selected:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_image_level_diagnostic_figures(
    labels: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
    output_dir: Path,
    threshold: float,
) -> None:
    """Save ROC, PR and confusion-matrix figures plus curve CSVs."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve.
    if len(np.unique(labels)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
        roc_df.to_csv(fig_dir / "image_roc_curve.csv", index=False)
        auc = float(roc_auc_score(labels, scores))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUROC={auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "image_roc_curve.png", dpi=180)
        plt.close(fig)

        # Precision-recall curve.
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        threshold_col = np.concatenate([pr_thresholds, [np.nan]])
        pd.DataFrame(
            {"precision": precision, "recall": recall, "threshold": threshold_col}
        ).to_csv(fig_dir / "image_pr_curve.csv", index=False)
        ap = float(average_precision_score(labels, scores))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, label=f"AUPRC={ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "image_pr_curve.png", dpi=180)
        plt.close(fig)

    # Confusion matrix at the frozen operating threshold.
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    pd.DataFrame(cm, index=["GT Good", "GT Defect"], columns=["Pred Good", "Pred Defect"]).to_csv(
        fig_dir / "confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Good", "Defect"])
    ax.set_yticks([0, 1], labels=["Good", "Defect"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Image-level confusion matrix @ threshold={threshold:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def evaluate_split(
    model: torch.nn.Module,
    csv_path: str | Path,
    dataset_root: str | Path,
    train_csv: str | Path,
    output_dir: str | Path,
    device: torch.device,
    data_mode: str = "patch",
    threshold: float | None = None,
    select_threshold: bool = False,
    fnr_limit: float = 0.10,
    tile_size: int = 512,
    stride: int = 256,
    tile_batch_size: int = 4,
    amp: bool = True,
    hist_bins: int = 4096,
    save_qualitative: int = 6,
    warmup_batches: int = 5,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    derived_bins = derive_train_bins(train_csv, dataset_root)
    (output_dir / "train_derived_bins.json").write_text(
        json.dumps(derived_bins, indent=2), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    global_pos = np.zeros(hist_bins, dtype=np.int64)
    global_neg = np.zeros(hist_bins, dtype=np.int64)
    total_e2e = total_model = total_tiles = 0.0
    e2e_times: list[float] = []
    peak_vram = 0.0

    # Timing protocol: warm-up is explicitly excluded from all reported latency.
    warmup_model_inference(
        model,
        device,
        tile_size=tile_size,
        tile_batch_size=tile_batch_size,
        amp=amp,
        iterations=warmup_batches,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _, row in df.iterrows():
        label = int(row["label"])
        iid = row_id(row)
        group = "Good" if label == 0 else row_group(row)
        image_path = resolve_path(dataset_root, row["image_path"])
        image = read_rgb(image_path)
        if label == 1:
            mask = read_binary_mask(resolve_path(dataset_root, row["mask_path"]))
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        prob, timing = predict_full_image(
            model,
            image,
            device,
            data_mode,
            tile_size,
            stride,
            tile_batch_size,
            amp,
        )
        pos_hist, neg_hist = histogram_counts(prob, mask, hist_bins)
        global_pos += pos_hist
        global_neg += neg_hist

        n, comp_map, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        component_count = max(0, n - 1)
        image_component_areas: list[int] = []
        for cid in range(1, n):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            image_component_areas.append(area)
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            w = int(stats[cid, cv2.CC_STAT_WIDTH])
            h = int(stats[cid, cv2.CC_STAT_HEIGHT])
            comp_prob = prob[comp_map == cid]
            regions.append(
                {
                    "image_id": iid,
                    "group": group,
                    "component_id": cid,
                    "area": area,
                    "bbox_width": w,
                    "bbox_height": h,
                    "bbox_x": x,
                    "bbox_y": y,
                    "max_probability": float(comp_prob.max()) if comp_prob.size else 0.0,
                    "size_bin": size_label(area, derived_bins),
                }
            )

        records.append(
            {
                "image_id": iid,
                "image_path": str(image_path),
                "mask_path": str(resolve_path(dataset_root, row["mask_path"])) if label == 1 else "",
                "label": label,
                "group": group,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "gt_pixels": int(mask.sum()),
                "num_components": component_count,
                "multi_region_bin": multi_region_label(component_count, derived_bins),
                "smallest_component_area": min(image_component_areas) if image_component_areas else 0,
                "smallest_component_size_bin": size_label(min(image_component_areas), derived_bins) if image_component_areas else "Good",
                "image_score": float(prob.max()),
                "pos_hist": pos_hist,
                "neg_hist": neg_hist,
                "tiles": int(timing["tiles"]),
                "model_seconds": float(timing["model_seconds"]),
                "e2e_seconds": float(timing["e2e_seconds"]),
            }
        )
        total_e2e += timing["e2e_seconds"]
        total_model += timing["model_seconds"]
        total_tiles += timing["tiles"]
        e2e_times.append(timing["e2e_seconds"])

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

    if select_threshold:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
        scan = threshold_scan_from_records(records, thresholds)
        threshold, selection_status = choose_threshold(scan, fnr_limit=fnr_limit)
        scan.to_csv(output_dir / "threshold_scan.csv", index=False)
        _plot_threshold_scan(scan, threshold, output_dir / "threshold_selection.png")
        (output_dir / "selected_threshold.json").write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "fnr_limit": fnr_limit,
                    "selection_status": selection_status,
                    "rule": "Val only: FNR<=limit, maximize Positive Dice, tie lower FPR",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif threshold is None:
        raise ValueError("threshold must be supplied when select_threshold=False")
    else:
        threshold = float(threshold)
        selection_status = "fixed_external_threshold"

    assert threshold is not None
    image_rows = []
    labels = []
    scores = []
    preds = []
    positive_dices = []
    positive_ious = []
    global_tp = global_fp = global_fn = global_tn = 0

    for r in records:
        tp, fp, fn, tn = counts_at_threshold(r["pos_hist"], r["neg_hist"], threshold)
        image_pred = int(r["image_score"] >= threshold)
        labels.append(r["label"])
        scores.append(r["image_score"])
        preds.append(image_pred)
        dice = _safe_div(2 * tp, 2 * tp + fp + fn) if r["label"] == 1 else float("nan")
        iou = _safe_div(tp, tp + fp + fn) if r["label"] == 1 else float("nan")
        if r["label"] == 1:
            positive_dices.append(dice)
            positive_ious.append(iou)
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn
        image_rows.append(
            {
                k: v
                for k, v in r.items()
                if k not in {"pos_hist", "neg_hist"}
            }
            | {
                "threshold": threshold,
                "image_pred": image_pred,
                "pixel_tp": tp,
                "pixel_fp": fp,
                "pixel_fn": fn,
                "pixel_tn": tn,
                "positive_dice": dice,
                "positive_iou": iou,
                "predicted_positive_pixels": tp + fp,
            }
        )

    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    preds_np = np.asarray(preds, dtype=np.int64)
    image_metrics = _binary_image_metrics(labels_np, preds_np)
    try:
        image_auroc = float(roc_auc_score(labels_np, scores_np))
    except ValueError:
        image_auroc = float("nan")
    try:
        image_auprc = float(average_precision_score(labels_np, scores_np))
    except ValueError:
        image_auprc = float("nan")

    for reg in regions:
        reg["threshold"] = threshold
        reg["detected"] = int(reg["max_probability"] >= threshold)

    region_recall = float(np.mean([r["detected"] for r in regions])) if regions else float("nan")
    pixel_ap = pixel_average_precision_from_hist(global_pos, global_neg)

    main_metrics = {
        "threshold": threshold,
        "threshold_status": selection_status,
        "image_auroc": image_auroc,
        "image_auprc": image_auprc,
        **image_metrics,
        "positive_dice": float(np.mean(positive_dices)) if positive_dices else float("nan"),
        "positive_iou": float(np.mean(positive_ious)) if positive_ious else float("nan"),
        "pixel_auprc_hist": pixel_ap,
        "pixel_recall": _safe_div(global_tp, global_tp + global_fn),
        "pixel_precision": _safe_div(global_tp, global_tp + global_fp),
        "region_recall_any_overlap": region_recall,
        "n_images": len(records),
        "n_regions": len(regions),
        "evaluation_total_seconds": float(total_e2e),
        "model_forward_total_seconds": float(total_model),
        "tiles_total": int(total_tiles),
        "tiles_per_second": _safe_div(total_tiles, total_model),
        "full_image_latency_mean_seconds": float(np.mean(e2e_times)) if e2e_times else 0.0,
        "full_image_latency_p50_seconds": float(np.quantile(e2e_times, 0.50)) if e2e_times else 0.0,
        "full_image_latency_p95_seconds": float(np.quantile(e2e_times, 0.95)) if e2e_times else 0.0,
        "eval_peak_vram_gb": float(peak_vram),
        "timing_warmup_batches": int(warmup_batches),
        "timing_note": "warm-up forward passes excluded from reported latency/throughput",
        "pixel_auprc_method": f"global histogram approximation with {hist_bins} bins",
    }

    save_image_level_diagnostic_figures(
        labels_np, scores_np, preds_np, output_dir, threshold
    )

    image_df = pd.DataFrame(image_rows)
    region_df = pd.DataFrame(regions)
    image_df.to_csv(output_dir / "per_image_metrics.csv", index=False)
    region_df.to_csv(output_dir / "per_region_metrics.csv", index=False)
    (output_dir / "main_metrics.json").write_text(json.dumps(main_metrics, indent=2), encoding="utf-8")
    pd.DataFrame([main_metrics]).to_csv(output_dir / "main_metrics.csv", index=False)

    # E4 defect-group analysis.
    group_rows = []
    for group, g in image_df[image_df["label"] == 1].groupby("group"):
        group_rows.append(
            {
                "group": group,
                "n_images": len(g),
                "image_recall": float(g["image_pred"].mean()),
                "positive_dice": float(g["positive_dice"].mean()),
                "positive_iou": float(g["positive_iou"].mean()),
                "small_sample_warning": bool(len(g) < 10),
            }
        )
    pd.DataFrame(group_rows).to_csv(output_dir / "defect_group_metrics.csv", index=False)

    # E3 size analysis. Region recall is component-wise; Dice/image recall are
    # stratified by the smallest GT component in each defect image so multi-defect
    # images are assigned reproducibly to one size bin.
    size_rows = []
    defect_images = image_df[image_df["label"] == 1]
    if len(region_df):
        for label_name in ["Tiny", "Small", "Medium", "Large"]:
            rg = region_df[region_df["size_bin"] == label_name]
            ig = defect_images[defect_images["smallest_component_size_bin"] == label_name]
            size_rows.append(
                {
                    "size_bin": label_name,
                    "n_regions": len(rg),
                    "region_recall": float(rg["detected"].mean()) if len(rg) else float("nan"),
                    "n_images_by_smallest_component": len(ig),
                    "image_recall": float(ig["image_pred"].mean()) if len(ig) else float("nan"),
                    "positive_dice": float(ig["positive_dice"].mean()) if len(ig) else float("nan"),
                    "positive_iou": float(ig["positive_iou"].mean()) if len(ig) else float("nan"),
                }
            )
    pd.DataFrame(size_rows).to_csv(output_dir / "defect_size_metrics.csv", index=False)

    # E5 multi-region analysis, including component-wise region recall.
    multi_rows = []
    for label_name in ["single", "few", "many"]:
        g = defect_images[defect_images["multi_region_bin"] == label_name]
        ids = set(g["image_id"].astype(str))
        rg = region_df[region_df["image_id"].astype(str).isin(ids)] if len(region_df) else region_df
        multi_rows.append(
            {
                "multi_region_bin": label_name,
                "n_images": len(g),
                "n_regions": len(rg),
                "image_recall": float(g["image_pred"].mean()) if len(g) else float("nan"),
                "positive_dice": float(g["positive_dice"].mean()) if len(g) else float("nan"),
                "positive_iou": float(g["positive_iou"].mean()) if len(g) else float("nan"),
                "region_recall": float(rg["detected"].mean()) if len(rg) else float("nan"),
            }
        )
    pd.DataFrame(multi_rows).to_csv(output_dir / "multi_region_metrics.csv", index=False)

    if save_qualitative > 0:
        save_qualitative_examples(
            model,
            image_df,
            dataset_root,
            output_dir / "qualitative",
            device,
            threshold,
            data_mode,
            tile_size,
            stride,
            tile_batch_size,
            amp,
            top_k=save_qualitative,
        )

    return main_metrics


def save_qualitative_examples(
    model,
    image_df: pd.DataFrame,
    dataset_root,
    output_dir: Path,
    device,
    threshold,
    data_mode,
    tile_size,
    stride,
    tile_batch_size,
    amp,
    top_k: int = 6,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    good = image_df[image_df["label"] == 0].sort_values("image_score", ascending=False).head(top_k)
    bad_defect = image_df[image_df["label"] == 1].sort_values("positive_dice", ascending=True).head(top_k)
    candidates.extend([(r, "high_fp") for _, r in good.iterrows()])
    candidates.extend([(r, "worst_defect") for _, r in bad_defect.iterrows()])

    for rank, (r, tag) in enumerate(candidates, start=1):
        image_path = Path(r["image_path"])
        image = read_rgb(image_path)
        if int(r["label"]) == 1 and str(r.get("mask_path", "")):
            gt = read_binary_mask(Path(r["mask_path"]))
        else:
            gt = np.zeros(image.shape[:2], dtype=np.uint8)
        prob, _ = predict_full_image(
            model, image, device, data_mode, tile_size, stride, tile_batch_size, amp
        )
        pred = prob >= threshold
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[1].imshow(gt, vmin=0, vmax=1)
        axes[1].set_title("GT")
        axes[2].imshow(prob, vmin=0, vmax=1)
        axes[2].set_title("Probability")
        axes[3].imshow(image)
        axes[3].imshow(pred, alpha=0.45)
        axes[3].set_title(f"Binary @ {threshold:.2f}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{tag} | {r['image_id']}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{tag}_{rank:02d}_{r['image_id']}.png", dpi=150)
        plt.close(fig)
