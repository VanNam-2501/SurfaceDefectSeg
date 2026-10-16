"""Evaluation metrics and plotting utilities for classification."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix as sk_confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of events / successes (e.g. false negatives).
        n: Total trials (e.g. total defect samples).
        confidence: Confidence level (default: 0.95, giving z ~ 1.95996).

    Returns:
        (low, high) confidence interval bounds bounded in [0, 1].
    """
    if n <= 0:
        return 0.0, 1.0
    # Standard normal quantile for two-sided confidence interval
    # For 0.95 confidence: z = 1.959963984540054
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-4 else 1.959963984540054
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))
    low = max(0.0, float(center - margin))
    high = min(1.0, float(center + margin))
    return low, high


def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: list[str],
) -> dict:
    """Compute comprehensive classification metrics.

    Returns a dict with accuracy, weighted/macro F1, per-class precision/recall/f1,
    and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = sk_confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "per_class_precision": per_precision.tolist(),
        "per_class_recall": per_recall.tolist(),
        "per_class_f1": per_f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(
    cm: list | np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """Save a publication-quality confusion matrix heatmap using pure matplotlib."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Text annotations inside matrix cells
    thresh = cm_arr.max() / 2.0 if cm_arr.size > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=11)

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_classification_report(report_dict: dict, save_path: str | Path) -> None:
    """Save classification report as JSON."""
    import json
    Path(save_path).write_text(json.dumps(report_dict, indent=4), encoding="utf-8")


def plot_roc_curves(
    y_true: list | np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """Plot one-vs-rest ROC curves for each class."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_hierarchical_metrics(
    y_true_5class: list | np.ndarray,
    s1_preds: list | np.ndarray,
    s2_preds: list | np.ndarray,
    confidence_good_th: float = 0.80,
    confidence_defect_th: float = 0.65,
) -> dict:
    """Compute performance metrics for the Two-Stage Hierarchical pipeline.

    Args:
        y_true_5class: ground truth 0..4 (0: good, 1: bent, 2: loose, 3: missing, 4: pierced)
        s1_preds: binary predictions from Stage 1 (0: good, 1: defect)
        s2_preds: 4-class defect predictions from Stage 2 (0: bent, 1: loose, 2: missing, 3: pierced)
    """
    y_true = np.array(y_true_5class)
    s1_pred = np.array(s1_preds)
    s2_pred = np.array(s2_preds)

    # Ground truth binary: 0 = good, 1 = defect
    y_true_s1 = (y_true > 0).astype(int)

    # 1. Stage 1 Triage Metrics
    tp = int(np.sum((y_true_s1 == 1) & (s1_pred == 1)))
    fn = int(np.sum((y_true_s1 == 1) & (s1_pred == 0)))
    tn = int(np.sum((y_true_s1 == 0) & (s1_pred == 0)))
    fp = int(np.sum((y_true_s1 == 0) & (s1_pred == 1)))

    stage1_acc = float((tp + tn) / max(1, len(y_true_s1)))
    defect_recall = float(tp / max(1, tp + fn))  # True positive rate (defects caught)
    fnr = float(fn / max(1, tp + fn))  # Critical: defective products leaked to market
    fpr = float(fp / max(1, tn + fp))  # False alarms (good products flagged as defective)

    # 2. Stage 2 Defect Diagnosis Metrics (on defective samples)
    defect_mask = y_true > 0
    if np.sum(defect_mask) > 0:
        y_true_s2 = y_true[defect_mask] - 1  # 0: bent, 1: loose, 2: missing, 3: pierced
        s2_pred_defects = s2_pred[defect_mask]
        stage2_acc = float(accuracy_score(y_true_s2, s2_pred_defects))
        stage2_macro_f1 = float(f1_score(y_true_s2, s2_pred_defects, average="macro", zero_division=0))
    else:
        stage2_acc, stage2_macro_f1 = 0.0, 0.0

    # 3. End-to-End Hierarchical Prediction (5-class space)
    # If Stage 1 says good -> predict good (0)
    # If Stage 1 says defect -> predict (s2_pred + 1)
    end_to_end_preds = np.where(s1_pred == 0, 0, s2_pred + 1)
    end_to_end_acc = float(accuracy_score(y_true, end_to_end_preds))
    end_to_end_macro_f1 = float(f1_score(y_true, end_to_end_preds, average="macro", zero_division=0))

    hierarchical_cm = sk_confusion_matrix(y_true, end_to_end_preds, labels=range(5))

    return {
        "stage1_triage": {
            "accuracy": stage1_acc,
            "defect_recall": defect_recall,
            "false_negative_rate_fnr": fnr,
            "false_positive_rate_fpr": fpr,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        },
        "stage2_diagnosis": {
            "accuracy": stage2_acc,
            "macro_f1": stage2_macro_f1,
            "evaluated_samples": int(np.sum(defect_mask)),
        },
        "end_to_end": {
            "overall_accuracy": end_to_end_acc,
            "overall_macro_f1": end_to_end_macro_f1,
            "confusion_matrix": hierarchical_cm.tolist(),
        },
    }

