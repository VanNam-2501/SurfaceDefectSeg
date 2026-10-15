"""Evaluation metrics and plotting utilities for classification."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Save a publication-quality confusion matrix heatmap."""
    cm_arr = np.array(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


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
