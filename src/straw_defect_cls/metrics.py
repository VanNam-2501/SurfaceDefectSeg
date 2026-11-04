"""Evaluation metrics and plotting utilities for classification."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts" / ".matplotlib"))
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

    labels = list(range(len(class_names)))
    per_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)

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
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])

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


def compute_audit_metrics(
    y_true: list[int] | np.ndarray,
    decisions: list[str] | np.ndarray,
    predicted_classes: list[int] | np.ndarray,
    cost_matrix: Optional[dict[str, float]] = None,
    pi_deploy: float = 0.05,
    pi_train: float = 0.80,
    latency_ms: float = 0.0,
    class_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Audit-grade performance metrics conforming to industrial KLTN thesis specs.

    Calculates:
    - FNR / FPR with Wilson 95% Confidence Interval.
    - Recall for every class (good, bent, loose, missing, pierced).
    - Deployment-reweighted metrics (prior shift adjustment to factory reality pi_deploy in 2-5%).
    - Review rate (% cartons routed to human operator inspection).
    - Expected cost per 1,000 cartons based on cost matrix (c_FN:c_REVIEW:c_FP = 100:10:1).
    - Latency in ms/carton.

    Args:
        y_true: Ground truth 5-class labels (0: Good, 1: Bent, 2: Loose, 3: Missing, 4: Pierced).
        decisions: Array of string decisions ('PASS', 'REVIEW', 'DEFECT').
        predicted_classes: Array of predicted 5-class labels (0..4, or -1 for review).
        cost_matrix: Dict with 'c_fn', 'c_review', 'c_fp' (default: 100:10:1).
        pi_deploy: Defect base rate in factory deployment (default: 0.05).
        pi_train: Defect base rate in dataset (default: 0.80).
        latency_ms: Inference latency per image in milliseconds.
        class_names: List of class names (default: Good, Bent, Loose, Missing, Pierced).

    Returns:
        Dict of comprehensive audit metrics.
    """
    if cost_matrix is None:
        cost_matrix = {"c_fn": 100.0, "c_review": 10.0, "c_fp": 1.0}
    c_fn = float(cost_matrix.get("c_fn", 100.0))
    c_rev = float(cost_matrix.get("c_review", 10.0))
    c_fp = float(cost_matrix.get("c_fp", 1.0))

    if class_names is None:
        class_names = ["Good", "Bent", "Loose", "Missing", "Pierced"]

    y_arr = np.asarray(y_true, dtype=int)
    dec_arr = np.asarray(decisions, dtype=str)
    pred_arr = np.asarray(predicted_classes, dtype=int)
    total_n = len(y_arr)

    is_defect = y_arr > 0
    is_good = y_arr == 0
    n_defects = int(np.sum(is_defect))
    n_good = int(np.sum(is_good))

    # 1. False Negative Rate (FNR): True defect marked PASS
    # (leaked defect into supply chain)
    fn_count = int(np.sum(is_defect & (dec_arr == "PASS")))
    fnr = float(fn_count / max(1, n_defects))
    fnr_ci_low, fnr_ci_high = wilson_score_interval(fn_count, n_defects, confidence=0.95)

    # 2. False Positive Rate (FPR): True good marked DEFECT
    # (false rejection of good carton)
    fp_count = int(np.sum(is_good & (dec_arr == "DEFECT")))
    fpr = float(fp_count / max(1, n_good))
    fpr_ci_low, fpr_ci_high = wilson_score_interval(fp_count, n_good, confidence=0.95)

    # 3. Review Rates
    rev_total = int(np.sum(dec_arr == "REVIEW"))
    rev_rate = float(rev_total / max(1, total_n))
    rev_good_rate = float(np.sum(is_good & (dec_arr == "REVIEW")) / max(1, n_good))
    rev_defect_rate = float(np.sum(is_defect & (dec_arr == "REVIEW")) / max(1, n_defects))

    # 4. Per-Class Empirical Recall
    recalls = {}
    for c_idx, c_name in enumerate(class_names):
        c_mask = y_arr == c_idx
        n_c = int(np.sum(c_mask))
        if n_c > 0:
            tp_c = int(np.sum(c_mask & (pred_arr == c_idx)))
            recalls[c_name.lower()] = round(float(tp_c / n_c), 4)
        else:
            recalls[c_name.lower()] = 0.0

    # Macro recall across defect classes (Bent, Loose, Missing, Pierced)
    defect_class_recalls = [recalls[name.lower()] for name in class_names[1:] if name.lower() in recalls]
    macro_recall_defect = float(np.mean(defect_class_recalls)) if defect_class_recalls else 0.0

    # 5. Reweighted Metrics Under Deployment Prior (pi_deploy in 2-5%)
    # In factory deployment: P(Good) = 1 - pi_deploy, P(Defect) = pi_deploy
    p_good_deploy = 1.0 - pi_deploy
    p_defect_deploy = pi_deploy

    p_fn_deploy = p_defect_deploy * fnr
    p_fp_deploy = p_good_deploy * fpr
    p_rev_deploy = (p_good_deploy * rev_good_rate) + (p_defect_deploy * rev_defect_rate)

    # Expected Cost per 1,000 cartons:
    expected_cost_1000 = 1000.0 * (c_fn * p_fn_deploy + c_rev * p_rev_deploy + c_fp * p_fp_deploy)

    # Reweighted deployment accuracy
    reweighted_acc = (p_good_deploy * recalls.get("good", 0.0)) + (p_defect_deploy * macro_recall_defect)

    return {
        "fnr": round(fnr, 4),
        "fnr_ci_low": round(fnr_ci_low, 4),
        "fnr_ci_high": round(fnr_ci_high, 4),
        "fn_count": fn_count,
        "n_defects": n_defects,
        "fpr": round(fpr, 4),
        "fpr_ci_low": round(fpr_ci_low, 4),
        "fpr_ci_high": round(fpr_ci_high, 4),
        "fp_count": fp_count,
        "n_good": n_good,
        "review_rate": round(rev_rate, 4),
        "review_rate_good": round(rev_good_rate, 4),
        "review_rate_defect": round(rev_defect_rate, 4),
        "recalls": recalls,
        "macro_recall_defect": round(macro_recall_defect, 4),
        "pi_deploy": pi_deploy,
        "reweighted_accuracy": round(reweighted_acc, 4),
        "expected_cost_per_1000": round(expected_cost_1000, 2),
        "latency_ms": round(latency_ms, 2),
    }

