"""Learn an automatic image-level verifier from cached segmentation probabilities.

This is deliberately separate from model training.  U-Net and SegFormer first
produce full-resolution probability maps.  The verifier then learns which
probability/component patterns are real defects and which are common false
alarms.  Validation is used through out-of-fold predictions; Test is evaluated
once with frozen thresholds.

Three fair branches are produced:

* unet: U-Net features only
* segformer: SegFormer features only
* fusion: both models plus spatial-agreement features

Each branch reports both a fully automatic binary decision and a three-level
PASS/REVIEW/DEFECT safety policy.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from calibrate_decision_policy import prediction_specs, probability_path, read_probability
from decision_policy import border_connected_dark_roi
from fullres_eval import read_rgb, resolve_path, row_group, row_id


FEATURE_SCHEMA_VERSION = 1
PROBABILITY_THRESHOLDS = (0.30, 0.50, 0.70, 0.90, 0.95)
TOP_K = (1, 4, 16, 64, 256, 1024)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def log_count(value: int | float) -> float:
    return float(math.log1p(max(0.0, float(value))))


def top_mean(values: np.ndarray, count: int) -> float:
    if values.size == 0:
        return 0.0
    count = min(int(count), int(values.size))
    if count >= values.size:
        return float(values.mean())
    selected = np.partition(values, values.size - count)[-count:]
    return float(selected.mean())


def component_summary(
    probability: np.ndarray,
    gray: np.ndarray,
    gradient: np.ndarray,
    roi: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    binary = ((probability >= threshold) & roi).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    prefix = f"t{int(round(threshold * 100)):02d}"
    if count <= 1:
        return {
            f"{prefix}_component_count_log": 0.0,
            f"{prefix}_total_area_log": 0.0,
            f"{prefix}_largest_area_log": 0.0,
            f"{prefix}_largest_peak": 0.0,
            f"{prefix}_largest_mean": 0.0,
            f"{prefix}_largest_aspect_log": 0.0,
            f"{prefix}_largest_extent": 0.0,
            f"{prefix}_largest_x": 0.5,
            f"{prefix}_largest_y": 0.5,
            f"{prefix}_largest_border_distance": 0.5,
            f"{prefix}_largest_gray": 0.0,
            f"{prefix}_largest_gray_contrast": 0.0,
            f"{prefix}_largest_gradient": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_id = int(np.argmax(areas)) + 1
    component = labels == largest_id
    area = int(stats[largest_id, cv2.CC_STAT_AREA])
    width = int(stats[largest_id, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_id, cv2.CC_STAT_HEIGHT])
    x_norm = float(centroids[largest_id, 0] / max(1, probability.shape[1] - 1))
    y_norm = float(centroids[largest_id, 1] / max(1, probability.shape[0] - 1))
    border_distance = min(x_norm, 1.0 - x_norm, y_norm, 1.0 - y_norm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component & roi
    component_gray = float(gray[component].mean()) if component.any() else 0.0
    ring_gray = float(gray[ring].mean()) if ring.any() else component_gray
    return {
        f"{prefix}_component_count_log": log_count(count - 1),
        f"{prefix}_total_area_log": log_count(int(binary.sum())),
        f"{prefix}_largest_area_log": log_count(area),
        f"{prefix}_largest_peak": float(probability[component].max()),
        f"{prefix}_largest_mean": float(probability[component].mean()),
        f"{prefix}_largest_aspect_log": float(math.log((width + 1.0) / (height + 1.0))),
        f"{prefix}_largest_extent": safe_div(area, width * height),
        f"{prefix}_largest_x": x_norm,
        f"{prefix}_largest_y": y_norm,
        f"{prefix}_largest_border_distance": border_distance,
        f"{prefix}_largest_gray": component_gray,
        f"{prefix}_largest_gray_contrast": component_gray - ring_gray,
        f"{prefix}_largest_gradient": float(gradient[component].mean()),
    }


def probability_features(
    name: str,
    probability: np.ndarray,
    gray: np.ndarray,
    gradient: np.ndarray,
    roi: np.ndarray,
) -> dict[str, float]:
    prefix = f"{name}__"
    values = probability[roi]
    features: dict[str, float] = {
        f"{prefix}mean": float(values.mean()) if values.size else 0.0,
        f"{prefix}std": float(values.std()) if values.size else 0.0,
        f"{prefix}max": float(values.max()) if values.size else 0.0,
    }
    for quantile in (0.90, 0.95, 0.99, 0.995, 0.999):
        key = str(quantile).replace("0.", "q")
        features[f"{prefix}{key}"] = (
            float(np.quantile(values, quantile)) if values.size else 0.0
        )
    for count in TOP_K:
        features[f"{prefix}top_{count}_mean"] = top_mean(values, count)

    weights = np.square(probability, dtype=np.float32) * roi
    weight_sum = float(weights.sum())
    features[f"{prefix}weighted_gray"] = safe_div(float((weights * gray).sum()), weight_sum)
    features[f"{prefix}weighted_gradient"] = safe_div(
        float((weights * gradient).sum()), weight_sum
    )
    height, width = roi.shape
    edge = np.zeros_like(roi)
    edge_y = max(1, int(round(height * 0.05)))
    edge_x = max(1, int(round(width * 0.05)))
    edge[:edge_y] = True
    edge[-edge_y:] = True
    edge[:, :edge_x] = True
    edge[:, -edge_x:] = True
    for threshold in PROBABILITY_THRESHOLDS:
        binary = (probability >= threshold) & roi
        code = int(round(threshold * 100))
        features[f"{prefix}t{code:02d}_pixels_log"] = log_count(int(binary.sum()))
        features[f"{prefix}t{code:02d}_edge_fraction"] = safe_div(
            int(np.count_nonzero(binary & edge)), int(binary.sum())
        )
        for key, value in component_summary(
            probability, gray, gradient, roi, threshold
        ).items():
            features[f"{prefix}{key}"] = value
    return features


def agreement_features(
    first_name: str,
    first: np.ndarray,
    second_name: str,
    second: np.ndarray,
    gray: np.ndarray,
    gradient: np.ndarray,
    roi: np.ndarray,
) -> dict[str, float]:
    prefix = f"pair_{first_name}_{second_name}__"
    features: dict[str, float] = {}
    valid_first = first[roi]
    valid_second = second[roi]
    features[f"{prefix}mean_abs_difference"] = (
        float(np.mean(np.abs(valid_first - valid_second))) if valid_first.size else 0.0
    )
    if valid_first.size and float(valid_first.std()) > 1e-8 and float(valid_second.std()) > 1e-8:
        features[f"{prefix}correlation"] = float(np.corrcoef(valid_first, valid_second)[0, 1])
    else:
        features[f"{prefix}correlation"] = 0.0

    minimum = np.minimum(first, second)
    average = (first + second) * 0.5
    maximum = np.maximum(first, second)
    for aggregate_name, aggregate in (
        ("minimum", minimum),
        ("average", average),
        ("maximum", maximum),
    ):
        values = aggregate[roi]
        features[f"{prefix}{aggregate_name}_max"] = float(values.max()) if values.size else 0.0
        for count in (16, 64, 256):
            features[f"{prefix}{aggregate_name}_top_{count}"] = top_mean(values, count)

    for threshold in PROBABILITY_THRESHOLDS:
        first_mask = (first >= threshold) & roi
        second_mask = (second >= threshold) & roi
        intersection = int(np.count_nonzero(first_mask & second_mask))
        union = int(np.count_nonzero(first_mask | second_mask))
        code = int(round(threshold * 100))
        features[f"{prefix}t{code:02d}_intersection_log"] = log_count(intersection)
        features[f"{prefix}t{code:02d}_union_log"] = log_count(union)
        features[f"{prefix}t{code:02d}_iou"] = safe_div(intersection, union)

    # Consensus morphology is especially useful for rejecting isolated highlights.
    features.update(
        {
            f"{prefix}{key}": value
            for key, value in component_summary(
                minimum, gray, gradient, roi, threshold=0.30
            ).items()
        }
    )
    return features


def image_features(image: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    values = gray[roi]
    gradient_values = gradient[roi]
    features = {
        "img__roi_fraction": float(roi.mean()),
        "img__gray_mean": float(values.mean()) if values.size else 0.0,
        "img__gray_std": float(values.std()) if values.size else 0.0,
        "img__gradient_mean": float(gradient_values.mean()) if values.size else 0.0,
        "img__gradient_q95": (
            float(np.quantile(gradient_values, 0.95)) if values.size else 0.0
        ),
    }
    return gray, gradient, features


def extract_split_features(
    records: pd.DataFrame,
    split: str,
    dataset_root: Path,
    predictions: dict[str, Path],
    roi_threshold: int,
    feature_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_names = list(predictions)
    for position, (_, row) in enumerate(records.iterrows(), start=1):
        image_id = row_id(row)
        image = read_rgb(resolve_path(dataset_root, row["image_path"]))
        original_height, original_width = image.shape[:2]
        scale = min(1.0, float(feature_size) / max(original_height, original_width))
        target_width = max(1, int(round(original_width * scale)))
        target_height = max(1, int(round(original_height * scale)))
        if (target_height, target_width) != (original_height, original_width):
            image = cv2.resize(
                image, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
        roi = border_connected_dark_roi(image, threshold=roi_threshold)
        gray, gradient, features = image_features(image, roi)
        probabilities: dict[str, np.ndarray] = {}
        for model, root in predictions.items():
            probability = read_probability(probability_path(root, split, image_id))
            if probability.shape != (original_height, original_width):
                raise ValueError(f"Shape mismatch for {model}/{image_id}")
            if probability.shape != roi.shape:
                probability = cv2.resize(
                    probability,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            probabilities[model] = probability
            features.update(
                probability_features(model, probability, gray, gradient, roi)
            )
        for first_index, first_name in enumerate(model_names):
            for second_name in model_names[first_index + 1 :]:
                features.update(
                    agreement_features(
                        first_name,
                        probabilities[first_name],
                        second_name,
                        probabilities[second_name],
                        gray,
                        gradient,
                        roi,
                    )
                )
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "label": int(row["label"]),
                "label_name": "Defect" if int(row["label"]) else "Good",
                "defect_group": row_group(row),
            }
            | features
        )
        if position % 25 == 0 or position == len(records):
            print(f"[features:{split}] {position}/{len(records)}", flush=True)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, branch: str) -> list[str]:
    common = [column for column in frame if column.startswith("img__")]
    if branch == "fusion":
        specific = [
            column
            for column in frame
            if "__" in column and not column.startswith("img__")
        ]
    else:
        specific = [column for column in frame if column.startswith(f"{branch}__")]
    result = common + specific
    if not specific:
        raise ValueError(f"No features found for branch {branch!r}")
    return result


def build_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=18,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def choose_binary_threshold(
    probabilities: np.ndarray, labels: np.ndarray, max_fnr: float
) -> tuple[float, dict[str, float]]:
    candidates = np.unique(
        np.concatenate(([0.0, 1.0], probabilities, np.linspace(0.0, 1.0, 1001)))
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = probabilities >= threshold
        fnr = float(np.mean(~predicted[labels == 1]))
        if fnr > max_fnr + 1e-12:
            continue
        fpr = float(np.mean(predicted[labels == 0]))
        candidate = (fpr, -float(threshold), fnr)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        threshold = 0.0
    else:
        threshold = -best[1]
    predicted = probabilities >= threshold
    return float(threshold), binary_metrics(labels, predicted)


def choose_triage_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    max_alert_fnr: float,
    max_defect_fpr: float,
) -> tuple[float, float, dict[str, float]]:
    candidates = np.unique(
        np.concatenate(([0.0, 1.0], probabilities, np.linspace(0.0, 1.0, 1001)))
    )
    pass_candidates = [
        float(threshold)
        for threshold in candidates
        if float(np.mean(probabilities[labels == 1] < threshold))
        <= max_alert_fnr + 1e-12
    ]
    pass_threshold = max(pass_candidates) if pass_candidates else 0.0
    defect_candidates = [
        float(threshold)
        for threshold in candidates
        if float(np.mean(probabilities[labels == 0] >= threshold))
        <= max_defect_fpr + 1e-12
    ]
    defect_threshold = min(defect_candidates) if defect_candidates else 1.0
    if pass_threshold > defect_threshold:
        # Both safety targets can be met with a fully automatic threshold.
        pass_threshold = defect_threshold = pass_threshold
    decisions = triage_decisions(probabilities, pass_threshold, defect_threshold)
    return pass_threshold, defect_threshold, triage_metrics(labels, decisions)


def choose_hybrid_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    trusted_defect: np.ndarray,
    max_fnr: float,
) -> tuple[float, dict[str, float]]:
    """Keep spatial consensus as DEFECT and learn only how to rescue its misses."""
    candidates = np.unique(
        np.concatenate(([0.0, 1.0], probabilities, np.linspace(0.0, 1.0, 1001)))
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = trusted_defect | (probabilities >= threshold)
        metrics = binary_metrics(labels, predicted)
        if metrics["fnr"] > max_fnr + 1e-12:
            continue
        candidate = (metrics["fpr"], -float(threshold), metrics["fnr"])
        if best is None or candidate < best:
            best = candidate
    threshold = -best[1] if best is not None else 0.0
    predicted = trusted_defect | (probabilities >= threshold)
    return float(threshold), binary_metrics(labels, predicted)


def select_branch_score_strategy(
    branch: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    learned_validation_score: np.ndarray,
    learned_test_score: np.ndarray,
    labels: np.ndarray,
    max_fnr: float,
) -> tuple[str, np.ndarray, np.ndarray, float, dict[str, float]]:
    """Select the simplest Validation-Pareto score without consulting Test labels."""
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "learned_hgb": (learned_validation_score, learned_test_score)
    }
    if branch != "fusion":
        prefix = f"{branch}__"
        for column in validation.columns:
            if not column.startswith(prefix):
                continue
            remainder = column[len(prefix) :]
            if remainder == "max" or remainder.startswith("q") or (
                remainder.startswith("top_") and remainder.endswith("_mean")
            ):
                candidates[column] = (
                    validation[column].to_numpy(dtype=np.float64),
                    test[column].to_numpy(dtype=np.float64),
                )

    selected: tuple[float, float, str, np.ndarray, np.ndarray, float, dict[str, float]] | None = None
    for name, (validation_score, test_score) in candidates.items():
        threshold, metrics = choose_binary_threshold(validation_score, labels, max_fnr)
        try:
            auc = float(roc_auc_score(labels, validation_score))
        except ValueError:
            auc = 0.0
        key = (
            float(metrics["fpr"]),
            float(metrics["fnr"]),
            -auc,
            name,
            validation_score,
            test_score,
            threshold,
            metrics,
        )
        if selected is None or key[:3] < selected[:3]:
            selected = key
    assert selected is not None
    return (
        selected[3],
        selected[4],
        selected[5],
        float(selected[6]),
        selected[7],
    )


def choose_two_specialist_hybrid(
    first_score: np.ndarray,
    second_score: np.ndarray,
    labels: np.ndarray,
    trusted_defect: np.ndarray,
    max_fnr: float,
) -> tuple[float, float, dict[str, float]]:
    """Tune independent rescue thresholds for two complementary specialists."""
    first_candidates = np.unique(
        np.concatenate(
            (np.quantile(first_score, np.linspace(0.0, 1.0, 301)), [first_score.max() + 1e-6])
        )
    )
    second_candidates = np.unique(
        np.concatenate(
            (
                np.quantile(second_score, np.linspace(0.0, 1.0, 301)),
                [second_score.max() + 1e-6],
            )
        )
    )
    best: tuple[float, float, float, float] | None = None
    for first_threshold in first_candidates:
        first_positive = first_score >= first_threshold
        for second_threshold in second_candidates:
            predicted = trusted_defect | first_positive | (second_score >= second_threshold)
            metrics = binary_metrics(labels, predicted)
            if metrics["fnr"] > max_fnr + 1e-12:
                continue
            candidate = (
                float(metrics["fpr"]),
                -float(first_threshold),
                -float(second_threshold),
                float(metrics["fnr"]),
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        first_threshold = second_threshold = 0.0
    else:
        first_threshold = -best[1]
        second_threshold = -best[2]
    predicted = (
        trusted_defect
        | (first_score >= first_threshold)
        | (second_score >= second_threshold)
    )
    return first_threshold, second_threshold, binary_metrics(labels, predicted)


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    positive = labels == 1
    good = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fn = int(np.count_nonzero(~predicted & positive))
    fp = int(np.count_nonzero(predicted & good))
    tn = int(np.count_nonzero(~predicted & good))
    return {
        "positive_images": int(positive.sum()),
        "good_images": int(good.sum()),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "fnr": safe_div(fn, tp + fn),
        "fpr": safe_div(fp, fp + tn),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, fp + tn),
        "accuracy": safe_div(tp + tn, len(labels)),
    }


def triage_decisions(
    probabilities: np.ndarray, pass_threshold: float, defect_threshold: float
) -> np.ndarray:
    decisions = np.full(probabilities.shape, "review", dtype=object)
    decisions[probabilities < pass_threshold] = "pass"
    decisions[probabilities >= defect_threshold] = "defect"
    return decisions


def triage_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    positive = labels == 1
    good = ~positive
    defect = decisions == "defect"
    review = decisions == "review"
    passed = decisions == "pass"
    return {
        "positive_images": int(positive.sum()),
        "good_images": int(good.sum()),
        "alert_fnr": float(np.mean(passed[positive])),
        "auto_defect_recall": float(np.mean(defect[positive])),
        "positive_review_rate": float(np.mean(review[positive])),
        "auto_defect_fpr": float(np.mean(defect[good])),
        "good_review_rate": float(np.mean(review[good])),
        "good_attention_rate": float(np.mean((defect | review)[good])),
        "overall_review_rate": float(np.mean(review)),
        "automatic_coverage": float(np.mean(~review)),
        "pass_count": int(np.count_nonzero(passed)),
        "review_count": int(np.count_nonzero(review)),
        "defect_count": int(np.count_nonzero(defect)),
    }


@dataclass
class BranchResult:
    branch: str
    feature_columns: list[str]
    folds: list[HistGradientBoostingClassifier]
    oof_probability: np.ndarray
    test_probability: np.ndarray
    binary_threshold: float
    pass_threshold: float
    defect_threshold: float


def train_branch(
    branch: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    folds: int,
    seed: int,
    max_fnr: float,
    max_defect_fpr: float,
) -> BranchResult:
    columns = feature_columns(validation, branch)
    x_validation = validation[columns].to_numpy(dtype=np.float32)
    x_test = test[columns].to_numpy(dtype=np.float32)
    labels = validation["label"].to_numpy(dtype=np.int64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(validation), dtype=np.float64)
    test_predictions: list[np.ndarray] = []
    estimators: list[HistGradientBoostingClassifier] = []
    for fold_index, (train_indices, holdout_indices) in enumerate(
        splitter.split(x_validation, labels), start=1
    ):
        estimator = build_classifier(seed + fold_index)
        estimator.fit(x_validation[train_indices], labels[train_indices])
        oof[holdout_indices] = estimator.predict_proba(x_validation[holdout_indices])[:, 1]
        test_predictions.append(estimator.predict_proba(x_test)[:, 1])
        estimators.append(estimator)
        print(f"[verifier:{branch}] fold {fold_index}/{folds}", flush=True)
    test_probability = np.mean(np.stack(test_predictions, axis=0), axis=0)
    binary_threshold, _ = choose_binary_threshold(oof, labels, max_fnr)
    pass_threshold, defect_threshold, _ = choose_triage_thresholds(
        oof, labels, max_fnr, max_defect_fpr
    )
    return BranchResult(
        branch=branch,
        feature_columns=columns,
        folds=estimators,
        oof_probability=oof,
        test_probability=test_probability,
        binary_threshold=binary_threshold,
        pass_threshold=pass_threshold,
        defect_threshold=defect_threshold,
    )


def group_rows(
    branch: str,
    frame: pd.DataFrame,
    binary_predicted: np.ndarray,
    decisions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, indices in frame.groupby("defect_group", dropna=False).groups.items():
        positions = frame.index.get_indexer(indices)
        labels = frame.loc[indices, "label"].to_numpy(dtype=np.int64)
        positive = labels == 1
        if not positive.any():
            continue
        branch_binary = binary_predicted[positions]
        branch_decisions = decisions[positions]
        rows.append(
            {
                "branch": branch,
                "defect_group": group,
                "positive_images": int(positive.sum()),
                "automatic_recall": float(np.mean(branch_binary[positive])),
                "triage_alert_recall": float(
                    np.mean(branch_decisions[positive] != "pass")
                ),
                "triage_auto_defect_recall": float(
                    np.mean(branch_decisions[positive] == "defect")
                ),
                "triage_positive_review_rate": float(
                    np.mean(branch_decisions[positive] == "review")
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--prediction", action="append", default=[], help="model=prediction_root"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-csv", default="")
    parser.add_argument("--test-csv", default="")
    parser.add_argument(
        "--base-val-decisions",
        default="",
        help="Optional frozen spatial-policy per_image_decisions.csv for hybrid fusion",
    )
    parser.add_argument(
        "--base-test-decisions",
        default="",
        help="Matching frozen spatial-policy Test decisions",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fnr", type=float, default=0.02)
    parser.add_argument("--max-defect-fpr", type=float, default=0.10)
    parser.add_argument(
        "--fnr-safety-margin",
        type=float,
        default=0.005,
        help="Extra Validation margin used by hybrid fusion to absorb split drift",
    )
    parser.add_argument("--border-dark-threshold", type=int, default=5)
    parser.add_argument(
        "--feature-size",
        type=int,
        default=256,
        help="Longest side used for verifier features; segmentation maps remain unchanged",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def load_or_extract(
    path: Path,
    records: pd.DataFrame,
    split: str,
    dataset_root: Path,
    predictions: dict[str, Path],
    roi_threshold: int,
    feature_size: int,
    rebuild: bool,
) -> pd.DataFrame:
    if path.is_file() and not rebuild:
        frame = pd.read_csv(path)
        expected = set(predictions)
        available = {
            column.split("__", 1)[0]
            for column in frame.columns
            if "__" in column and not column.startswith("img__")
        }
        if expected.issubset(available) and len(frame) == len(records):
            print(f"[features:{split}] using cache {path}")
            return frame
    frame = extract_split_features(
        records, split, dataset_root, predictions, roi_threshold, feature_size
    )
    frame.to_csv(path, index=False)
    return frame


def main() -> None:
    args = parse_args()
    if not 2 <= args.folds <= 10:
        raise ValueError("--folds must be between 2 and 10")
    if not 0.0 <= args.max_fnr < 1.0:
        raise ValueError("--max-fnr must be in [0, 1)")
    if not 0.0 <= args.max_defect_fpr < 1.0:
        raise ValueError("--max-defect-fpr must be in [0, 1)")
    if args.feature_size < 64:
        raise ValueError("--feature-size must be at least 64")
    predictions = prediction_specs(args.prediction)
    if len(predictions) < 2:
        raise ValueError("Supply U-Net and SegFormer prediction caches")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = dataset_root / "dataset_audit" / "splits"
    val_csv = Path(args.val_csv).resolve() if args.val_csv else split_root / "val.csv"
    test_csv = Path(args.test_csv).resolve() if args.test_csv else split_root / "test.csv"
    val_records = pd.read_csv(val_csv).fillna("")
    test_records = pd.read_csv(test_csv).fillna("")

    validation = load_or_extract(
        output_dir / "val_features.csv",
        val_records,
        "val",
        dataset_root,
        predictions,
        args.border_dark_threshold,
        args.feature_size,
        args.rebuild_features,
    )
    test = load_or_extract(
        output_dir / "test_features.csv",
        test_records,
        "test",
        dataset_root,
        predictions,
        args.border_dark_threshold,
        args.feature_size,
        args.rebuild_features,
    )
    validation.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    val_labels = validation["label"].to_numpy(dtype=np.int64)
    test_labels = test["label"].to_numpy(dtype=np.int64)

    branch_names = list(predictions) + ["fusion"]
    selected_branch_scores: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    summary_rows: list[dict[str, Any]] = []
    group_metrics: list[dict[str, Any]] = []
    per_image = test[["image_id", "split", "label", "label_name", "defect_group"]].copy()
    per_validation = validation[
        ["image_id", "split", "label", "label_name", "defect_group"]
    ].copy()
    base_validation: pd.DataFrame | None = None
    base_test: pd.DataFrame | None = None
    if args.base_val_decisions or args.base_test_decisions:
        if not args.base_val_decisions or not args.base_test_decisions:
            raise ValueError("Supply both --base-val-decisions and --base-test-decisions")
        base_validation = pd.read_csv(Path(args.base_val_decisions).resolve())
        base_test = pd.read_csv(Path(args.base_test_decisions).resolve())
        base_validation = validation[["image_id"]].merge(
            base_validation[["image_id", "decision"]], on="image_id", how="left"
        )
        base_test = test[["image_id"]].merge(
            base_test[["image_id", "decision"]], on="image_id", how="left"
        )
        if base_validation["decision"].isna().any() or base_test["decision"].isna().any():
            raise ValueError("Base spatial decisions do not cover Validation and Test")
    policy: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "type": "cross_validated_learned_verifier",
        "targets": {
            "max_automatic_fnr": float(args.max_fnr),
            "max_triage_auto_defect_fpr": float(args.max_defect_fpr),
        },
        "calibration": {
            "split": "val",
            "csv": str(val_csv),
            "images": int(len(validation)),
            "folds": int(args.folds),
            "seed": int(args.seed),
            "feature_size": int(args.feature_size),
        },
        "prediction_sources": {name: str(root) for name, root in predictions.items()},
        "branches": {},
    }

    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for branch in branch_names:
        result = train_branch(
            branch,
            validation,
            test,
            args.folds,
            args.seed,
            args.max_fnr,
            args.max_defect_fpr,
        )
        (
            score_strategy,
            validation_score,
            test_score,
            binary_threshold,
            oof_binary_metrics,
        ) = select_branch_score_strategy(
            branch,
            validation,
            test,
            result.oof_probability,
            result.test_probability,
            val_labels,
            args.max_fnr,
        )
        selected_branch_scores[branch] = (
            validation_score,
            test_score,
            score_strategy,
        )
        oof_binary = validation_score >= binary_threshold
        test_binary = test_score >= binary_threshold
        test_binary_metrics = binary_metrics(test_labels, test_binary)
        pass_threshold, defect_threshold, _ = choose_triage_thresholds(
            validation_score,
            val_labels,
            args.max_fnr,
            args.max_defect_fpr,
        )
        oof_triage = triage_decisions(validation_score, pass_threshold, defect_threshold)
        test_triage = triage_decisions(test_score, pass_threshold, defect_threshold)
        oof_triage_metrics = triage_metrics(val_labels, oof_triage)
        test_triage_metrics = triage_metrics(test_labels, test_triage)
        try:
            oof_auc = float(roc_auc_score(val_labels, validation_score))
            test_auc = float(roc_auc_score(test_labels, test_score))
        except ValueError:
            oof_auc = test_auc = float("nan")

        summary_rows.append(
            {
                "branch": branch,
                "mode": "fully_automatic",
                "score_strategy": score_strategy,
                "threshold_low": binary_threshold,
                "threshold_high": binary_threshold,
                "val_fnr": oof_binary_metrics["fnr"],
                "val_fpr": oof_binary_metrics["fpr"],
                "test_fnr": test_binary_metrics["fnr"],
                "test_fpr": test_binary_metrics["fpr"],
                "test_recall": test_binary_metrics["recall"],
                "test_specificity": test_binary_metrics["specificity"],
                "test_accuracy": test_binary_metrics["accuracy"],
                "test_review_rate": 0.0,
                "test_good_attention_rate": test_binary_metrics["fpr"],
                "val_auc": oof_auc,
                "test_auc": test_auc,
            }
        )
        summary_rows.append(
            {
                "branch": branch,
                "mode": "safety_triage",
                "score_strategy": score_strategy,
                "threshold_low": pass_threshold,
                "threshold_high": defect_threshold,
                "val_fnr": oof_triage_metrics["alert_fnr"],
                "val_fpr": oof_triage_metrics["auto_defect_fpr"],
                "test_fnr": test_triage_metrics["alert_fnr"],
                "test_fpr": test_triage_metrics["auto_defect_fpr"],
                "test_recall": test_triage_metrics["auto_defect_recall"],
                "test_specificity": 1.0 - test_triage_metrics["auto_defect_fpr"],
                "test_accuracy": float("nan"),
                "test_review_rate": test_triage_metrics["overall_review_rate"],
                "test_good_review_rate": test_triage_metrics["good_review_rate"],
                "test_positive_review_rate": test_triage_metrics["positive_review_rate"],
                "test_good_attention_rate": test_triage_metrics["good_attention_rate"],
                "test_automatic_coverage": test_triage_metrics["automatic_coverage"],
                "val_auc": oof_auc,
                "test_auc": test_auc,
            }
        )
        per_image[f"{branch}_score"] = test_score
        per_image[f"{branch}_automatic"] = np.where(test_binary, "defect", "pass")
        per_image[f"{branch}_triage"] = test_triage
        per_validation[f"{branch}_oof_score"] = validation_score
        per_validation[f"{branch}_automatic"] = np.where(
            oof_binary, "defect", "pass"
        )
        per_validation[f"{branch}_triage"] = oof_triage
        group_metrics.extend(
            group_rows(branch, test, test_binary, test_triage)
        )
        policy["branches"][branch] = {
            "features": result.feature_columns,
            "score_strategy": score_strategy,
            "fully_automatic_threshold": binary_threshold,
            "triage_pass_threshold": pass_threshold,
            "triage_defect_threshold": defect_threshold,
            "validation_oof_auc": oof_auc,
            "test_metrics_for_report_only": {
                "fully_automatic": test_binary_metrics,
                "safety_triage": test_triage_metrics,
                "auc": test_auc,
            },
        }
        with (model_dir / f"{branch}_folds.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "branch": branch,
                    "feature_columns": result.feature_columns,
                    "estimators": result.folds,
                },
                handle,
            )

        # The automatic hybrid is deliberately a *pair* policy: a frozen
        # spatial-consensus decision is rescued by two independent specialist
        # scores.  A three-model run still produces the useful standalone and
        # learned-fusion branches; it simply has no pair-specific hybrid row.
        if (
            branch == "fusion"
            and base_validation is not None
            and base_test is not None
            and len(predictions) == 2
        ):
            val_trusted = base_validation["decision"].to_numpy() == "defect"
            test_trusted = base_test["decision"].to_numpy() == "defect"
            specialist_names = list(predictions)
            first_name, second_name = specialist_names
            first_validation, first_test, first_strategy = selected_branch_scores[first_name]
            second_validation, second_test, second_strategy = selected_branch_scores[second_name]
            hybrid_fnr_limit = max(0.0, args.max_fnr - args.fnr_safety_margin)
            (
                first_threshold,
                second_threshold,
                hybrid_val_metrics,
            ) = choose_two_specialist_hybrid(
                first_validation,
                second_validation,
                val_labels,
                val_trusted,
                hybrid_fnr_limit,
            )
            hybrid_test_predicted = (
                test_trusted
                | (first_test >= first_threshold)
                | (second_test >= second_threshold)
            )
            hybrid_test_metrics = binary_metrics(test_labels, hybrid_test_predicted)
            summary_rows.append(
                {
                    "branch": "hybrid_fusion",
                    "mode": "fully_automatic",
                    "score_strategy": "spatial_consensus_plus_two_specialist_rescue",
                    "threshold_low": first_threshold,
                    "threshold_high": second_threshold,
                    "val_fnr": hybrid_val_metrics["fnr"],
                    "val_fpr": hybrid_val_metrics["fpr"],
                    "test_fnr": hybrid_test_metrics["fnr"],
                    "test_fpr": hybrid_test_metrics["fpr"],
                    "test_recall": hybrid_test_metrics["recall"],
                    "test_specificity": hybrid_test_metrics["specificity"],
                    "test_accuracy": hybrid_test_metrics["accuracy"],
                    "test_review_rate": 0.0,
                    "test_good_attention_rate": hybrid_test_metrics["fpr"],
                    "val_auc": oof_auc,
                    "test_auc": test_auc,
                }
            )
            per_image["hybrid_fusion_automatic"] = np.where(
                hybrid_test_predicted, "defect", "pass"
            )
            per_validation["hybrid_fusion_automatic"] = np.where(
                val_trusted
                | (first_validation >= first_threshold)
                | (second_validation >= second_threshold),
                "defect",
                "pass",
            )
            group_metrics.extend(
                group_rows(
                    "hybrid_fusion",
                    test,
                    hybrid_test_predicted,
                    np.where(hybrid_test_predicted, "defect", "pass"),
                )
            )
            policy["branches"]["hybrid_fusion"] = {
                "base": "frozen_spatial_consensus",
                "rescue": {
                    first_name: {
                        "score_strategy": first_strategy,
                        "threshold": first_threshold,
                    },
                    second_name: {
                        "score_strategy": second_strategy,
                        "threshold": second_threshold,
                    },
                },
                "validation_fnr_limit_with_safety_margin": hybrid_fnr_limit,
                "validation_metrics": hybrid_val_metrics,
                "test_metrics_for_report_only": hybrid_test_metrics,
            }

    if base_validation is not None and base_test is not None:
        val_spatial = base_validation["decision"].to_numpy() == "defect"
        test_spatial = base_test["decision"].to_numpy() == "defect"
        val_spatial_metrics = binary_metrics(val_labels, val_spatial)
        test_spatial_metrics = binary_metrics(test_labels, test_spatial)
        summary_rows.append(
            {
                "branch": "spatial_consensus",
                "mode": "fully_automatic",
                "threshold_low": float("nan"),
                "threshold_high": float("nan"),
                "val_fnr": val_spatial_metrics["fnr"],
                "val_fpr": val_spatial_metrics["fpr"],
                "test_fnr": test_spatial_metrics["fnr"],
                "test_fpr": test_spatial_metrics["fpr"],
                "test_recall": test_spatial_metrics["recall"],
                "test_specificity": test_spatial_metrics["specificity"],
                "test_accuracy": test_spatial_metrics["accuracy"],
                "test_review_rate": 0.0,
                "test_good_attention_rate": test_spatial_metrics["fpr"],
                "val_auc": float("nan"),
                "test_auc": float("nan"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "model_comparison.csv", index=False)
    per_image.to_csv(output_dir / "per_image_predictions.csv", index=False)
    per_validation.to_csv(
        output_dir / "per_validation_oof_predictions.csv", index=False
    )
    pd.DataFrame(group_metrics).to_csv(
        output_dir / "defect_group_comparison.csv", index=False
    )
    (output_dir / "learned_policy.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== TEST COMPARISON ===")
    print(
        summary[
            [
                "branch",
                "mode",
                "test_fnr",
                "test_fpr",
                "test_review_rate",
                "test_good_attention_rate",
                "test_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved learned-verifier report to: {output_dir}")


if __name__ == "__main__":
    main()
