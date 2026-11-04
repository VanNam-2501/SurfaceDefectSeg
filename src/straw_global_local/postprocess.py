"""Validation-fitted image-level B1 decision policy.

The classifier has no pixel mask, so spatial morphology is not applicable here.
This policy changes only the Good/defect operating point. Defect subtype remains
the highest-scoring of the four defect classes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def probabilities(logits, temperature: float = 1.0) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError('Expected finite logits with shape (N, 5)')
    values = values / temperature
    values -= values.max(axis=1, keepdims=True)
    scores = np.exp(values)
    return scores / scores.sum(axis=1, keepdims=True)


def predict_from_probabilities(p: np.ndarray, threshold: float | None) -> np.ndarray:
    values = np.asarray(p, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError('Expected finite probabilities with shape (N, 5)')
    if threshold is None:
        return values.argmax(axis=1)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must lie strictly between 0 and 1')
    defect = (1.0 - values[:, 0]) >= threshold
    return np.where(defect, values[:, 1:].argmax(axis=1) + 1, 0)


def counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError('Labels and predictions must be aligned one-dimensional arrays')
    defect = labels != 0
    return {
        'fn': int(np.sum(defect & (predictions == 0))),
        'fp': int(np.sum(~defect & (predictions != 0))),
        'wrong_type': int(np.sum(defect & (predictions != 0) & (predictions != labels))),
        'total_wrong': int(np.sum(predictions != labels)),
    }


def select_zero_fp_threshold(labels: np.ndarray, p: np.ndarray) -> tuple[float | None, dict]:
    """Choose a validation-only threshold that reduces FN without increasing FP.

    Candidate thresholds are midpoints of distinct observed defect scores. The
    lowest FN wins; ties prefer fewer FP and then the highest threshold.
    If none improves on argmax under zero additional FP, return None.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != len(p) or not (labels == 0).any() or not (labels != 0).any():
        raise ValueError('Aligned validation labels must include Good and defect examples')
    score = 1.0 - np.asarray(p, dtype=np.float64)[:, 0]
    unique = np.unique(score)
    thresholds = [float(low / 2 + high / 2) for low, high in zip(unique[:-1], unique[1:])]
    baseline = counts(labels, predict_from_probabilities(p, None))
    eligible = []
    for threshold in thresholds:
        if not 0 < threshold < 1:
            continue
        current = counts(labels, predict_from_probabilities(p, threshold))
        if current['fp'] <= baseline['fp'] and current['fn'] < baseline['fn']:
            eligible.append((current['fn'], current['fp'], -threshold, threshold, current))
    if not eligible:
        return None, {'original': baseline, 'selected': baseline, 'reason': 'no_validation_improvement_without_extra_false_positives'}
    _, _, _, chosen, selected = min(eligible)
    return chosen, {'original': baseline, 'selected': selected, 'reason': 'fewer_validation_false_negatives_at_no_extra_false_positives'}


def apply_policy(logits: np.ndarray, policy: dict, checkpoint: str | Path) -> tuple[np.ndarray, np.ndarray]:
    if policy.get('schema') != 1 or policy.get('variant') != 'B1' or policy.get('input_size') != 512:
        raise ValueError('Unsupported B1 policy')
    if policy.get('checkpoint_sha256') != file_sha256(checkpoint):
        raise ValueError('Decision policy checkpoint hash does not match')
    p = probabilities(logits, float(policy.get('temperature', 1.0)))
    return predict_from_probabilities(p, policy.get('threshold')), p
