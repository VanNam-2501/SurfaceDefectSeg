"""Post-processing policy for low-false-alarm defect decisions.

The segmentation networks produce probabilities.  This module turns those
probabilities into a reproducible three-level image decision:

PASS   - no meaningful component was found;
REVIEW - a tiny/high-confidence component or model disagreement was found;
DEFECT - enough models agree spatially on a meaningful component.

Policy values are calibrated on Validation and then frozen for Test/web use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


DECISIONS = ("pass", "review", "defect")
POLICY_SCHEMA_VERSION = 1


def load_decision_policy(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if int(policy.get("schema_version", 0)) != POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported decision policy schema at {policy_path}: "
            f"{policy.get('schema_version')!r}"
        )
    models = policy.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"Decision policy has no model settings: {policy_path}")
    return policy


def save_decision_policy(policy: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(policy), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def border_connected_dark_roi(image: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Keep the image except near-black regions connected to an outer border.

    This removes black camera padding/fixtures without deleting internal dark
    grooves.  A threshold of zero disables grayscale tolerance but still
    excludes exactly-black border padding.  A negative threshold disables ROI
    filtering completely.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}")
    if threshold < 0:
        return np.ones(image.shape[:2], dtype=bool)

    dark = np.max(image[..., :3], axis=2) <= int(threshold)
    count, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.ones(image.shape[:2], dtype=bool)

    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.ones(image.shape[:2], dtype=bool)
    return ~np.isin(labels, border_labels)


def filter_component_area(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(1, int(minimum_area)))
    keep = keep[keep != 0]
    return np.isin(labels, keep) if keep.size else np.zeros(mask.shape, dtype=bool)


def analyze_probability(
    probability: np.ndarray,
    model_policy: Mapping[str, Any],
    roi: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract strong and review-only connected components for one model."""
    prob = np.asarray(probability, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected 2-D probability, got {prob.shape}")
    if roi is None:
        roi = np.ones(prob.shape, dtype=bool)
    if roi.shape != prob.shape:
        raise ValueError(f"ROI shape {roi.shape} differs from probability {prob.shape}")

    threshold = float(model_policy["pixel_threshold"])
    minimum_area = max(1, int(model_policy["min_component_area_px"]))
    tiny_minimum = max(1, int(model_policy.get("tiny_min_area_px", 1)))
    tiny_threshold = float(model_policy.get("tiny_high_threshold", min(0.99, threshold + 0.20)))
    binary = ((prob >= threshold) & roi).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    strong_ids: list[int] = []
    review_ids: list[int] = []
    components: list[dict[str, Any]] = []
    if count > 1:
        maxima = np.zeros(count, dtype=np.float32)
        np.maximum.at(maxima, labels.ravel(), prob.ravel())
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            peak = float(maxima[component_id])
            kind = "ignored"
            if area >= minimum_area:
                strong_ids.append(component_id)
                kind = "strong"
            elif area >= tiny_minimum and peak >= tiny_threshold:
                review_ids.append(component_id)
                kind = "tiny_high_confidence"
            components.append(
                {
                    "area_px": area,
                    "peak_probability": peak,
                    "kind": kind,
                    "x": int(stats[component_id, cv2.CC_STAT_LEFT]),
                    "y": int(stats[component_id, cv2.CC_STAT_TOP]),
                    "width": int(stats[component_id, cv2.CC_STAT_WIDTH]),
                    "height": int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                }
            )

    strong_mask = np.isin(labels, strong_ids) if strong_ids else np.zeros(prob.shape, dtype=bool)
    review_mask = np.isin(labels, review_ids) if review_ids else np.zeros(prob.shape, dtype=bool)
    candidate_mask = strong_mask | review_mask
    return {
        "threshold": threshold,
        "min_component_area_px": minimum_area,
        "tiny_high_threshold": tiny_threshold,
        "strong_mask": strong_mask,
        "review_mask": review_mask,
        "candidate_mask": candidate_mask,
        "has_strong": bool(strong_ids),
        "has_candidate": bool(strong_ids or review_ids),
        "strong_component_count": len(strong_ids),
        "review_component_count": len(review_ids),
        "candidate_pixels": int(candidate_mask.sum()),
        "components": components,
    }


def required_majority(model_count: int, ensemble_policy: Mapping[str, Any]) -> int:
    if model_count <= 1:
        return 1
    configured = ensemble_policy.get("defect_votes")
    if configured is not None:
        return min(model_count, max(1, int(configured)))
    return model_count // 2 + 1


def combine_model_analyses(
    analyses: Mapping[str, Mapping[str, Any]],
    ensemble_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not analyses:
        raise ValueError("At least one model analysis is required")
    shapes = {tuple(np.asarray(item["strong_mask"]).shape) for item in analyses.values()}
    if len(shapes) != 1:
        raise ValueError(f"Model mask shapes differ: {sorted(shapes)}")
    shape = next(iter(shapes))

    names = list(analyses)
    strong_masks = [np.asarray(analyses[name]["strong_mask"], dtype=bool) for name in names]
    candidate_masks = [np.asarray(analyses[name]["candidate_mask"], dtype=bool) for name in names]
    votes_required = required_majority(len(names), ensemble_policy)
    dilation = max(0, int(ensemble_policy.get("agreement_dilation_px", 0)))
    consensus_minimum = max(1, int(ensemble_policy.get("min_consensus_area_px", 1)))

    vote_masks = strong_masks
    if dilation > 0:
        size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        vote_masks = [cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) for mask in strong_masks]
    vote_map = np.sum(np.stack(vote_masks, axis=0), axis=0, dtype=np.uint8)
    raw_consensus = vote_map >= votes_required
    consensus = filter_component_area(raw_consensus, consensus_minimum)
    is_defect = bool(consensus.any())
    has_candidate = any(bool(item["has_candidate"]) for item in analyses.values())

    union_strong = np.logical_or.reduce(strong_masks) if strong_masks else np.zeros(shape, dtype=bool)
    union_candidate = (
        np.logical_or.reduce(candidate_masks) if candidate_masks else np.zeros(shape, dtype=bool)
    )
    if is_defect:
        if dilation > 0:
            size = dilation * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            support = cv2.dilate(consensus.astype(np.uint8), kernel).astype(bool)
            final_mask = union_strong & support
            if not final_mask.any():
                final_mask = consensus
        else:
            final_mask = union_strong & consensus
            if not final_mask.any():
                final_mask = consensus
        decision = "defect"
        reason = f"{votes_required}/{len(names)} model(s) agree spatially"
    elif has_candidate:
        final_mask = union_candidate
        decision = "review"
        strong_names = [name for name in names if analyses[name]["has_strong"]]
        reason = (
            "models disagree or only one model found a strong component"
            if strong_names
            else "only tiny high-confidence component(s) were found"
        )
    else:
        final_mask = np.zeros(shape, dtype=bool)
        decision = "pass"
        reason = "no component passed the calibrated policy"

    return {
        "decision": decision,
        "reason": reason,
        "mask": final_mask,
        "consensus_mask": consensus,
        "models_used": names,
        "required_votes": votes_required,
        "max_spatial_votes": int(vote_map.max(initial=0)),
        "strong_models": [name for name in names if analyses[name]["has_strong"]],
        "candidate_models": [name for name in names if analyses[name]["has_candidate"]],
        "mask_pixels": int(final_mask.sum()),
    }


def apply_decision_policy(
    image: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    model_settings = policy["models"]
    missing = sorted(set(probabilities) - set(model_settings))
    if missing:
        raise ValueError(f"Policy has no settings for model(s): {', '.join(missing)}")
    roi_policy = policy.get("roi", {})
    roi = border_connected_dark_roi(
        image, threshold=int(roi_policy.get("border_dark_threshold", 5))
    )
    analyses = {
        name: analyze_probability(probability, model_settings[name], roi=roi)
        for name, probability in probabilities.items()
    }
    combined = combine_model_analyses(analyses, policy.get("ensemble", {}))
    combined["analyses"] = analyses
    combined["roi"] = roi
    return combined

