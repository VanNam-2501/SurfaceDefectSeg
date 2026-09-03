"""Calibrate a rule-only adaptive component policy for each segmentation model.

Unlike a global image score, this policy decides from connected components:

* component area at a permissive threshold;
* component peak probability;
* number of pixels that remain at a stricter threshold (persistence);
* local grayscale contrast against a narrow ring around the component.

No additional neural network is trained.  All rules are selected on Validation
and then frozen before Test is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from calibrate_decision_policy import prediction_specs, probability_path, read_probability
from decision_policy import border_connected_dark_roi
from fullres_eval import read_rgb, resolve_path, row_group, row_id


LOW_THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
# Include each low threshold itself.  This permits a broad, medium-confidence
# component to be accepted when its area/persistence is strong enough, while
# high-confidence tiny components remain a separate path.
PEAK_THRESHOLDS = (
    0.25,
    0.35,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
)
MIN_AREAS = (1, 2, 4, 8, 16, 32, 64, 128)
MIN_PERSISTENT_AREAS = (1, 2, 4, 8, 16)
MIN_CONTRASTS = (0.00, 0.01, 0.02, 0.04)
CACHE_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class PolicyConfig:
    low_threshold: float
    min_area: int
    peak_threshold: float
    min_persistent_area: int
    min_local_contrast: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "low_threshold": self.low_threshold,
            "min_area_px": self.min_area,
            "peak_threshold": self.peak_threshold,
            "min_persistent_area_px": self.min_persistent_area,
            "min_local_contrast": self.min_local_contrast,
        }


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def empty_component_frame() -> dict[str, np.ndarray]:
    return {
        "area": np.zeros(0, dtype=np.int32),
        "peak": np.zeros(0, dtype=np.float32),
        "contrast": np.zeros(0, dtype=np.float32),
        "persistent": np.zeros((0, len(PEAK_THRESHOLDS)), dtype=np.int32),
    }


def components_for_threshold(
    probability: np.ndarray, gray: np.ndarray, roi: np.ndarray, low_threshold: float
) -> dict[str, np.ndarray]:
    binary = ((probability >= low_threshold) & roi).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return empty_component_frame()
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    maxima = np.zeros(count, dtype=np.float32)
    np.maximum.at(maxima, labels.ravel(), probability.ravel())
    peaks = maxima[1:]
    persistent = np.zeros((count - 1, len(PEAK_THRESHOLDS)), dtype=np.int32)
    for index, threshold in enumerate(PEAK_THRESHOLDS):
        ids = labels[(probability >= threshold) & (labels > 0)]
        if ids.size:
            persistent[:, index] = np.bincount(ids, minlength=count)[1:]

    contrasts = np.zeros(count - 1, dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for component_id in range(1, count):
        component = labels == component_id
        ring = (
            cv2.dilate(component.astype(np.uint8), kernel).astype(bool)
            & ~component
            & roi
        )
        inside = float(gray[component].mean())
        outside = float(gray[ring].mean()) if ring.any() else inside
        contrasts[component_id - 1] = abs(inside - outside)
    return {
        "area": areas,
        "peak": peaks,
        "contrast": contrasts,
        "persistent": persistent,
    }


def component_cache_for_split(
    records: pd.DataFrame,
    split: str,
    dataset_root: Path,
    prediction_root: Path,
    roi_threshold: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(records.iterrows(), start=1):
        image_id = row_id(row)
        image = read_rgb(resolve_path(dataset_root, row["image_path"]))
        probability = read_probability(probability_path(prediction_root, split, image_id))
        if probability.shape != image.shape[:2]:
            raise ValueError(f"Probability shape mismatch: {image_id}")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        roi = border_connected_dark_roi(image, threshold=roi_threshold)
        by_low = {
            low: components_for_threshold(probability, gray, roi, low)
            for low in LOW_THRESHOLDS
        }
        result.append(
            {
                "image_id": image_id,
                "label": int(row["label"]),
                "label_name": "Defect" if int(row["label"]) else "Good",
                "defect_group": row_group(row),
                "components": by_low,
            }
        )
        if position % 25 == 0 or position == len(records):
            print(f"[components:{split}] {position}/{len(records)}", flush=True)
    return result


def accepted_components(frame: dict[str, np.ndarray], config: PolicyConfig) -> np.ndarray:
    peak_index = PEAK_THRESHOLDS.index(config.peak_threshold)
    return (
        (frame["area"] >= config.min_area)
        & (frame["peak"] >= config.peak_threshold)
        & (frame["persistent"][:, peak_index] >= config.min_persistent_area)
        & (frame["contrast"] >= config.min_local_contrast)
    )


def evaluate_config(records: list[dict[str, Any]], config: PolicyConfig) -> dict[str, float | int]:
    positive = good = tp = fn = fp = tn = 0
    component_total = 0
    for row in records:
        accepted = accepted_components(row["components"][config.low_threshold], config)
        predicted = bool(accepted.any())
        component_total += int(accepted.sum())
        if row["label"]:
            positive += 1
            tp += int(predicted)
            fn += int(not predicted)
        else:
            good += 1
            fp += int(predicted)
            tn += int(not predicted)
    return {
        "positive_images": positive,
        "good_images": good,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "fnr": safe_div(fn, positive),
        "fpr": safe_div(fp, good),
        "recall": safe_div(tp, positive),
        "specificity": safe_div(tn, good),
        "accuracy": safe_div(tp + tn, positive + good),
        "mean_accepted_components": safe_div(component_total, positive + good),
    }


def candidate_configs() -> list[PolicyConfig]:
    result: list[PolicyConfig] = []
    for low in LOW_THRESHOLDS:
        for peak in PEAK_THRESHOLDS:
            if peak < low:
                continue
            for area in MIN_AREAS:
                for persistent in MIN_PERSISTENT_AREAS:
                    for contrast in MIN_CONTRASTS:
                        result.append(
                            PolicyConfig(low, area, peak, persistent, contrast)
                        )
    return result


def select_config(
    records: list[dict[str, Any]], max_fnr: float
) -> tuple[PolicyConfig, dict[str, float | int], pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    configs = candidate_configs()
    for position, config in enumerate(configs, start=1):
        metrics = evaluate_config(records, config)
        rows.append(config.as_dict() | metrics)
        if position % 1000 == 0 or position == len(configs):
            print(f"[scan] {position}/{len(configs)}", flush=True)
    scan = pd.DataFrame(rows)
    strict = scan[scan["fnr"] <= max_fnr + 1e-12]
    if len(strict):
        pool = strict
        status = "max_fnr_satisfied"
    else:
        pool = scan[scan["fnr"] == scan["fnr"].min()]
        status = "max_fnr_unattainable_fallback"
    selected = pool.sort_values(
        ["fpr", "fnr", "mean_accepted_components", "recall"],
        ascending=[True, True, True, False],
    ).iloc[0]
    config = PolicyConfig(
        low_threshold=float(selected["low_threshold"]),
        min_area=int(selected["min_area_px"]),
        peak_threshold=float(selected["peak_threshold"]),
        min_persistent_area=int(selected["min_persistent_area_px"]),
        min_local_contrast=float(selected["min_local_contrast"]),
    )
    return config, evaluate_config(records, config), scan, status


def test_rows(records: list[dict[str, Any]], config: PolicyConfig, model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in records:
        accepted = accepted_components(row["components"][config.low_threshold], config)
        rows.append(
            {
                "image_id": row["image_id"],
                "model": model,
                "label": row["label"],
                "label_name": row["label_name"],
                "defect_group": row["defect_group"],
                "decision": "defect" if accepted.any() else "pass",
                "accepted_component_count": int(accepted.sum()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prediction", action="append", default=[], help="model=prediction_root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-fnr", type=float, default=0.02)
    parser.add_argument("--border-dark-threshold", type=int, default=5)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def _stat_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


def cache_fingerprint(
    records: pd.DataFrame,
    split: str,
    dataset_root: Path,
    prediction_root: Path,
    roi_threshold: int,
) -> str:
    """Identify every input that can change the component cache result."""
    payload: list[Any] = [
        CACHE_SCHEMA_VERSION,
        split,
        str(dataset_root.resolve()),
        str(prediction_root.resolve()),
        int(roi_threshold),
        LOW_THRESHOLDS,
        PEAK_THRESHOLDS,
        [(row_id(row), str(row["image_path"]), int(row["label"])) for _, row in records.iterrows()],
    ]
    for _, row in records.iterrows():
        image_id = row_id(row)
        image_path = resolve_path(dataset_root, row["image_path"])
        probability_path_value = probability_path(prediction_root, split, image_id)
        payload.append(
            (
                image_id,
                _stat_signature(image_path),
                _stat_signature(probability_path_value),
            )
        )
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_build_cache(
    cache_path: Path,
    records: pd.DataFrame,
    split: str,
    dataset_root: Path,
    prediction_root: Path,
    roi_threshold: int,
    rebuild: bool,
) -> list[dict[str, Any]]:
    fingerprint = cache_fingerprint(
        records, split, dataset_root, prediction_root, roi_threshold
    )
    if cache_path.is_file() and not rebuild:
        try:
            with cache_path.open("rb") as handle:
                cache = pickle.load(handle)
        except (OSError, EOFError, pickle.PickleError):
            cache = None
        if (
            isinstance(cache, dict)
            and cache.get("schema_version") == CACHE_SCHEMA_VERSION
            and cache.get("fingerprint") == fingerprint
            and isinstance(cache.get("records"), list)
            and len(cache["records"]) == len(records)
        ):
            print(f"[components:{split}] using cache {cache_path}")
            return cache["records"]
        print(f"[components:{split}] cache is stale; rebuilding {cache_path}")
    component_records = component_cache_for_split(
        records, split, dataset_root, prediction_root, roi_threshold
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "records": component_records,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(cache_path)
    return component_records


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.max_fnr < 1.0:
        raise ValueError("--max-fnr must be in [0, 1)")
    predictions = prediction_specs(args.prediction)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = dataset_root / "dataset_audit" / "splits"
    validation = pd.read_csv(split_root / "val.csv").fillna("")
    test = pd.read_csv(split_root / "test.csv").fillna("")
    summary_rows: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    policy: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "type": "adaptive_component_evidence",
        "validation_target_max_fnr": args.max_fnr,
        "roi_border_dark_threshold": args.border_dark_threshold,
        "models": {},
    }
    for model, prediction_root in predictions.items():
        cache_dir = output_dir / "component_cache" / model
        cache_dir.mkdir(parents=True, exist_ok=True)
        val_records = load_or_build_cache(
            cache_dir / "v2_val.pkl",
            validation,
            "val",
            dataset_root,
            prediction_root,
            args.border_dark_threshold,
            args.rebuild_cache,
        )
        test_records = load_or_build_cache(
            cache_dir / "v2_test.pkl",
            test,
            "test",
            dataset_root,
            prediction_root,
            args.border_dark_threshold,
            args.rebuild_cache,
        )
        config, val_metrics, scan, selection_status = select_config(val_records, args.max_fnr)
        scan.to_csv(output_dir / f"{model}_adaptive_scan.csv", index=False)
        test_metrics = evaluate_config(test_records, config)
        summary_rows.append(
            {
                "model": model,
                "selection_status": selection_status,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **config.as_dict(),
            }
        )
        rows = test_rows(test_records, config, model)
        all_test_rows.extend(rows)
        frame = pd.DataFrame(rows)
        for group, subset in frame.groupby("defect_group", dropna=False):
            positive = subset[subset["label"] == 1]
            if len(positive):
                group_rows.append(
                    {
                        "model": model,
                        "defect_group": group,
                        "positive_images": int(len(positive)),
                        "recall": float((positive["decision"] == "defect").mean()),
                    }
                )
        policy["models"][model] = {
            "selection_status": selection_status,
            "config": config.as_dict(),
            "validation_metrics": val_metrics,
            "test_metrics_for_report_only": test_metrics,
        }
        print(
            f"Selected {model}: {config.as_dict()} | "
            f"Validation FNR={float(val_metrics['fnr']):.4f}, FPR={float(val_metrics['fpr']):.4f} | "
            f"Test FNR={float(test_metrics['fnr']):.4f}, FPR={float(test_metrics['fpr']):.4f}",
            flush=True,
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "adaptive_model_comparison.csv", index=False)
    pd.DataFrame(all_test_rows).to_csv(output_dir / "adaptive_per_image_test.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "adaptive_defect_group_test.csv", index=False)
    (output_dir / "adaptive_component_policy.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== ADAPTIVE COMPONENT TEST RESULTS ===")
    print(
        summary[
            ["model", "test_fnr", "test_fpr", "test_recall", "test_specificity", "test_accuracy"]
        ].to_string(index=False)
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
