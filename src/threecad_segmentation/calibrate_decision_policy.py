"""Calibrate a low-false-alarm decision policy using Validation only."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from decision_policy import (
    POLICY_SCHEMA_VERSION,
    analyze_probability,
    border_connected_dark_roi,
    filter_component_area,
    save_decision_policy,
)
from fullres_eval import read_binary_mask, read_rgb, resolve_path, row_group, row_id


def parse_float_grid(value: str) -> list[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(not 0.0 < item < 1.0 for item in result):
        raise argparse.ArgumentTypeError("threshold grid must contain values between 0 and 1")
    return result


def parse_int_grid(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("integer grid must contain non-negative values")
    return result


def prediction_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction must use model=path syntax: {value}")
        model, raw_path = value.split("=", 1)
        model = model.strip().lower()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not model or not path.is_dir():
            raise ValueError(f"Invalid prediction source: {value}")
        result[model] = path
    if not result:
        raise ValueError("Supply at least one --prediction model=path")
    return result


def probability_path(root: Path, split: str, image_id: str) -> Path:
    candidates = (
        root / split / "probability" / f"{image_id}.png",
        root / "probability" / f"{image_id}.png",
        root / split / f"{image_id}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No probability PNG for {image_id} below {root}")


def read_probability(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array.astype(np.float32) / 255.0


def empty_counts() -> dict[str, float]:
    return {
        "positive": 0,
        "good": 0,
        "alert_tp": 0,
        "alert_fn": 0,
        "defect_tp": 0,
        "defect_fn": 0,
        "defect_fp": 0,
        "defect_tn": 0,
        "good_review": 0,
        "positive_review": 0,
        "dice_sum": 0.0,
        "dice_count": 0,
    }


def update_counts(
    counts: dict[str, float],
    label: int,
    defect: bool,
    alert: bool,
    dice: float | None,
) -> None:
    if label:
        counts["positive"] += 1
        counts["alert_tp"] += int(alert)
        counts["alert_fn"] += int(not alert)
        counts["defect_tp"] += int(defect)
        counts["defect_fn"] += int(not defect)
        counts["positive_review"] += int(alert and not defect)
        if dice is not None:
            counts["dice_sum"] += float(dice)
            counts["dice_count"] += 1
    else:
        counts["good"] += 1
        counts["defect_fp"] += int(defect)
        counts["defect_tn"] += int(not defect)
        counts["good_review"] += int(alert and not defect)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def finalize_counts(counts: dict[str, float]) -> dict[str, float]:
    positive = counts["positive"]
    good = counts["good"]
    total = positive + good
    return {
        "alert_fnr": safe_div(counts["alert_fn"], positive),
        "alert_recall": safe_div(counts["alert_tp"], positive),
        "defect_recall": safe_div(counts["defect_tp"], positive),
        "defect_fnr": safe_div(counts["defect_fn"], positive),
        "defect_fpr": safe_div(counts["defect_fp"], good),
        "good_review_rate": safe_div(counts["good_review"], good),
        "positive_review_rate": safe_div(counts["positive_review"], positive),
        "overall_review_rate": safe_div(
            counts["good_review"] + counts["positive_review"], total
        ),
        "positive_dice": safe_div(counts["dice_sum"], counts["dice_count"]),
        "positive_images": int(positive),
        "good_images": int(good),
    }


def choose_configuration(
    frame: pd.DataFrame,
    fnr_limit: float,
    min_defect_recall: float,
    review_cost: float,
) -> tuple[pd.Series, str]:
    scored = frame.copy()
    scored["operating_cost"] = scored["defect_fpr"] + review_cost * scored["good_review_rate"]
    strict = scored[
        (scored["alert_fnr"] <= fnr_limit + 1e-12)
        & (scored["defect_recall"] >= min_defect_recall - 1e-12)
    ]
    if len(strict):
        pool = strict
        status = "all_constraints_satisfied"
    else:
        alert_only = scored[scored["alert_fnr"] <= fnr_limit + 1e-12]
        if len(alert_only):
            pool = alert_only
            status = "defect_recall_constraint_relaxed"
        else:
            best_fnr = float(scored["alert_fnr"].min())
            pool = scored[scored["alert_fnr"] <= best_fnr + 1e-12]
            status = "fnr_constraint_unattainable_fallback"
    sort_columns = [
        "operating_cost",
        "defect_fpr",
        "good_review_rate",
        "positive_dice",
        "defect_recall",
    ]
    ascending = [True, True, True, False, False]
    return pool.sort_values(sort_columns, ascending=ascending).iloc[0], status


def component_features(
    probability: np.ndarray,
    roi: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary = ((probability >= threshold) & roi).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    maxima = np.zeros(count, dtype=np.float32)
    np.maximum.at(maxima, labels.ravel(), probability.ravel())
    overlaps = np.bincount(
        labels[ground_truth.astype(bool)].ravel(), minlength=count
    ).astype(np.int64)
    return areas, maxima[1:], overlaps[1:]


def calibrate_model(
    model: str,
    prediction_root: Path,
    split: str,
    records: pd.DataFrame,
    dataset_root: Path,
    thresholds: list[float],
    minimum_areas: list[int],
    tiny_margin: float,
    roi_threshold: int,
) -> pd.DataFrame:
    keys = [(threshold, area) for threshold in thresholds for area in minimum_areas]
    accumulators = {key: empty_counts() for key in keys}
    for index, row in records.iterrows():
        image_id = row_id(row)
        image = read_rgb(resolve_path(dataset_root, row["image_path"]))
        probability = read_probability(probability_path(prediction_root, split, image_id))
        if probability.shape != image.shape[:2]:
            raise ValueError(f"Shape mismatch for {model}/{image_id}")
        label = int(row["label"])
        ground_truth = (
            read_binary_mask(resolve_path(dataset_root, row["mask_path"])).astype(bool)
            if label
            else np.zeros(probability.shape, dtype=bool)
        )
        roi = border_connected_dark_roi(image, roi_threshold)
        gt_area = int(ground_truth.sum())
        for threshold in thresholds:
            areas, maxima, overlaps = component_features(
                probability, roi, ground_truth, threshold
            )
            for minimum_area in minimum_areas:
                strong = areas >= minimum_area
                tiny_high = (~strong) & (areas >= 1) & (
                    maxima >= min(0.99, threshold + tiny_margin)
                )
                defect = bool(np.any(strong))
                alert = bool(defect or np.any(tiny_high))
                dice = None
                if label:
                    predicted = int(areas[strong].sum()) if areas.size else 0
                    true_positive = int(overlaps[strong].sum()) if overlaps.size else 0
                    denominator = predicted + gt_area
                    dice = 2.0 * true_positive / denominator if denominator else 1.0
                update_counts(accumulators[(threshold, minimum_area)], label, defect, alert, dice)
        if (index + 1) % 50 == 0:
            print(f"[{model}] calibrated features {index + 1}/{len(records)}", flush=True)

    rows = []
    for (threshold, minimum_area), counts in accumulators.items():
        rows.append(
            {
                "model": model,
                "pixel_threshold": threshold,
                "min_component_area_px": minimum_area,
                "tiny_high_threshold": min(0.99, threshold + tiny_margin),
            }
            | finalize_counts(counts)
        )
    return pd.DataFrame(rows)


def ensemble_scan(
    prediction_roots: dict[str, Path],
    model_policies: dict[str, dict[str, Any]],
    split: str,
    records: pd.DataFrame,
    dataset_root: Path,
    roi_threshold: int,
    dilations: list[int],
    consensus_areas: list[int],
) -> pd.DataFrame:
    model_names = list(prediction_roots)
    vote_options = list(range(1, len(model_names) + 1))
    keys = [
        (votes, dilation, area)
        for votes in vote_options
        for dilation in dilations
        for area in consensus_areas
    ]
    accumulators = {key: empty_counts() for key in keys}

    for index, row in records.iterrows():
        image_id = row_id(row)
        image = read_rgb(resolve_path(dataset_root, row["image_path"]))
        roi = border_connected_dark_roi(image, roi_threshold)
        label = int(row["label"])
        ground_truth = (
            read_binary_mask(resolve_path(dataset_root, row["mask_path"])).astype(bool)
            if label
            else np.zeros(image.shape[:2], dtype=bool)
        )
        analyses = {}
        for model in model_names:
            probability = read_probability(probability_path(prediction_roots[model], split, image_id))
            analyses[model] = analyze_probability(probability, model_policies[model], roi=roi)
        alert = any(item["has_candidate"] for item in analyses.values())
        strong_masks = [item["strong_mask"] for item in analyses.values()]

        for dilation in dilations:
            if dilation > 0:
                size = dilation * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                voting = [
                    cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
                    for mask in strong_masks
                ]
            else:
                voting = strong_masks
            vote_map = np.sum(np.stack(voting, axis=0), axis=0, dtype=np.uint8)
            for votes in vote_options:
                raw = vote_map >= votes
                for minimum_area in consensus_areas:
                    mask = filter_component_area(raw, minimum_area)
                    defect = bool(mask.any())
                    dice = None
                    if label:
                        intersection = int(np.count_nonzero(mask & ground_truth))
                        denominator = int(mask.sum()) + int(ground_truth.sum())
                        dice = 2.0 * intersection / denominator if denominator else 1.0
                    update_counts(
                        accumulators[(votes, dilation, minimum_area)],
                        label,
                        defect,
                        alert,
                        dice,
                    )
        if (index + 1) % 50 == 0:
            print(f"[ensemble] {index + 1}/{len(records)}", flush=True)

    rows = []
    for (votes, dilation, minimum_area), counts in accumulators.items():
        rows.append(
            {
                "defect_votes": votes,
                "agreement_dilation_px": dilation,
                "min_consensus_area_px": minimum_area,
            }
            | finalize_counts(counts)
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--val-csv", default="")
    parser.add_argument("--prediction", action="append", default=[], help="model=prediction_root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fnr-limit", type=float, default=0.02)
    parser.add_argument("--min-defect-recall", type=float, default=0.80)
    parser.add_argument("--review-cost", type=float, default=0.25)
    parser.add_argument(
        "--thresholds",
        type=parse_float_grid,
        default=parse_float_grid("0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"),
    )
    parser.add_argument(
        "--min-areas", type=parse_int_grid, default=parse_int_grid("1,2,4,8,16,32,64")
    )
    parser.add_argument("--tiny-margin", type=float, default=0.20)
    parser.add_argument("--border-dark-threshold", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=0, help="Smoke-test limit; 0 uses all Validation images")
    parser.add_argument(
        "--agreement-dilations", type=parse_int_grid, default=parse_int_grid("0,2,4,8")
    )
    parser.add_argument(
        "--consensus-areas", type=parse_int_grid, default=parse_int_grid("1,2,4,8,16")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.fnr_limit < 1.0:
        raise ValueError("--fnr-limit must be in [0, 1)")
    if not 0.0 <= args.min_defect_recall <= 1.0:
        raise ValueError("--min-defect-recall must be in [0, 1]")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    val_csv = (
        Path(args.val_csv).expanduser().resolve()
        if args.val_csv
        else dataset_root / "dataset_audit" / "splits" / "val.csv"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = prediction_specs(args.prediction)
    records = pd.read_csv(val_csv).fillna("")
    if args.max_images > 0:
        records = records.head(args.max_images).copy()
    if not len(records):
        raise ValueError(f"Validation split is empty: {val_csv}")

    chosen_models: dict[str, dict[str, Any]] = {}
    model_metrics: dict[str, Any] = {}
    for model, root in predictions.items():
        scan = calibrate_model(
            model=model,
            prediction_root=root,
            split="val",
            records=records,
            dataset_root=dataset_root,
            thresholds=args.thresholds,
            minimum_areas=args.min_areas,
            tiny_margin=args.tiny_margin,
            roi_threshold=args.border_dark_threshold,
        )
        scan["operating_cost"] = scan["defect_fpr"] + args.review_cost * scan["good_review_rate"]
        scan.to_csv(output_dir / f"{model}_policy_scan.csv", index=False)
        best, status = choose_configuration(
            scan, args.fnr_limit, args.min_defect_recall, args.review_cost
        )
        chosen_models[model] = {
            "pixel_threshold": float(best["pixel_threshold"]),
            "min_component_area_px": int(best["min_component_area_px"]),
            "tiny_min_area_px": 1,
            "tiny_high_threshold": float(best["tiny_high_threshold"]),
        }
        model_metrics[model] = {
            "selection_status": status,
            **{
                key: float(best[key])
                for key in (
                    "alert_fnr",
                    "alert_recall",
                    "defect_recall",
                    "defect_fpr",
                    "good_review_rate",
                    "positive_dice",
                    "operating_cost",
                )
            },
        }
        print(f"Selected {model}: {chosen_models[model]} | {model_metrics[model]}")

    ensemble = ensemble_scan(
        prediction_roots=predictions,
        model_policies=chosen_models,
        split="val",
        records=records,
        dataset_root=dataset_root,
        roi_threshold=args.border_dark_threshold,
        dilations=args.agreement_dilations,
        consensus_areas=[max(1, item) for item in args.consensus_areas],
    )
    ensemble["operating_cost"] = (
        ensemble["defect_fpr"] + args.review_cost * ensemble["good_review_rate"]
    )
    ensemble.to_csv(output_dir / "ensemble_policy_scan.csv", index=False)
    best_ensemble, ensemble_status = choose_configuration(
        ensemble, args.fnr_limit, args.min_defect_recall, args.review_cost
    )
    ensemble_policy = {
        "mode": "spatial_vote",
        "defect_votes": int(best_ensemble["defect_votes"]),
        "agreement_dilation_px": int(best_ensemble["agreement_dilation_px"]),
        "min_consensus_area_px": int(best_ensemble["min_consensus_area_px"]),
    }
    ensemble_metrics = {
        "selection_status": ensemble_status,
        **{
            key: float(best_ensemble[key])
            for key in (
                "alert_fnr",
                "alert_recall",
                "defect_recall",
                "defect_fpr",
                "good_review_rate",
                "positive_review_rate",
                "overall_review_rate",
                "positive_dice",
                "operating_cost",
            )
        },
    }
    policy = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": chosen_models,
        "roi": {
            "mode": "exclude_border_connected_dark",
            "border_dark_threshold": int(args.border_dark_threshold),
        },
        "ensemble": ensemble_policy,
        "targets": {
            "max_alert_fnr": float(args.fnr_limit),
            "min_defect_recall": float(args.min_defect_recall),
            "review_cost": float(args.review_cost),
        },
        "calibration": {
            "split": "val",
            "csv": str(val_csv),
            "image_count": int(len(records)),
            "positive_images": int((records["label"].astype(int) == 1).sum()),
            "good_images": int((records["label"].astype(int) == 0).sum()),
            "prediction_sources": {name: str(path) for name, path in predictions.items()},
            "model_metrics": model_metrics,
            "ensemble_metrics": ensemble_metrics,
        },
    }
    policy_path = save_decision_policy(policy, output_dir / "decision_policy.json")
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(policy["calibration"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Selected ensemble: {ensemble_policy} | {ensemble_metrics}")
    print(f"Frozen Validation policy: {policy_path}")


if __name__ == "__main__":
    main()
