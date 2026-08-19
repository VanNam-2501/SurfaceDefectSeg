"""Evaluate one frozen decision policy without changing it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "threecad_segmentation"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from calibrate_decision_policy import prediction_specs, probability_path, read_probability
from decision_policy import apply_decision_policy, load_decision_policy
from fullres_eval import read_binary_mask, read_rgb, resolve_path, row_group, row_id


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def atomic_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(mask.astype(np.uint8) * 255).save(temporary, format="PNG", optimize=True)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--csv", default="")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--prediction", action="append", default=[], help="model=prediction_root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="Smoke-test limit; 0 uses the full split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    csv_path = (
        Path(args.csv).expanduser().resolve()
        if args.csv
        else dataset_root / "dataset_audit" / "splits" / f"{args.split}.csv"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_decision_policy(args.policy)
    predictions = prediction_specs(args.prediction)
    expected = set(policy["models"])
    supplied = set(predictions)
    if supplied != expected:
        raise ValueError(
            "Frozen evaluation requires exactly the calibrated models. "
            f"Expected {sorted(expected)}, supplied {sorted(supplied)}"
        )
    records = pd.read_csv(csv_path).fillna("")
    if args.max_images > 0:
        records = records.head(args.max_images).copy()

    rows: list[dict[str, Any]] = []
    alert_tp = alert_fn = 0
    defect_tp = defect_fn = defect_fp = defect_tn = 0
    positive_review = good_review = 0
    positive_dices: list[float] = []
    for index, row in records.iterrows():
        image_id = row_id(row)
        label = int(row["label"])
        image = read_rgb(resolve_path(dataset_root, row["image_path"]))
        probabilities = {
            model: read_probability(probability_path(root, args.split, image_id))
            for model, root in predictions.items()
        }
        result = apply_decision_policy(image, probabilities, policy)
        decision = result["decision"]
        alert = decision != "pass"
        defect = decision == "defect"
        if label:
            alert_tp += int(alert)
            alert_fn += int(not alert)
            defect_tp += int(defect)
            defect_fn += int(not defect)
            positive_review += int(decision == "review")
            ground_truth = read_binary_mask(
                resolve_path(dataset_root, row["mask_path"])
            ).astype(bool)
            mask = np.asarray(result["mask"], dtype=bool)
            intersection = int(np.count_nonzero(mask & ground_truth))
            denominator = int(mask.sum()) + int(ground_truth.sum())
            dice = 2.0 * intersection / denominator if denominator else 1.0
            positive_dices.append(dice)
        else:
            defect_fp += int(defect)
            defect_tn += int(not defect)
            good_review += int(decision == "review")
            dice = float("nan")

        if args.save_masks:
            atomic_mask(np.asarray(result["mask"], dtype=bool), output_dir / "masks" / f"{image_id}.png")
        rows.append(
            {
                "image_id": image_id,
                "split": args.split,
                "label": label,
                "label_name": "Defect" if label else "Good",
                "defect_group": row_group(row),
                "decision": decision,
                "reason": result["reason"],
                "alert_pred": int(alert),
                "defect_pred": int(defect),
                "required_votes": int(result["required_votes"]),
                "max_spatial_votes": int(result["max_spatial_votes"]),
                "strong_models": ",".join(result["strong_models"]),
                "candidate_models": ",".join(result["candidate_models"]),
                "predicted_positive_pixels": int(result["mask_pixels"]),
                "positive_dice": dice,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"[{args.split}] {index + 1}/{len(records)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_image_decisions.csv", index=False)
    positive = int((frame["label"] == 1).sum())
    good = int((frame["label"] == 0).sum())
    metrics = {
        "split": args.split,
        "policy": str(Path(args.policy).resolve()),
        "models": list(predictions),
        "images": int(len(frame)),
        "positive_images": positive,
        "good_images": good,
        "alert_fnr": safe_div(alert_fn, positive),
        "alert_recall": safe_div(alert_tp, positive),
        "defect_recall": safe_div(defect_tp, positive),
        "defect_fnr": safe_div(defect_fn, positive),
        "defect_fpr": safe_div(defect_fp, good),
        "good_review_rate": safe_div(good_review, good),
        "positive_review_rate": safe_div(positive_review, positive),
        "overall_review_rate": safe_div(positive_review + good_review, len(frame)),
        "positive_dice": float(np.mean(positive_dices)) if positive_dices else 0.0,
        "decision_counts": {
            str(key): int(value) for key, value in frame["decision"].value_counts().items()
        },
    }
    (output_dir / "decision_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    confusion = pd.crosstab(frame["label_name"], frame["decision"], margins=True)
    confusion.to_csv(output_dir / "decision_confusion.csv")
    group_metrics = []
    for group, subset in frame.groupby("defect_group", dropna=False):
        positives = subset[subset["label"] == 1]
        group_metrics.append(
            {
                "defect_group": group,
                "images": int(len(subset)),
                "positive_images": int(len(positives)),
                "alert_recall": float((positives["decision"] != "pass").mean())
                if len(positives)
                else float("nan"),
                "defect_recall": float((positives["decision"] == "defect").mean())
                if len(positives)
                else float("nan"),
                "review_rate": float((subset["decision"] == "review").mean()),
            }
        )
    pd.DataFrame(group_metrics).to_csv(output_dir / "defect_group_decisions.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Frozen policy evaluation saved to: {output_dir}")


if __name__ == "__main__":
    main()
