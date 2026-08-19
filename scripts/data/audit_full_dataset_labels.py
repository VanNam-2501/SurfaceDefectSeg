"""Audit label quality across Train/Validation/Test using frozen model policies.

This is deliberately *not* an experiment and it does not report a new Test
score.  The automatic component policies are first selected on Validation by
``adaptive_component_policy.py``.  This program reads those frozen settings
without changing them, applies them to every requested split, and builds a
review queue to distinguish likely annotation issues from model-specific
errors.

The output is evidence for human inspection, not an automatic relabelling:

* a non-empty mask on a Good image or an empty mask on a Defect image is a
  direct label/mask consistency error;
* three independent models disagreeing with the image label is a high-priority
  label candidate (but can still be a shared model failure);
* one/two models disagreeing is useful evidence of a model-specific weakness.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "threecad_segmentation"
REPORTING_ROOT = REPO_ROOT / "scripts" / "reporting"
for module_root in (SOURCE_ROOT, REPORTING_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from build_test_case_audit import gt_overlay, probability_image, resize, safe_name
from decision_policy import border_connected_dark_roi
from fullres_eval import read_rgb, resolve_path, row_group, row_id


MODELS = ("unet", "segformer", "vmamba")


def parse_prediction_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--prediction needs model=probability-root, got {value!r}")
        name, raw = value.split("=", 1)
        name = name.strip().lower()
        path = Path(raw.strip()).expanduser().resolve()
        if name not in MODELS:
            raise ValueError(f"Unsupported model {name!r}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        result[name] = path
    if set(result) != set(MODELS):
        raise ValueError("Supply all three predictions: unet, segformer and vmamba")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frozen-policy", required=True, help="adaptive_component_policy.json from Validation")
    parser.add_argument("--prediction", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument(
        "--preview-limit", type=int, default=0,
        help="Maximum visual review boards; 0 means every suspicious candidate.",
    )
    parser.add_argument("--preview-size", type=int, default=288)
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def probability(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def frozen_component_decision(
    prob: np.ndarray, gray: np.ndarray, roi: np.ndarray, config: dict[str, float | int]
) -> tuple[bool, int]:
    """Evaluate exactly one frozen adaptive-component policy efficiently."""
    low = float(config["low_threshold"])
    peak = float(config["peak_threshold"])
    min_area = int(config["min_area_px"])
    min_persistent = int(config["min_persistent_area_px"])
    min_contrast = float(config["min_local_contrast"])
    binary = ((prob >= low) & roi).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return False, 0

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    maxima = np.zeros(count, dtype=np.float32)
    np.maximum.at(maxima, labels.ravel(), prob.ravel())
    peaks = maxima[1:]
    persistent = np.zeros(count - 1, dtype=np.int32)
    high_ids = labels[(prob >= peak) & (labels > 0)]
    if high_ids.size:
        persistent = np.bincount(high_ids, minlength=count)[1:]
    candidates = np.flatnonzero(
        (areas >= min_area) & (peaks >= peak) & (persistent >= min_persistent)
    )
    if not len(candidates):
        return False, 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    accepted = 0
    for index in candidates:
        component = labels == (int(index) + 1)
        ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component & roi
        inside = float(gray[component].mean())
        outside = float(gray[ring].mean()) if ring.any() else inside
        if abs(inside - outside) >= min_contrast:
            accepted += 1
    return accepted > 0, accepted


def load_mask(path: Path | None, expected_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    if path is None or not path.is_file():
        return np.zeros(expected_shape, dtype=bool), "missing"
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != expected_shape:
        return np.zeros(expected_shape, dtype=bool), f"shape_mismatch:{mask.shape[1]}x{mask.shape[0]}"
    return mask, "ok"


def review_reason(label: int, defect_votes: int, mask_status: str, mask_pixels: int) -> tuple[str, int]:
    if (label == 1 and (mask_status != "ok" or mask_pixels == 0)) or (label == 0 and mask_pixels > 0):
        return "label_mask_inconsistent", 1000
    if label == 0 and defect_votes == 3:
        return "good_label_three_model_defect", 900
    if label == 1 and defect_votes == 0:
        return "defect_label_three_model_pass", 900
    if label == 0 and defect_votes:
        return "good_partial_false_alarm", 600 + 20 * defect_votes
    if label == 1 and defect_votes < 3:
        return "defect_partial_miss", 600 + 20 * (3 - defect_votes)
    return "models_and_label_agree", 0


def build_records(
    dataset: Path,
    roots: dict[str, Path],
    policy: dict[str, object],
    splits: tuple[str, ...],
) -> pd.DataFrame:
    roi_threshold = int(policy.get("roi_border_dark_threshold", 5))
    policy_models = policy.get("models", {})
    if not isinstance(policy_models, dict) or set(policy_models) != set(MODELS):
        raise ValueError("Frozen policy does not contain U-Net, SegFormer and VMamba configurations")
    records: list[dict[str, object]] = []
    for split in splits:
        csv_path = dataset / "dataset_audit" / "splits" / f"{split}.csv"
        frame = pd.read_csv(csv_path).fillna("")
        for position, (_, row) in enumerate(frame.iterrows(), start=1):
            image_id = row_id(row)
            label = int(row["label"])
            image_path = resolve_path(dataset, row["image_path"])
            image = read_rgb(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            roi = border_connected_dark_roi(image, threshold=roi_threshold)
            raw_mask = str(row.get("mask_path", "")).strip()
            mask_path = resolve_path(dataset, raw_mask) if raw_mask else None
            if label == 0 and mask_path is None:
                # Good samples normally have no binary mask file.  This is an
                # expected state, not a data-integrity problem.
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask_status = "not_required_good"
            else:
                mask, mask_status = load_mask(mask_path, image.shape[:2])
            item: dict[str, object] = {
                "split": split,
                "image_id": image_id,
                "label": label,
                "label_name": "Defect" if label else "Good",
                "defect_group": row_group(row),
                "source_image_path": str(image_path),
                "source_mask_path": str(mask_path) if mask_path else "",
                "mask_status": mask_status,
                "mask_pixels": int(mask.sum()),
            }
            defect_votes = 0
            for model in MODELS:
                map_path = roots[model] / split / "probability" / f"{image_id}.png"
                if not map_path.is_file():
                    raise FileNotFoundError(f"Missing {model}/{split} map: {map_path}")
                prob = probability(map_path)
                if prob.shape != image.shape[:2]:
                    raise ValueError(f"Map shape mismatch: {model}/{split}/{image_id}")
                model_policy = policy_models[model]
                if not isinstance(model_policy, dict) or "config" not in model_policy:
                    raise ValueError(f"Frozen policy has no config for {model}")
                predicted, accepted = frozen_component_decision(prob, gray, roi, model_policy["config"])
                decision = "defect" if predicted else "pass"
                defect_votes += int(predicted)
                item[f"{model}_decision"] = decision
                item[f"{model}_accepted_components"] = accepted
                item[f"{model}_peak_probability"] = float(prob.max())
                item[f"probability_{model}_path"] = str(map_path)
            item["defect_model_votes"] = defect_votes
            item["model_agreement"] = (
                "all_defect" if defect_votes == 3 else "all_pass" if defect_votes == 0 else "disagree"
            )
            reason, priority = review_reason(label, defect_votes, mask_status, int(mask.sum()))
            item["review_reason"] = reason
            item["review_priority"] = priority
            records.append(item)
            if position % 50 == 0 or position == len(frame):
                print(f"[audit:{split}] {position}/{len(frame)}", flush=True)
    return pd.DataFrame(records)


def preview_card(width: int, height: int, record: pd.Series) -> Image.Image:
    serious = str(record["review_reason"]) != "models_and_label_agree"
    color = (190, 43, 43) if serious else (24, 116, 77)
    card = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    lines = [
        "DATA AUDIT (not a metric)",
        f"Label: {record['label_name']}",
        f"U/S/M: {record['unet_decision'][0].upper()} / {record['segformer_decision'][0].upper()} / {record['vmamba_decision'][0].upper()}",
        str(record["review_reason"]),
        f"split: {record['split']}",
        str(record["image_id"]),
    ]
    y = 14
    for line in lines:
        draw.text((12, y), line, fill="white", font=font)
        y += 22
    return card


def preview(record: pd.Series, width: int) -> Image.Image:
    image = Image.fromarray(read_rgb(Path(record["source_image_path"])))
    original = resize(image, width)
    gt = resize(gt_overlay(image, str(record["source_mask_path"])), width)
    maps = [str(record[f"probability_{model}_path"]) for model in MODELS]
    evidence = probability_image(maps, original.size)
    conclusion = preview_card(original.width, original.height, record)
    canvas = Image.new("RGB", (original.width * 4, original.height + 24), "white")
    for index, panel in enumerate((original, gt, evidence, conclusion)):
        canvas.paste(panel, (index * original.width, 24))
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(("Input", "Ground truth", "Mean U/S/M probability", "Audit finding")):
        draw.text((index * original.width + 4, 4), title, fill="black", font=ImageFont.load_default())
    return canvas


def write_outputs(frame: pd.DataFrame, output: Path, preview_limit: int, preview_size: int, no_previews: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["review_priority", "split", "image_id"], ascending=[False, True, True])
    frame.to_csv(output / "01_all_images_frozen_policy.csv", index=False)
    queue = frame[frame["review_reason"] != "models_and_label_agree"].copy()
    queue.to_csv(output / "02_review_queue.csv", index=False)
    summary = (
        frame.groupby(["split", "review_reason", "model_agreement"], as_index=False)
        .size().rename(columns={"size": "images"})
    )
    summary.to_csv(output / "03_summary_by_split_and_reason.csv", index=False)
    integrity = frame[
        ((frame["label"] == 1) & (frame["mask_status"] != "ok"))
        | ((frame["label"] == 0) & (frame["mask_pixels"] > 0))
    ].copy()
    integrity.to_csv(output / "04_mask_integrity_issues.csv", index=False)

    model_rows: list[dict[str, object]] = []
    for split, split_frame in frame.groupby("split"):
        for model in MODELS:
            pred = split_frame[f"{model}_decision"] == "defect"
            label = split_frame["label"] == 1
            model_rows.append({
                "split": split, "model": model,
                "correct_pass": int((~label & ~pred).sum()),
                "correct_defect": int((label & pred).sum()),
                "missed_defect": int((label & ~pred).sum()),
                "false_alarm": int((~label & pred).sum()),
            })
    pd.DataFrame(model_rows).to_csv(output / "05_frozen_policy_model_outcomes_by_split.csv", index=False)

    readme = """# Full-dataset label audit

Đây là kiểm tra dữ liệu độc lập, không phải thí nghiệm và không được dùng để đổi threshold/policy.
Policy component của U-Net, SegFormer và VMamba đã được chốt ở Validation; cùng policy đó được áp dụng bất biến lên Train, Validation và Test.

`label_mask_inconsistent` là lỗi kiểm tra trực tiếp label/mask. Các nhóm `*_three_model_*` là ứng viên ưu tiên để người gán nhãn xem lại: đồng thuận ba model là bằng chứng mạnh hơn một model, nhưng không tự động chứng minh label sai.

`02_review_queue.csv` có đường dẫn ảnh gốc, mask GT, map xác suất từng model và lý do xếp hàng. Không file nào ở đây tự sửa label/mask.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    if no_previews:
        return
    selected = queue if preview_limit == 0 else queue.head(preview_limit)
    gallery = output / "visual_gallery"
    for position, (_, record) in enumerate(selected.iterrows(), start=1):
        destination = gallery / str(record["review_reason"]) / str(record["split"]) / f"{safe_name(str(record['image_id']))}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        preview(record, preview_size).save(destination, quality=84, optimize=True)
        if position % 100 == 0 or position == len(selected):
            print(f"[audit previews] {position}/{len(selected)}", flush=True)


def main() -> None:
    args = arguments()
    if args.preview_limit < 0 or args.preview_size < 96:
        raise ValueError("--preview-limit must be >= 0 and --preview-size must be >= 96")
    dataset = Path(args.dataset_root).expanduser().resolve()
    policy_path = Path(args.frozen_policy).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    roots = parse_prediction_specs(args.prediction)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if output.exists():
        shutil.rmtree(output)
    frame = build_records(dataset, roots, policy, tuple(args.splits))
    write_outputs(frame, output, args.preview_limit, args.preview_size, args.no_previews)
    print(f"FULL DATASET AUDIT READY: {output}", flush=True)
    print(f"Images audited: {len(frame)} | candidates: {(frame['review_priority'] > 0).sum()}", flush=True)


if __name__ == "__main__":
    main()
