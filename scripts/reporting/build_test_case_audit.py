"""Create a complete, visual Test-set audit for every deployable experiment.

The decision pipeline writes aggregate metrics, but aggregate FNR/FPR alone
does not tell an annotator *which* samples caused them.  This utility turns
all final PASS/DEFECT branches into a traceable Test ledger:

* every Test image is classified as correct PASS, correct DEFECT, missed
  Defect (FN), or false alarm (FP);
* the corresponding original image, GT-mask and probability-map paths are
  included in CSV files;
* review-ready preview boards are generated for all images of U-Net,
  SegFormer, VMamba and the learned three-model fusion; and
* all FNs/FPs of every automatic experiment receive a visual preview.

This script never changes a threshold.  It only reads already-frozen Test
predictions, so it is safe to run after the Validation -> Test protocol.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "threecad_segmentation"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fullres_eval import read_rgb, resolve_path, row_group, row_id


OUTCOME_ORDER = (
    "correct_pass",
    "correct_defect",
    "missed_defect",
    "false_alarm",
)
OUTCOME_VI = {
    "correct_pass": "Dung: Good -> PASS (TN)",
    "correct_defect": "Dung: Defect -> DEFECT (TP)",
    "missed_defect": "Bo sot: Defect -> PASS (FN)",
    "false_alarm": "Bao dong gia: Good -> DEFECT (FP)",
}


def parse_key_paths(values: list[str], flag: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} needs model=path, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"{flag} has an empty model name")
        path = Path(raw.strip()).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{flag}: {path}")
        result[key] = path
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--decision-report", required=True)
    parser.add_argument(
        "--prediction", action="append", default=[],
        help="model=probability-cache-root; supply unet, segformer and vmamba",
    )
    parser.add_argument(
        "--base-test", action="append", default=[],
        help="model=per_image_metrics.csv produced by evaluate_model.py",
    )
    parser.add_argument("--preview-size", type=int, default=288)
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def outcome(label: int, decision: str) -> str:
    positive = int(label) == 1
    predicted = str(decision).lower() == "defect"
    if positive and predicted:
        return "correct_defect"
    if positive and not predicted:
        return "missed_defect"
    if not positive and predicted:
        return "false_alarm"
    return "correct_pass"


def model_set_for(experiment_id: str) -> list[str]:
    if experiment_id.startswith("base__") or experiment_id.startswith("adaptive__"):
        return [experiment_id.rsplit("__", 1)[-1]]
    if experiment_id.startswith("learned_all3__"):
        branch = experiment_id.split("__")[1]
        return [branch] if branch in {"unet", "segformer", "vmamba"} else ["unet", "segformer", "vmamba"]
    if experiment_id.startswith("hybrid_"):
        pair = experiment_id.split("__", 1)[0].removeprefix("hybrid_")
        return pair.split("_")
    return ["unet", "segformer", "vmamba"]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def build_dataset_index(dataset_root: Path) -> pd.DataFrame:
    csv_path = dataset_root / "dataset_audit" / "splits" / "test.csv"
    test = pd.read_csv(csv_path).fillna("")
    records: list[dict[str, object]] = []
    for _, row in test.iterrows():
        label = int(row["label"])
        records.append({
            "image_id": row_id(row),
            "label": label,
            "label_name": "Defect" if label else "Good",
            "defect_group": row_group(row),
            "source_image_path": str(resolve_path(dataset_root, row["image_path"])),
            "source_mask_path": str(resolve_path(dataset_root, row["mask_path"])) if label else "",
        })
    result = pd.DataFrame(records)
    if not result["image_id"].is_unique:
        raise ValueError("Test split has duplicate image_id values")
    return result


def source_rows(
    index: pd.DataFrame,
    experiment_id: str,
    family: str,
    decisions: pd.DataFrame,
    decision_column: str,
) -> pd.DataFrame:
    require_columns(decisions, ("image_id", decision_column), Path(experiment_id))
    current = decisions[["image_id", decision_column]].copy()
    current = current.rename(columns={decision_column: "decision"})
    current["decision"] = current["decision"].astype(str).str.lower()
    if not current["image_id"].is_unique:
        raise ValueError(f"{experiment_id} has duplicate image_id values")
    frame = index.merge(current, on="image_id", how="left", validate="one_to_one")
    if frame["decision"].isna().any():
        raise ValueError(f"{experiment_id} does not cover every Test image")
    invalid = sorted(set(frame["decision"]) - {"pass", "defect"})
    if invalid:
        raise ValueError(f"{experiment_id} has non-binary decisions: {invalid}")
    frame.insert(0, "experiment_id", experiment_id)
    frame.insert(1, "family", family)
    frame["models_used"] = "+".join(model_set_for(experiment_id))
    frame["outcome"] = [outcome(label, decision) for label, decision in zip(frame["label"], frame["decision"])]
    frame["outcome_vi"] = frame["outcome"].map(OUTCOME_VI)
    return frame


def gather_experiments(
    index: pd.DataFrame, report: Path, base_paths: dict[str, Path]
) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for model in ("unet", "segformer", "vmamba"):
        path = base_paths.get(model)
        if path is None:
            raise ValueError(f"Missing --base-test {model}=.../per_image_metrics.csv")
        base = pd.read_csv(path).fillna("")
        require_columns(base, ("image_id", "image_pred"), path)
        base["decision"] = np.where(base["image_pred"].astype(int) == 1, "defect", "pass")
        rows.append(source_rows(index, f"base__{model}", "raw_segmentation", base, "decision"))

    adaptive_path = report / "adaptive_single" / "adaptive_per_image_test.csv"
    adaptive = pd.read_csv(adaptive_path).fillna("")
    require_columns(adaptive, ("image_id", "model", "decision"), adaptive_path)
    for model in ("unet", "segformer", "vmamba"):
        rows.append(source_rows(
            index, f"adaptive__{model}", "adaptive_component_rule",
            adaptive[adaptive["model"].str.lower() == model], "decision",
        ))

    learned_path = report / "learned_all3" / "per_image_predictions.csv"
    learned = pd.read_csv(learned_path).fillna("")
    for branch in ("unet", "segformer", "vmamba", "fusion"):
        column = f"{branch}_automatic"
        rows.append(source_rows(
            index, f"learned_all3__{branch}__fully_automatic", "learned_verifier",
            learned, column,
        ))

    pairs = ("unet_segformer", "unet_vmamba", "segformer_vmamba")
    for pair in pairs:
        path = report / "hybrid_pairs" / pair / "per_image_predictions.csv"
        hybrid = pd.read_csv(path).fillna("")
        rows.append(source_rows(
            index, f"hybrid_{pair}__hybrid_fusion__fully_automatic", "automatic_pair_hybrid",
            hybrid, "hybrid_fusion_automatic",
        ))
    return rows


def probability_paths(frame: pd.DataFrame, roots: dict[str, Path]) -> pd.DataFrame:
    for model in ("unet", "segformer", "vmamba"):
        root = roots[model]
        paths = [root / "test" / "probability" / f"{image_id}.png" for image_id in frame["image_id"]]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {model} Test probability maps; first: {missing[0]}")
        frame[f"probability_{model}_path"] = [str(path) for path in paths]
    return frame


def write_case_csvs(all_cases: pd.DataFrame, audit: Path, report: Path) -> pd.DataFrame:
    columns = [
        "experiment_id", "family", "models_used", "outcome", "outcome_vi", "image_id",
        "label", "label_name", "defect_group", "decision", "source_image_path", "source_mask_path",
        "probability_unet_path", "probability_segformer_path", "probability_vmamba_path",
    ]
    all_cases = all_cases[columns].sort_values(["experiment_id", "outcome", "image_id"])
    all_cases.to_csv(audit / "01_all_experiments_all_test_images.csv", index=False)

    summary = (
        all_cases.groupby(["experiment_id", "family", "models_used", "outcome"], as_index=False)
        .size().rename(columns={"size": "images"})
    )
    pivot = summary.pivot_table(
        index=["experiment_id", "family", "models_used"], columns="outcome", values="images", fill_value=0
    ).reset_index()
    for name in OUTCOME_ORDER:
        if name not in pivot:
            pivot[name] = 0
    pivot["total_test_images"] = pivot[list(OUTCOME_ORDER)].sum(axis=1)
    pivot["fnr_pct"] = 100.0 * pivot["missed_defect"] / (pivot["missed_defect"] + pivot["correct_defect"]).clip(lower=1)
    pivot["fpr_pct"] = 100.0 * pivot["false_alarm"] / (pivot["false_alarm"] + pivot["correct_pass"]).clip(lower=1)
    pivot = pivot.sort_values("experiment_id")
    pivot.to_csv(audit / "02_summary_by_experiment.csv", index=False)
    pivot.to_csv(report / "tables" / "03_test_case_outcome_summary.csv", index=False)

    experiment_root = audit / "per_experiment"
    for experiment_id, subset in all_cases.groupby("experiment_id", sort=True):
        root = experiment_root / safe_name(experiment_id)
        root.mkdir(parents=True, exist_ok=True)
        subset.to_csv(root / "all_test_images.csv", index=False)
        for label in OUTCOME_ORDER:
            subset[subset["outcome"] == label].to_csv(root / f"{label}.csv", index=False)
    return all_cases


def resize(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def gt_overlay(image: Image.Image, mask_path: str) -> Image.Image:
    result = image.convert("RGBA")
    if mask_path:
        mask = Image.open(mask_path).convert("L")
        red = Image.new("RGBA", result.size, (235, 45, 45, 125))
        alpha = mask.point(lambda value: 150 if value else 0)
        red.putalpha(alpha)
        result.alpha_composite(red)
    return result.convert("RGB")


def probability_image(paths: list[str], size: tuple[int, int]) -> Image.Image:
    arrays: list[np.ndarray] = []
    for path in paths:
        arrays.append(np.asarray(Image.open(path).convert("L"), dtype=np.uint8))
    average = np.rint(np.mean(np.stack(arrays), axis=0)).astype(np.uint8)
    colored = cv2.applyColorMap(average, cv2.COLORMAP_MAGMA)
    image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    return image.resize(size, Image.Resampling.BILINEAR)


def title_card(width: int, height: int, record: pd.Series) -> Image.Image:
    is_error = record["outcome"] in {"missed_defect", "false_alarm"}
    color = (194, 43, 43) if is_error else (24, 116, 77)
    card = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    lines = [
        "FINAL DECISION",
        record["decision"].upper(),
        f"Truth: {record['label_name']}",
        record["outcome"].replace("_", " ").upper(),
        f"Models: {record['models_used']}",
        str(record["image_id"]),
    ]
    y = 14
    for line in lines:
        draw.text((12, y), line, fill="white", font=font)
        y += 22
    return card


def board(record: pd.Series, preview_size: int) -> Image.Image:
    image = Image.fromarray(read_rgb(Path(record["source_image_path"])))
    original = resize(image, preview_size)
    gt = resize(gt_overlay(image, str(record["source_mask_path"])), preview_size)
    evidence_models = [model for model in str(record["models_used"]).split("+") if model in {"unet", "segformer", "vmamba"}]
    evidence_paths = [str(record[f"probability_{model}_path"]) for model in evidence_models]
    probability = probability_image(evidence_paths, original.size)
    verdict = title_card(original.width, original.height, record)
    canvas = Image.new("RGB", (original.width * 4, original.height + 24), "white")
    for index, panel in enumerate((original, gt, probability, verdict)):
        canvas.paste(panel, (index * original.width, 24))
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(("Input", "Ground truth", "Evidence probability", "Automatic conclusion")):
        draw.text((index * original.width + 4, 4), title, fill="black", font=ImageFont.load_default())
    return canvas


def make_previews(all_cases: pd.DataFrame, audit: Path, preview_size: int) -> None:
    """Create full galleries for direct model comparison, errors for all policies."""
    gallery = audit / "visual_gallery"
    full = {
        "base__unet", "base__segformer", "base__vmamba",
        "learned_all3__fusion__fully_automatic",
    }
    for experiment_id, subset in all_cases.groupby("experiment_id", sort=True):
        is_full = experiment_id in full
        selected = subset if is_full else subset[subset["outcome"].isin(["missed_defect", "false_alarm"])]
        root = gallery / ("all_cases" if is_full else "errors_only") / safe_name(experiment_id)
        for position, (_, record) in enumerate(selected.iterrows(), start=1):
            destination = root / record["outcome"] / f"{safe_name(str(record['image_id']))}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            board(record, preview_size).save(destination, quality=84, optimize=True)
            if position % 100 == 0 or position == len(selected):
                print(f"[preview:{experiment_id}] {position}/{len(selected)}", flush=True)

    (gallery / "README.txt").write_text(
        "all_cases contains every Test image for U-Net, SegFormer, VMamba and the final learned 3-model fusion.\n"
        "errors_only contains every false negative and false positive for each other automatic experiment.\n"
        "Each JPG shows Input | GT overlay | probability evidence | final PASS/DEFECT conclusion.\n",
        encoding="utf-8",
    )


def write_readme(audit: Path, total_images: int, experiment_count: int) -> None:
    text = f"""# Test-case audit

Mọi policy đã được chốt bằng Validation trước khi Test được đọc. Audit này chỉ phân loại kết quả Test, không tối ưu lại threshold.

- Test images: {total_images}
- PASS/DEFECT experiments: {experiment_count}
- `01_all_experiments_all_test_images.csv`: toàn bộ ảnh Test của toàn bộ experiment.
- `per_experiment/<experiment>/`: mỗi experiment có file `all_test_images.csv`, `correct_pass.csv`, `correct_defect.csv`, `missed_defect.csv`, `false_alarm.csv`.
- `visual_gallery/all_cases/`: tất cả ảnh của ba model gốc và fusion ba model.
- `visual_gallery/errors_only/`: toàn bộ ảnh bỏ sót/báo động giả của mọi logic tự động khác.

Ý nghĩa: `missed_defect` = Defect thật nhưng kết luận PASS (FN); `false_alarm` = Good thật nhưng kết luận DEFECT (FP).
"""
    (audit / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = arguments()
    if args.preview_size < 96:
        raise ValueError("--preview-size must be at least 96")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    report = Path(args.decision_report).expanduser().resolve()
    roots = parse_key_paths(args.prediction, "--prediction")
    base_paths = parse_key_paths(args.base_test, "--base-test")
    if set(roots) != {"unet", "segformer", "vmamba"}:
        raise ValueError("Supply exactly three --prediction values: unet, segformer, vmamba")
    audit = report / "test_case_audit"
    if audit.exists():
        shutil.rmtree(audit)
    audit.mkdir(parents=True)
    (report / "tables").mkdir(parents=True, exist_ok=True)

    index = build_dataset_index(dataset_root)
    experiments = gather_experiments(index, report, base_paths)
    all_cases = pd.concat(experiments, ignore_index=True)
    all_cases = probability_paths(all_cases, roots)
    all_cases = write_case_csvs(all_cases, audit, report)
    write_readme(audit, len(index), all_cases["experiment_id"].nunique())
    if not args.no_previews:
        make_previews(all_cases, audit, args.preview_size)
    print(f"TEST CASE AUDIT READY: {audit}", flush=True)
    print(f"SUMMARY TABLE: {report / 'tables' / '03_test_case_outcome_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
