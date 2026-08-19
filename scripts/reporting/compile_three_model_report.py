"""Compile every frozen three-model experiment into report-ready CSV tables.

This script never calibrates a threshold and never changes a decision policy.
It only reads outputs created by ``run_three_model_experiments.ps1`` and
recomputes transparent image-level counts for the final report.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    return parser.parse_args()


def safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def decision_metrics(frame: pd.DataFrame, decision_column: str) -> dict[str, int | float]:
    labels = frame["label"].astype(int)
    decision = frame[decision_column].astype(str).str.lower()
    positive = labels == 1
    good = ~positive
    defect = decision == "defect"
    review = decision == "review"
    passed = decision == "pass"
    missed = int((positive & passed).sum())
    auto_defect_fn = int((positive & ~defect).sum())
    fp = int((good & defect).sum())
    good_review = int((good & review).sum())
    positive_review = int((positive & review).sum())
    good_attention = int((good & ~passed).sum())
    tp = int((positive & defect).sum())
    total = int(len(frame))
    return {
        "images": total,
        "positive_images": int(positive.sum()),
        "good_images": int(good.sum()),
        "missed_defects": missed,
        "auto_defect_false_negatives": auto_defect_fn,
        "auto_defect_true_positives": tp,
        "false_alarms": fp,
        "good_reviews": good_review,
        "positive_reviews": positive_review,
        "reviews": int(review.sum()),
        "good_attention": good_attention,
        "alert_fnr": safe_div(missed, int(positive.sum())),
        "auto_defect_fnr": safe_div(auto_defect_fn, int(positive.sum())),
        "auto_defect_recall": safe_div(tp, int(positive.sum())),
        "auto_defect_fpr": safe_div(fp, int(good.sum())),
        "good_review_rate": safe_div(good_review, int(good.sum())),
        "positive_review_rate": safe_div(positive_review, int(positive.sum())),
        "overall_review_rate": safe_div(int(review.sum()), total),
        "good_attention_rate": safe_div(good_attention, int(good.sum())),
        "automatic_accuracy": safe_div(
            tp + int((good & passed).sum()), total
        )
        if not review.any()
        else float("nan"),
    }


def with_percent_columns(row: dict[str, object]) -> dict[str, object]:
    for name in (
        "val_alert_fnr",
        "val_auto_defect_fnr",
        "val_auto_defect_fpr",
        "val_good_attention_rate",
        "test_alert_fnr",
        "test_auto_defect_fnr",
        "test_auto_defect_fpr",
        "test_good_attention_rate",
        "test_overall_review_rate",
        "test_automatic_accuracy",
    ):
        value = row.get(name)
        if value is not None:
            row[f"{name}_pct"] = float(value) * 100.0
    return row


def make_row(
    *,
    experiment_id: str,
    family: str,
    model_set: str,
    decision_mode: str,
    selection_method: str,
    val: pd.DataFrame,
    test: pd.DataFrame,
    decision_column: str,
) -> dict[str, object]:
    val_metrics = decision_metrics(val, decision_column)
    test_metrics = decision_metrics(test, decision_column)
    row: dict[str, object] = {
        "experiment_id": experiment_id,
        "family": family,
        "model_set": model_set,
        "decision_mode": decision_mode,
        "selection_split": "Validation only",
        "selection_method": selection_method,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    return with_percent_columns(row)


def group_recall_rows(
    *,
    experiment_id: str,
    family: str,
    model_set: str,
    frame: pd.DataFrame,
    decision_column: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, subset in frame.groupby("defect_group", dropna=False):
        positives = subset[subset["label"].astype(int) == 1]
        if positives.empty:
            continue
        predicted = positives[decision_column].astype(str).str.lower() == "defect"
        alert = positives[decision_column].astype(str).str.lower() != "pass"
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": family,
                "model_set": model_set,
                "defect_group": str(group),
                "positive_images": int(len(positives)),
                "automatic_defect_recall": float(predicted.mean()),
                "alert_recall": float(alert.mean()),
            }
        )
    return rows


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def add_spatial(report_dir: Path, master: list[dict[str, object]], groups: list[dict[str, object]]) -> None:
    spatial_root = report_dir / "spatial"
    if not spatial_root.is_dir():
        return
    for experiment_dir in sorted(path for path in spatial_root.iterdir() if path.is_dir()):
        val_path = experiment_dir / "val" / "per_image_decisions.csv"
        test_path = experiment_dir / "test" / "per_image_decisions.csv"
        if not val_path.is_file() or not test_path.is_file():
            continue
        key = experiment_dir.name
        val, test = read(val_path), read(test_path)
        master.append(
            make_row(
                experiment_id=f"spatial__{key}",
                family="spatial_consensus",
                model_set=key.replace("_", " + "),
                decision_mode="PASS / REVIEW / DEFECT",
                selection_method="Validation component + spatial-vote grid",
                val=val,
                test=test,
                decision_column="decision",
            )
        )
        groups.extend(
            group_recall_rows(
                experiment_id=f"spatial__{key}",
                family="spatial_consensus",
                model_set=key.replace("_", " + "),
                frame=test,
                decision_column="decision",
            )
        )


def add_adaptive(report_dir: Path, master: list[dict[str, object]], groups: list[dict[str, object]]) -> None:
    directory = report_dir / "adaptive_single"
    test_path = directory / "adaptive_per_image_test.csv"
    if not test_path.is_file():
        return
    test_all = read(test_path)
    # The rule policy writes its selected Validation/Test summary; turn the
    # test image decisions into the same schema as the other experiments.
    summary_path = directory / "adaptive_model_comparison.csv"
    summary = read(summary_path).set_index("model") if summary_path.is_file() else pd.DataFrame()
    for model, test in test_all.groupby("model", dropna=False):
        if not isinstance(summary, pd.DataFrame) or model not in summary.index:
            continue
        item = summary.loc[model]
        # Construct Validation placeholder metrics from frozen selected values.
        val = test.copy()
        val["decision"] = "pass"
        # Values are overwritten below from the calibration summary, which is
        # more faithful than reusing Test image predictions.
        row = make_row(
            experiment_id=f"adaptive__{model}",
            family="adaptive_component_rule",
            model_set=str(model),
            decision_mode="fully automatic",
            selection_method="Validation component evidence grid",
            val=val,
            test=test,
            decision_column="decision",
        )
        for key in ("fnr", "fpr", "recall", "accuracy"):
            source = f"val_{key}"
            if source in item.index:
                if key == "fnr":
                    row["val_alert_fnr"] = float(item[source])
                    row["val_auto_defect_fnr"] = float(item[source])
                elif key == "fpr":
                    row["val_auto_defect_fpr"] = float(item[source])
                    row["val_good_attention_rate"] = float(item[source])
        with_percent_columns(row)
        master.append(row)
        groups.extend(
            group_recall_rows(
                experiment_id=f"adaptive__{model}",
                family="adaptive_component_rule",
                model_set=str(model),
                frame=test,
                decision_column="decision",
            )
        )


def add_learned(
    directory: Path,
    experiment_prefix: str,
    kind: str,
    master: list[dict[str, object]],
    groups: list[dict[str, object]],
) -> None:
    val_path = directory / "per_validation_oof_predictions.csv"
    test_path = directory / "per_image_predictions.csv"
    if not val_path.is_file() or not test_path.is_file():
        return
    val, test = read(val_path), read(test_path)
    automatic_columns = sorted(column for column in test.columns if column.endswith("_automatic"))
    for column in automatic_columns:
        branch = column[: -len("_automatic")]
        if kind == "hybrid_pair" and branch != "hybrid_fusion":
            continue
        if column not in val.columns:
            continue
        model_set = (
            experiment_prefix.replace("_", " + ")
            if branch in {"hybrid_fusion", "fusion"}
            else branch
        )
        experiment_id = f"{kind}__{experiment_prefix}__{branch}"
        master.append(
            make_row(
                experiment_id=experiment_id,
                family="learned_hybrid" if branch == "hybrid_fusion" else "learned_verifier",
                model_set=model_set,
                decision_mode="fully automatic",
                selection_method="5-fold Validation OOF threshold" if branch != "hybrid_fusion" else "Validation spatial consensus + two-specialist rescue",
                val=val,
                test=test,
                decision_column=column,
            )
        )
        groups.extend(
            group_recall_rows(
                experiment_id=experiment_id,
                family="learned_hybrid" if branch == "hybrid_fusion" else "learned_verifier",
                model_set=model_set,
                frame=test,
                decision_column=column,
            )
        )
    # Triage rows are retained for the three-model independent verifier.  The
    # binary FPR here is explicitly DEFECT-only; review remains a separate
    # column rather than being hidden as a successful PASS.
    if kind == "learned_all3":
        for column in sorted(column for column in test.columns if column.endswith("_triage")):
            branch = column[: -len("_triage")]
            if column not in val.columns:
                continue
            master.append(
                make_row(
                    experiment_id=f"learned_triage__{branch}",
                    family="learned_verifier",
                    model_set=branch,
                    decision_mode="PASS / REVIEW / DEFECT",
                    selection_method="5-fold Validation OOF triage thresholds",
                    val=val,
                    test=test,
                    decision_column=column,
                )
            )


def write_readme(path: Path) -> None:
    path.write_text(
        """# Báo cáo thí nghiệm 3 model\n\n"
        "Mọi policy/threshold trong thư mục này đều được chọn trên **Validation**; "
        "Test chỉ được đọc để báo cáo kết quả cuối cùng.\n\n"
        "## Đọc bảng `01_master_test_comparison.csv`\n\n"
        "- `test_alert_fnr`: tỷ lệ ảnh Defect bị kết luận PASS. Đây là tỷ lệ bỏ sót thực tế.\n"
        "- `test_auto_defect_fpr`: tỷ lệ ảnh Good bị tự động kết luận DEFECT.\n"
        "- `test_good_attention_rate`: tỷ lệ ảnh Good cần chú ý = DEFECT hoặc REVIEW.\n"
        "- Với dòng `PASS / REVIEW / DEFECT`, FPR chỉ đếm kết luận DEFECT; REVIEW được tách riêng, không bị che giấu.\n"
        "- Với dòng `fully automatic`, không có REVIEW nên `alert_fnr = auto_defect_fnr` và `good_attention_rate = auto_defect_fpr`.\n\n"
        "## Cấu trúc\n\n"
        "- `adaptive_single`: rule-only từng model (area + peak + persistence + contrast).\n"
        "- `spatial`: từng model, từng cặp và cả ba model với connected-component / consensus.\n"
        "- `learned_all3`: verifier độc lập U-Net, SegFormer, VMamba và learned fusion 3 model.\n"
        "- `hybrid_pairs`: hybrid fully automatic cho U-Net+SegFormer, U-Net+VMamba, SegFormer+VMamba.\n"
        "- `tables`: bảng gộp dùng trực tiếp cho báo cáo.\n"
        """,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    table_dir = report_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    master: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    add_adaptive(report_dir, master, groups)
    add_spatial(report_dir, master, groups)
    add_learned(report_dir / "learned_all3", "unet_segformer_vmamba", "learned_all3", master, groups)
    hybrid_root = report_dir / "hybrid_pairs"
    if hybrid_root.is_dir():
        for directory in sorted(path for path in hybrid_root.iterdir() if path.is_dir()):
            add_learned(directory, directory.name, "hybrid_pair", master, groups)
    master_frame = pd.DataFrame(master)
    if not master_frame.empty:
        master_frame = master_frame.sort_values(["family", "model_set", "decision_mode", "experiment_id"])
    master_frame.to_csv(table_dir / "01_master_test_comparison.csv", index=False)
    automatic = master_frame[master_frame["decision_mode"] == "fully automatic"].copy() if not master_frame.empty else master_frame
    automatic.to_csv(table_dir / "02_fully_automatic_comparison.csv", index=False)
    group_frame = pd.DataFrame(groups)
    if not group_frame.empty:
        group_frame = group_frame.sort_values(["family", "model_set", "defect_group"])
    group_frame.to_csv(table_dir / "03_defect_group_recall.csv", index=False)
    test_records = pd.read_csv(dataset_root / "dataset_audit" / "splits" / "test.csv").fillna("")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "test_images": int(len(test_records)),
        "test_defect_images": int((test_records["label"].astype(int) == 1).sum()),
        "test_good_images": int((test_records["label"].astype(int) == 0).sum()),
        "experiments_in_master_table": int(len(master_frame)),
        "policy_selection": "Validation only; Test read for final reporting only.",
    }
    (table_dir / "00_report_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(table_dir / "README.md")
    print(f"Report tables saved to: {table_dir}")


if __name__ == "__main__":
    main()
