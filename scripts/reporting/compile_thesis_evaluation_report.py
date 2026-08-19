"""Compile the selected thesis experiments from frozen Val/Test evaluations.

Scope deliberately excludes E1, E6, E9, E10 and E11.  It has no training
step and does not change any model threshold.  The compiler only aggregates
the artifacts created by ``evaluate_model.py`` after Validation has frozen the
threshold and Test has been evaluated once.

Included thesis sections:
E0 technical preflight, E2 architecture comparison, E3 defect-size analysis,
E4 defect-group analysis, E5 multi-region analysis, E7 threshold sensitivity,
and E8 qualitative error manifest.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = ("unet", "segformer", "vmamba")
DISPLAY = {"unet": "U-Net/ResNet18", "segformer": "SegFormer-B0", "vmamba": "VMamba-T"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unet-run", required=True)
    parser.add_argument("--segformer-run", required=True)
    parser.add_argument("--vmamba-run", required=True)
    parser.add_argument("--protocol-check", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_paths(args: argparse.Namespace) -> dict[str, Path]:
    result = {
        "unet": Path(args.unet_run).expanduser().resolve(),
        "segformer": Path(args.segformer_run).expanduser().resolve(),
        "vmamba": Path(args.vmamba_run).expanduser().resolve(),
    }
    for model, root in result.items():
        for required in ("validation/selected_threshold.json", "test/main_metrics.json"):
            if not (root / required).is_file():
                raise FileNotFoundError(f"{model}: missing {root / required}")
    return result


def copy_with_model(runs: dict[str, Path], filename: str, destination: Path) -> None:
    frames: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        source = runs[model] / "test" / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        frame.insert(0, "model", DISPLAY[model])
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(destination, index=False)


def qualitative_manifest(runs: dict[str, Path], destination: Path) -> None:
    rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        directory = runs[model] / "test" / "qualitative"
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.png")):
            if path.name.startswith("high_fp_"):
                category = "false_alarm_candidate"
            elif path.name.startswith("worst_defect_"):
                category = "lowest_dice_defect"
            else:
                category = "other"
            rows.append({"model": DISPLAY[model], "category": category, "image_file": path.name, "path": str(path)})
    pd.DataFrame(rows).to_csv(destination, index=False)


def main() -> None:
    args = arguments()
    runs = run_paths(args)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)

    # E2: only metrics measured from frozen Test evaluation.  E6 train cost is
    # intentionally not added because the selected scope defers it.
    e2_rows: list[dict[str, object]] = []
    e7_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        root = runs[model]
        test = read_json(root / "test" / "main_metrics.json")
        threshold = read_json(root / "validation" / "selected_threshold.json")
        e2_rows.append({"model": DISPLAY[model], **test})
        e7_rows.append({
            "model": DISPLAY[model],
            "threshold": threshold.get("threshold"),
            "fnr_limit_validation": threshold.get("fnr_limit"),
            "selection_status": threshold.get("selection_status"),
            "selection_rule": threshold.get("rule"),
            "threshold_scan_path": str(root / "validation" / "threshold_scan.csv"),
        })
    e2 = pd.DataFrame(e2_rows)
    e2.to_csv(tables / "01_e2_architecture_comparison.csv", index=False)
    pd.DataFrame(e7_rows).to_csv(tables / "05_e7_thresholds_from_validation.csv", index=False)

    # E3/E4/E5 retain all per-bin/per-group rows; no aggregate hides a small
    # sample.  The source evaluator marks low-count groups explicitly.
    copy_with_model(runs, "defect_size_metrics.csv", tables / "02_e3_defect_size.csv")
    copy_with_model(runs, "defect_group_metrics.csv", tables / "03_e4_defect_group.csv")
    copy_with_model(runs, "multi_region_metrics.csv", tables / "04_e5_multi_region.csv")
    qualitative_manifest(runs, tables / "06_e8_qualitative_manifest.csv")

    plot_columns = ["image_auprc", "image_recall", "positive_dice", "region_recall_any_overlap"]
    plot = e2.set_index("model")[[column for column in plot_columns if column in e2]]
    axis = plot.plot(kind="bar", figsize=(11, 6))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("E2 — Architecture comparison (frozen Test evaluation)")
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures / "e2_architecture_comparison.png", dpi=170)
    plt.close()

    protocol = {}
    if args.protocol_check:
        path = Path(args.protocol_check).expanduser().resolve()
        protocol = read_json(path) if path.is_file() else {"status": "missing", "path": str(path)}
    scope = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodological_rule": "Thresholds selected on Validation only; Test used once for final reporting.",
        "included": ["E0", "E2", "E3", "E4", "E5", "E7", "E8"],
        "excluded_by_project_scope": ["E1", "E6", "E9", "E10", "E11"],
        "e6_note": "Training efficiency will be added later from complete VMamba training artifacts.",
        "protocol_preflight": protocol,
        "runs": {model: str(root) for model, root in runs.items()},
    }
    (output / "README.md").write_text(
        "# Báo cáo thí nghiệm đã chọn\n\n"
        "Bao gồm E0, E2, E3, E4, E5, E7, E8. E1/E6/E9/E10/E11 không thuộc scope lần chạy này. "
        "Các ngưỡng được chọn trên Validation; Test chỉ dùng để báo cáo.\n",
        encoding="utf-8",
    )
    (output / "scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"THESIS EVALUATION REPORT READY: {output}")


if __name__ == "__main__":
    main()
