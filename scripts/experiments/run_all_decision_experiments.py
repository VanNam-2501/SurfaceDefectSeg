"""Run every fair three-model decision experiment from cached probability maps.

All policy parameters are calibrated on Validation only.  Test is read only
after a policy is frozen, and is used solely for the final report.

Outputs include:
* single-model adaptive component policies;
* spatial PASS/REVIEW/DEFECT policies for each model, each pair, and all 3;
* report-ready consolidated CSV tables.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_prediction_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--prediction must be model=probability_root, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        path = Path(raw_path.strip()).expanduser().resolve()
        if name not in {"unet", "segformer", "vmamba"}:
            raise ValueError(f"Unsupported model: {name}")
        result[name] = path
    if set(result) != {"unet", "segformer", "vmamba"}:
        raise ValueError("Supply exactly unet, segformer and vmamba prediction roots")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prediction", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-fnr", type=float, default=0.02)
    parser.add_argument("--rebuild-caches", action="store_true")
    parser.add_argument(
        "--automatic-only",
        action="store_true",
        help="Report only final PASS/DEFECT branches; spatial REVIEW is internal evidence for rule-based calibration and is not a final output.",
    )
    return parser.parse_args()


def run(project: Path, title: str, values: list[str]) -> None:
    command = [sys.executable, *values]
    print(f"\n===== {title} =====", flush=True)
    print(" ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=project, check=True)


def cache_count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.is_dir() else 0


def check_probability_caches(dataset: Path, predictions: dict[str, Path]) -> None:
    counts = {
        split: len(pd.read_csv(dataset / "dataset_audit" / "splits" / f"{split}.csv"))
        for split in ("val", "test")
    }
    for model, root in predictions.items():
        for split, expected in counts.items():
            actual = cache_count(root / split / "probability")
            if actual < expected:
                raise FileNotFoundError(
                    f"{model}/{split}: only {actual}/{expected} probability maps under {root}. "
                    "Run export_probability_cache.py first."
                )
            print(f"[{model}/{split}] {actual}/{expected} probability maps", flush=True)


def add_prediction_args(target: list[str], predictions: dict[str, Path], names: list[str]) -> None:
    for name in names:
        target.extend(["--prediction", f"{name}={predictions[name]}"])


def load_spatial_metrics(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "test_alert_fnr": data.get("alert_fnr"),
        "test_auto_defect_fnr": data.get("defect_fnr"),
        "test_auto_defect_fpr": data.get("defect_fpr"),
        "test_good_attention_rate": (
            float(data.get("defect_fpr", 0.0)) + float(data.get("good_review_rate", 0.0))
        ),
        "test_review_rate": data.get("overall_review_rate"),
        "test_accuracy": "",  # triage is not a single automatic binary decision
        "test_positive_dice": data.get("positive_dice"),
    }


def percent_columns(row: dict[str, object]) -> dict[str, object]:
    for key in (
        "test_alert_fnr",
        "test_auto_defect_fnr",
        "test_auto_defect_fpr",
        "test_good_attention_rate",
        "test_review_rate",
        "test_accuracy",
    ):
        value = row.get(key)
        if value not in (None, ""):
            row[f"{key}_pct"] = 100.0 * float(value)
    return row


def build_report(output: Path, automatic_only: bool = False) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    adaptive_path = output / "adaptive_single" / "adaptive_model_comparison.csv"
    if adaptive_path.is_file():
        for item in pd.read_csv(adaptive_path).fillna("").to_dict("records"):
            rows.append(percent_columns({
                "experiment_id": f"adaptive__{item['model']}",
                "family": "adaptive_component_rule",
                "model_set": item["model"],
                "decision_mode": "fully automatic",
                "selection_split": "Validation only",
                "test_alert_fnr": item.get("test_fnr"),
                "test_auto_defect_fnr": item.get("test_fnr"),
                "test_auto_defect_fpr": item.get("test_fpr"),
                "test_good_attention_rate": item.get("test_fpr"),
                "test_review_rate": 0.0,
                "test_accuracy": item.get("test_accuracy"),
                "policy": json.dumps({
                    key: item.get(key) for key in (
                        "low_threshold", "min_area_px", "peak_threshold",
                        "min_persistent_area_px", "min_local_contrast",
                    )
                }),
            }))

    spatial_root = output / "spatial"
    if spatial_root.is_dir() and not automatic_only:
        for directory in sorted(path for path in spatial_root.iterdir() if path.is_dir()):
            metrics_path = directory / "test" / "decision_metrics.json"
            policy_path = directory / "policy" / "decision_policy.json"
            if not metrics_path.is_file():
                continue
            row = {
                "experiment_id": f"spatial__{directory.name}",
                "family": "spatial_consensus",
                "model_set": directory.name.replace("_", " + "),
                "decision_mode": "PASS / REVIEW / DEFECT",
                "selection_split": "Validation only",
                **load_spatial_metrics(metrics_path),
                "policy": policy_path.read_text(encoding="utf-8") if policy_path.is_file() else "",
            }
            rows.append(percent_columns(row))

    master = pd.DataFrame(rows)
    if not master.empty:
        master = master.sort_values(["family", "model_set", "decision_mode", "experiment_id"])
    master.to_csv(tables / "01_master_test_comparison.csv", index=False)
    automatic = master[master["decision_mode"] == "fully_automatic"].copy() if "decision_mode" in master else master.copy()
    if not master.empty:
        automatic = master[master["decision_mode"].isin(["fully automatic", "fully_automatic"])].copy()
    automatic.to_csv(tables / "02_fully_automatic_comparison.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_protocol": "All thresholds and policies are frozen from Validation only. Test is report-only.",
        "metric_note": (
            "Final table is PASS/DEFECT only. Spatial REVIEW, when used internally, is never emitted as a final conclusion."
            if automatic_only
            else "For PASS/REVIEW/DEFECT rows, auto_defect_fpr counts only Good→DEFECT; good_attention_rate counts Good→DEFECT or REVIEW."
        ),
        "experiment_count": int(len(master)),
    }
    (tables / "00_readme.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nREPORT TABLES: {tables}", flush=True)
    if not master.empty:
        print(master[["experiment_id", "test_alert_fnr", "test_auto_defect_fpr", "test_good_attention_rate"]].to_string(index=False), flush=True)


def main() -> None:
    args = arguments()
    if not 0.0 <= args.max_fnr < 1.0:
        raise ValueError("--max-fnr must be in [0, 1)")
    experiment_root = Path(__file__).resolve().parent
    repo_root = experiment_root.parents[1]
    source_root = repo_root / "src" / "threecad_segmentation"
    dataset = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = parse_prediction_specs(args.prediction)
    check_probability_caches(dataset, predictions)

    # 1) Each standalone model: connected-component evidence only.
    adaptive_args = [
        str(source_root / "adaptive_component_policy.py"), "--dataset-root", str(dataset),
        "--output-dir", str(output / "adaptive_single"), "--max-fnr", str(args.max_fnr),
    ]
    add_prediction_args(adaptive_args, predictions, ["unet", "segformer", "vmamba"])
    if args.rebuild_caches:
        adaptive_args.append("--rebuild-cache")
    run(repo_root, "ADAPTIVE COMPONENTS · SINGLE MODELS", adaptive_args)

    # 2) Spatial policies for 3 individual models, every pair, and all 3.
    experiments: dict[str, list[str]] = {
        "unet": ["unet"], "segformer": ["segformer"], "vmamba": ["vmamba"],
        "unet_segformer": ["unet", "segformer"],
        "unet_vmamba": ["unet", "vmamba"],
        "segformer_vmamba": ["segformer", "vmamba"],
        "unet_segformer_vmamba": ["unet", "segformer", "vmamba"],
    }
    for key, names in experiments.items():
        root = output / "spatial" / key
        policy_dir = root / "policy"
        calibrate_args = [
            str(source_root / "calibrate_decision_policy.py"), "--dataset-root", str(dataset),
            "--output-dir", str(policy_dir), "--fnr-limit", str(args.max_fnr),
            "--min-defect-recall", "0.80", "--review-cost", "0.25",
        ]
        add_prediction_args(calibrate_args, predictions, names)
        run(repo_root, f"SPATIAL CALIBRATION · {key}", calibrate_args)
        policy = policy_dir / "decision_policy.json"
        for split in ("val", "test"):
            evaluate_args = [
                str(experiment_root / "evaluate_decision_policy.py"), "--dataset-root", str(dataset), "--split", split,
                "--policy", str(policy), "--output-dir", str(root / split),
            ]
            add_prediction_args(evaluate_args, predictions, names)
            run(repo_root, f"SPATIAL {split.upper()} · {key}", evaluate_args)

    build_report(output, automatic_only=args.automatic_only)
    print(f"\nALL DECISION EXPERIMENTS COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
