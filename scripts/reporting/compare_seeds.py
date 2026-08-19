"""Aggregate optional E9 multi-seed runs into mean ± standard deviation tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    p.add_argument("--output", default="results/comparison/seed_summary.csv")
    args = p.parse_args()

    model_dirs = {
        "U-Net/ResNet18": "unet_r18",
        "SegFormer-B0": "segformer_b0",
        "VMamba-T": "vmamba_t_s2l5",
    }
    rows = []
    root = Path(args.results_root)
    for model_label, model_dir in model_dirs.items():
        for seed in args.seeds:
            path = root / model_dir / f"main_seed{seed}" / "test" / "main_metrics.json"
            if not path.exists():
                print(f"SKIP missing: {path}")
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            row["model"] = model_label
            row["seed"] = seed
            rows.append(row)

    if not rows:
        raise FileNotFoundError("No seed Test metrics found")
    df = pd.DataFrame(rows)
    metric_cols = [
        "image_auroc", "image_auprc", "image_recall", "image_fnr", "image_fpr",
        "positive_dice", "positive_iou", "pixel_auprc_hist", "pixel_recall",
        "region_recall_any_overlap", "full_image_latency_mean_seconds",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_name("seed_runs_raw.csv"), index=False)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
