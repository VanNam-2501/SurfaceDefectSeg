"""Aggregate E1-E6 result files after all models have been evaluated on Test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path, model_label: str):
    main = load_json(run_dir / "test" / "main_metrics.json")
    train = load_json(run_dir / "training_summary.json")
    info = load_json(run_dir / "model_info.json")
    row = {"model": model_label, **main}
    row.update(
        {
            "total_parameters": info["total_parameters"],
            "trainable_parameters": info["trainable_parameters"],
            "best_epoch": train["best_epoch"],
            "training_wall_seconds": train["total_train_run_wall_seconds"],
            "training_only_seconds": train["total_train_only_seconds"],
            "validation_during_training_seconds": train["total_validation_seconds"],
            "train_peak_vram_gb": train["max_peak_vram_gb"],
        }
    )
    return row


def append_model_csv(frames, run_dir: Path, filename: str, model: str):
    path = run_dir / "test" / filename
    if path.exists():
        df = pd.read_csv(path)
        df.insert(0, "model", model)
        frames.append(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default="results/unet_r18/main_seed42")
    p.add_argument("--segformer", default="results/segformer_b0/main_seed42")
    p.add_argument("--vmamba", default="results/vmamba_t_s2l5/main_seed42")
    p.add_argument("--unet-resize", default="results/unet_r18/e1_resize_seed42")
    p.add_argument("--output-dir", default="results/comparison")
    args = p.parse_args()

    runs = [
        (Path(args.unet), "U-Net/ResNet18"),
        (Path(args.segformer), "SegFormer-B0"),
        (Path(args.vmamba), "VMamba-T"),
    ]
    for run_dir, name in runs:
        if not (run_dir / "test" / "main_metrics.json").exists():
            raise FileNotFoundError(f"Missing Test results for {name}: {run_dir}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_rows = [load_run(path, name) for path, name in runs]
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(out / "main_metrics.csv", index=False)

    # E6 efficiency table.
    efficiency_cols = [
        "model",
        "total_parameters",
        "trainable_parameters",
        "best_epoch",
        "training_wall_seconds",
        "training_only_seconds",
        "train_peak_vram_gb",
        "eval_peak_vram_gb",
        "tiles_per_second",
        "full_image_latency_mean_seconds",
        "full_image_latency_p50_seconds",
        "full_image_latency_p95_seconds",
        "evaluation_total_seconds",
    ]
    main_df[[c for c in efficiency_cols if c in main_df.columns]].to_csv(
        out / "efficiency.csv", index=False
    )

    for filename, outname in [
        ("defect_size_metrics.csv", "defect_size_metrics.csv"),
        ("defect_group_metrics.csv", "defect_group_metrics.csv"),
        ("multi_region_metrics.csv", "multi_region_metrics.csv"),
    ]:
        frames = []
        for path, name in runs:
            append_model_csv(frames, path, filename, name)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / outname, index=False)

    # Architecture comparison plot (E2).
    metrics = ["image_auprc", "image_recall", "positive_dice", "region_recall_any_overlap"]
    plot_df = main_df.set_index("model")[[m for m in metrics if m in main_df.columns]]
    ax = plot_df.plot(kind="bar", figsize=(11, 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("E2 — Architecture comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "architecture_comparison.png", dpi=170)
    plt.close()

    # E6 plot.
    eff_plot = main_df.set_index("model")[["full_image_latency_mean_seconds", "tiles_per_second"]]
    ax = eff_plot.plot(kind="bar", figsize=(10, 5), secondary_y="tiles_per_second")
    ax.set_title("E6 — Inference efficiency")
    ax.set_ylabel("Full-image latency (s)")
    plt.tight_layout()
    plt.savefig(out / "efficiency_comparison.png", dpi=170)
    plt.close()

    # E1 preprocessing: U-Net patch vs resize, when the optional run exists.
    resize_run = Path(args.unet_resize)
    if (resize_run / "test" / "main_metrics.json").exists():
        patch_run = Path(args.unet)
        e1_rows = []
        for rr, label in [
            (patch_run, "Native patch + sliding window"),
            (resize_run, "Full-image resize + pad"),
        ]:
            row = load_run(rr, label)
            size_path = rr / "test" / "defect_size_metrics.csv"
            if size_path.exists():
                size_df = pd.read_csv(size_path).set_index("size_bin")
                for size_name in ["Tiny", "Small"]:
                    if size_name in size_df.index:
                        row[f"{size_name.lower()}_region_recall"] = float(size_df.loc[size_name, "region_recall"])
            e1_rows.append(row)
        pd.DataFrame(e1_rows).to_csv(out / "e1_preprocessing.csv", index=False)

    print(main_df.to_string(index=False))
    print(f"\nSaved comparison outputs to: {out}")


if __name__ == "__main__":
    main()
