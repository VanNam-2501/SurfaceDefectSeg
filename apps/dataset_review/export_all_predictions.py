"""Export a probability map for every dataset image using a trained checkpoint.

The exporter is resumable: existing PNG files are skipped unless --overwrite is
used, and a manifest is updated periodically. The review web app discovers the
result automatically when it lives under <run_dir>/predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CODE = REPO_ROOT / "src" / "threecad_segmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-image probability maps for Data Review Studio."
    )
    parser.add_argument("--model", required=True, choices=["unet", "segformer", "vmamba"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data" / "3cad_ani"))
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--train-csv", default="")
    parser.add_argument("--val-csv", default="")
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-file", default="")
    parser.add_argument("--data-mode", choices=["auto", "patch", "resize"], default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tile-batch-size", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="0 means every image")
    parser.add_argument("--manifest-every", type=int, default=25)
    parser.add_argument("--model-code-dir", default=str(DEFAULT_MODEL_CODE))
    return parser.parse_args()


def resolve_threshold(args: argparse.Namespace, run_dir: Path) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    path = Path(args.threshold_file) if args.threshold_file else run_dir / "validation" / "selected_threshold.json"
    if path.is_file():
        return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    print(f"Warning: threshold file not found ({path}); using 0.5.")
    return 0.5


def atomic_png(array: np.ndarray, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    Image.fromarray(array).save(temporary, format="PNG", optimize=True)
    temporary.replace(target)


def atomic_csv(rows: dict[str, dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame = pd.DataFrame(rows.values()).sort_values("image_id") if rows else pd.DataFrame()
    frame.to_csv(temporary, index=False)
    temporary.replace(target)


def read_existing_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path).fillna("")
    return {str(row["image_id"]): dict(row) for row in frame.to_dict("records")}


def image_path(dataset_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else dataset_root / path


def manifest_probability_path(target: Path, run_dir: Path) -> str:
    try:
        value = target.relative_to(run_dir)
    except ValueError:
        value = target
    return str(value).replace("\\", "/")


def split_csvs(args: argparse.Namespace, dataset_root: Path) -> dict[str, Path]:
    split_dir = dataset_root / "dataset_audit" / "splits"
    return {
        "train": Path(args.train_csv) if args.train_csv else split_dir / "train.csv",
        "val": Path(args.val_csv) if args.val_csv else split_dir / "val.csv",
        "test": Path(args.test_csv) if args.test_csv else split_dir / "test.csv",
    }


def main() -> None:
    args = parse_args()
    print("Preparing prediction export...", flush=True)
    print("Loading Python packages (first launch can take a few seconds)...", flush=True)
    # Keep imports here rather than at module load so the user sees immediate
    # progress instead of an apparently idle terminal.
    global np, pd, torch, Image
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    model_code_dir = Path(args.model_code_dir).resolve()
    if not model_code_dir.is_dir():
        raise FileNotFoundError(f"Model source directory not found: {model_code_dir}")
    sys.path.insert(0, str(model_code_dir))
    from fullres_eval import predict_full_image, read_binary_mask, read_rgb, warmup_model_inference
    from model_factory import build_model

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    dataset_root = Path(args.dataset_root).resolve()
    run_dir = checkpoint.parent.parent
    output_root = Path(args.output_root).resolve() if args.output_root else run_dir / "predictions"
    threshold = resolve_threshold(args, run_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # These are local training checkpoints supplied by the user and contain
    # optimizer/config metadata in addition to tensors.
    print("Loading checkpoint into GPU memory...", flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    data_mode = config.get("data_mode", "patch") if args.data_mode == "auto" else args.data_mode
    tile_batch_size = args.tile_batch_size or int(config.get("val_tile_batch_size", 1))

    print("Building model and warming up GPU...", flush=True)
    model = build_model(args.model, pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    warmup_model_inference(
        model,
        device,
        tile_size=args.tile_size,
        tile_batch_size=tile_batch_size,
        amp=not args.no_amp,
        iterations=2,
    )

    print(f"Model: {args.model} | device: {device} | mode: {data_mode} | threshold: {threshold:.4f}")
    print(f"Output: {output_root}")
    csvs = split_csvs(args, dataset_root)
    for split in args.splits:
        csv_path = csvs[split]
        if not csv_path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path).fillna("")
        if args.max_images > 0:
            frame = frame.head(args.max_images)
        probability_dir = output_root / split / "probability"
        manifest_path = output_root / split / "manifest.csv"
        manifest = read_existing_manifest(manifest_path)
        started = time.perf_counter()
        processed_now = 0

        for index, row in enumerate(frame.to_dict("records"), start=1):
            current_id = str(row.get("image_id", "")).strip()
            if not current_id:
                raise ValueError(f"Missing image_id in {csv_path} row {index}")
            target = probability_dir / f"{current_id}.png"
            if target.is_file() and not args.overwrite:
                if current_id not in manifest:
                    with Image.open(target) as saved:
                        quantized = np.asarray(saved, dtype=np.uint8)
                    restored = quantized.astype(np.float32) / 255.0
                    manifest[current_id] = {
                        "image_id": current_id,
                        "split": split,
                        "image_score": float(restored.max(initial=0.0)),
                        "image_pred": int(np.any(restored >= threshold)),
                        "threshold": threshold,
                        "predicted_positive_pixels": int(np.count_nonzero(restored >= threshold)),
                        "probability_path": manifest_probability_path(target, run_dir),
                        "model_seconds": "",
                        "e2e_seconds": "",
                    }
                continue

            source = image_path(dataset_root, row.get("image_path", ""))
            image = read_rgb(source)
            probability, timing = predict_full_image(
                model=model,
                image=image,
                device=device,
                data_mode=data_mode,
                tile_size=args.tile_size,
                stride=args.stride,
                tile_batch_size=tile_batch_size,
                amp=not args.no_amp,
            )
            quantized = np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)
            atomic_png(quantized, target)
            binary = probability >= threshold
            label = int(row.get("label", 0))
            mask_value = str(row.get("mask_path", "")).strip()
            if mask_value:
                ground_truth = read_binary_mask(image_path(dataset_root, mask_value)).astype(bool)
            else:
                ground_truth = np.zeros(binary.shape, dtype=bool)
            if ground_truth.shape != binary.shape:
                raise ValueError(
                    f"Mask shape {ground_truth.shape} differs from prediction {binary.shape} for {current_id}"
                )
            pixel_tp = int(np.count_nonzero(binary & ground_truth))
            pixel_fp = int(np.count_nonzero(binary & ~ground_truth))
            pixel_fn = int(np.count_nonzero(~binary & ground_truth))
            dice_denominator = 2 * pixel_tp + pixel_fp + pixel_fn
            manifest[current_id] = {
                "image_id": current_id,
                "split": split,
                "image_score": float(probability.max(initial=0.0)),
                "image_pred": int(np.any(binary)),
                "threshold": threshold,
                "predicted_positive_pixels": int(np.count_nonzero(binary)),
                "pixel_tp": pixel_tp,
                "pixel_fp": pixel_fp,
                "pixel_fn": pixel_fn,
                "positive_dice": (
                    float(2 * pixel_tp / dice_denominator)
                    if label and dice_denominator
                    else (1.0 if label and not dice_denominator else "")
                ),
                "probability_path": manifest_probability_path(target, run_dir),
                "model_seconds": float(timing["model_seconds"]),
                "e2e_seconds": float(timing["e2e_seconds"]),
            }
            processed_now += 1
            if processed_now % max(1, args.manifest_every) == 0:
                atomic_csv(manifest, manifest_path)
                elapsed = time.perf_counter() - started
                rate = processed_now / max(elapsed, 1e-6)
                remaining = max(0, len(frame) - index)
                print(
                    f"[{split}] {index}/{len(frame)} | {rate:.2f} image/s | "
                    f"ETA {remaining / max(rate, 1e-6) / 60:.1f} min"
                )

        atomic_csv(manifest, manifest_path)
        print(f"[{split}] complete: {len(frame)} images; newly exported: {processed_now}")

    metadata = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "splits": args.splits,
        "threshold": threshold,
        "data_mode": data_mode,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "tile_batch_size": tile_batch_size,
        "probability_encoding": "uint8_png_0_255",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("All requested predictions are ready. Restart Data Review Studio to load them.")


if __name__ == "__main__":
    main()
