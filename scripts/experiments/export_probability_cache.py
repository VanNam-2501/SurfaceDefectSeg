"""Export resumable full-resolution probability maps from one checkpoint.

Designed for Kaggle after VMamba training, but it also works for U-Net and
SegFormer.  It writes the portable cache layout consumed by
``run_three_model_experiments.ps1``:

    <output-root>/train/probability/<image_id>.png
    <output-root>/val/probability/<image_id>.png
    <output-root>/test/probability/<image_id>.png

The maps are 8-bit probabilities (value / 255), so only the maps need to be
downloaded to Windows. The trained checkpoint remains untouched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("unet", "segformer", "vmamba"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("val", "test"))
    parser.add_argument("--data-mode", choices=("auto", "patch", "resize"), default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tile-batch-size", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def atomic_png(image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    image.save(temporary, format="PNG", optimize=True)
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    project_root = Path(__file__).resolve().parents[2] / "src" / "threecad_segmentation"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from fullres_eval import predict_full_image, read_rgb, resolve_path, row_id, warmup_model_inference
    from model_factory import build_model

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print("Loading checkpoint...", flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    data_mode = config.get("data_mode", "patch") if args.data_mode == "auto" else args.data_mode
    tile_batch_size = args.tile_batch_size or int(config.get("val_tile_batch_size", 1))
    print(f"Building {args.model} on {device}...", flush=True)
    model = build_model(args.model, pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    warmup_model_inference(
        model, device, tile_size=args.tile_size, tile_batch_size=tile_batch_size,
        amp=not args.no_amp, iterations=2,
    )
    print(
        f"model={args.model} | mode={data_mode} | tile={args.tile_size}/{args.stride} "
        f"| tile_batch={tile_batch_size} | output={output_root}",
        flush=True,
    )
    for split in args.splits:
        csv_path = dataset_root / "dataset_audit" / "splits" / f"{split}.csv"
        records = pd.read_csv(csv_path).fillna("")
        if args.max_images:
            records = records.head(args.max_images).copy()
        processed = skipped = 0
        started = time.perf_counter()
        for position, (_, row) in enumerate(records.iterrows(), start=1):
            image_id = row_id(row)
            target = output_root / split / "probability" / f"{image_id}.png"
            if target.is_file() and not args.overwrite:
                skipped += 1
                continue
            image = read_rgb(resolve_path(dataset_root, row["image_path"]))
            probability, _ = predict_full_image(
                model=model, image=image, device=device, data_mode=data_mode,
                tile_size=args.tile_size, stride=args.stride,
                tile_batch_size=tile_batch_size, amp=not args.no_amp,
            )
            encoded = Image.fromarray(np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8))
            atomic_png(encoded, target)
            processed += 1
            if position % 25 == 0 or position == len(records):
                elapsed = max(1e-6, time.perf_counter() - started)
                rate = position / elapsed
                remaining = (len(records) - position) / max(rate, 1e-6)
                print(f"[{split}] {position}/{len(records)} | {rate:.2f} image/s | ETA {remaining / 60:.1f} min", flush=True)
        print(f"[{split}] completed: exported={processed}, skipped={skipped}", flush=True)
    print(f"Portable probability cache ready: {output_root}")


if __name__ == "__main__":
    main()
