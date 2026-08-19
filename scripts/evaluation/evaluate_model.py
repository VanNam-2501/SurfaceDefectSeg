"""Evaluate one trained model using full-resolution inference.

Validation:
- runs the frozen full-resolution protocol
- selects threshold using Validation only
- saves threshold_scan.csv and selected_threshold.json

Test:
- requires the threshold selected on Validation
- produces E2-E6 metrics and qualitative examples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "threecad_segmentation"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fullres_eval import evaluate_split
from model_factory import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["unet", "segformer", "vmamba"])
    p.add_argument("--checkpoint", required=True)
    dataset_root = REPO_ROOT / "data" / "3cad_ani"
    p.add_argument("--dataset-root", default=str(dataset_root))
    p.add_argument("--train-csv", default=str(dataset_root / "dataset_audit" / "splits" / "train.csv"))
    p.add_argument("--val-csv", default=str(dataset_root / "dataset_audit" / "splits" / "val.csv"))
    p.add_argument("--test-csv", default=str(dataset_root / "dataset_audit" / "splits" / "test.csv"))
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--threshold-file", default="")
    p.add_argument("--data-mode", choices=["auto", "patch", "resize"], default="auto")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--tile-batch-size", type=int, default=4)
    p.add_argument("--fnr-limit", type=float, default=0.10)
    p.add_argument("--hist-bins", type=int, default=4096)
    p.add_argument("--qualitative", type=int, default=6)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    cfg = state.get("config", {})
    data_mode = cfg.get("data_mode", "patch") if args.data_mode == "auto" else args.data_mode

    model = build_model(args.model, pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device)

    run_dir = checkpoint.parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / ("validation" if args.split == "val" else "test")
    csv_path = args.val_csv if args.split == "val" else args.test_csv

    if args.split == "val":
        metrics = evaluate_split(
            model=model,
            csv_path=csv_path,
            dataset_root=args.dataset_root,
            train_csv=args.train_csv,
            output_dir=out_dir,
            device=device,
            data_mode=data_mode,
            threshold=None,
            select_threshold=True,
            fnr_limit=args.fnr_limit,
            tile_size=args.tile_size,
            stride=args.stride,
            tile_batch_size=args.tile_batch_size,
            amp=not args.no_amp,
            hist_bins=args.hist_bins,
            save_qualitative=args.qualitative,
            warmup_batches=args.warmup_batches,
        )
    else:
        threshold_file = Path(args.threshold_file) if args.threshold_file else run_dir / "validation" / "selected_threshold.json"
        if not threshold_file.exists():
            raise FileNotFoundError(
                f"Validation threshold not found: {threshold_file}. Run --split val first."
            )
        threshold = float(json.loads(threshold_file.read_text(encoding="utf-8"))["threshold"])
        metrics = evaluate_split(
            model=model,
            csv_path=csv_path,
            dataset_root=args.dataset_root,
            train_csv=args.train_csv,
            output_dir=out_dir,
            device=device,
            data_mode=data_mode,
            threshold=threshold,
            select_threshold=False,
            tile_size=args.tile_size,
            stride=args.stride,
            tile_batch_size=args.tile_batch_size,
            amp=not args.no_amp,
            hist_bins=args.hist_bins,
            save_qualitative=args.qualitative,
            warmup_batches=args.warmup_batches,
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
