"""Generate a separate EXIF-correct B1 diagnostic set from a frozen role.

This preserves legacy raw-orientation sets and records source and output hashes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.make_degraded_set import degrade_image, degraded_filename
from src.straw_global_local.data import file_hash, load_protocol


def make_one(task):
    row, output, seed = task
    cv2.setNumThreads(1)
    sample_seed = int(hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()[:8], 16)
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    source = Path(row["image_path"])
    source_sha = file_hash(source)
    if source_sha != row["sha256"]:
        raise ValueError(f"Original image hash mismatch: {source}")
    with Image.open(source) as opened:
        orientation = int(opened.getexif().get(274, 1))
        image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    degraded = degrade_image(image)
    target = Path(output) / "images" / degraded_filename(source, row)
    Image.fromarray(degraded).save(target, quality=88)
    return dict(row, source_image_path=str(source), source_sha256=source_sha,
        source_exif_orientation=orientation, exif_transpose_applied=True,
        background_mask="border_connected_white", sample_seed=sample_seed,
        image_path=str(target), sha256=file_hash(target), is_degraded=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "artifacts/straw_global_local/data_v1")
    parser.add_argument("--role", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    protocol, frames = load_protocol(args.protocol)
    output = (args.output_dir or ROOT / "data" / f"b1_{args.role}_degraded_v2").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    (output / "images").mkdir(parents=True)
    code_paths = [ROOT / "scripts/data/make_degraded_set.py", Path(__file__).resolve()]
    code_sha = {path.name: file_hash(path) for path in code_paths}
    tasks = [(row, str(output), args.seed) for row in frames[args.role].to_dict("records")]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(make_one, tasks):
            records.append(row)
            if len(records) % 25 == 0 or len(records) == len(tasks):
                print(f"Generated {len(records)}/{len(tasks)} EXIF-correct images", flush=True)
    frame = pd.DataFrame(records)
    if frame.sample_id.tolist() != frames[args.role].sample_id.tolist():
        raise ValueError("Generated rows differ from frozen source role")
    if frame.image_path.nunique() != len(tasks):
        raise ValueError("Generated filenames collide")
    if code_sha != {path.name: file_hash(path) for path in code_paths}:
        raise ValueError("Generator source changed during the run; refuse to certify these outputs")
    frame.to_csv(output / f"{args.role}_degraded.csv", index=False)
    recipe = {
        "schema": 1, "role": args.role, "seed": args.seed,
        "purpose": "synthetic degradation diagnostic; not camera performance or clean test",
        "source_protocol": protocol["fingerprint"],
        "source_manifest_sha256": file_hash(args.protocol / f"{args.role}.csv"),
        "generator_sha256": code_sha,
        "randomness": "SHA256(base_seed:sample_id) first 32 bits; independent of row/worker order",
        "exif_transpose_before_degradation": True,
        "background_mask": "border_connected_white; gray backgrounds remain unchanged",
        "n": len(frame), "classes": protocol["classes"],
        "outputs": {str(row.sample_id): file_hash(row.image_path) for row in frame.itertuples()},
    }
    (output / "generation.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    print(f"B1 diagnostic manifest: {output / (args.role + '_degraded.csv')}")


if __name__ == "__main__":
    main()
