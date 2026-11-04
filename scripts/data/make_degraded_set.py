"""Factory Environment Degradation Simulator for Validation Set.

Generates a realistic factory-degraded validation set from ``val.csv``:
1. Motion blur (simulating conveyor movement / vibration).
2. Gaussian sensor noise.
3. Illumination drop (30% - 50% brightness reduction).
4. Perspective angle distortion (±5 degrees homography).
5. Background replacement: studio white replaced with industrial conveyor gray / texture.

CRITICAL PROTOCOL REQUIREMENT:
- All defect labels and filenames are strictly preserved.
- This set is EXCLUSIVELY used for calibration stress-testing and audit.
- MUST NOT BE USED FOR TRAINING.

Usage:
    .venv/Scripts/python.exe scripts/data/make_degraded_set.py
    .venv/Scripts/python.exe scripts/data/make_degraded_set.py --val-csv data/straw_defects_splits/val.csv
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def degraded_filename(orig_path: Path, row: dict) -> str:
    """Return a stable output filename that is unique across brands."""
    brand = str(row.get("brand", orig_path.parent.parent.name)).strip() or "unknown"
    safe_brand = re.sub(r"[^A-Za-z0-9.-]+", "_", brand).strip("_") or "unknown"
    identity = f"{brand}|{orig_path.parent.name}|{orig_path.name}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"degraded_{safe_brand}_{orig_path.stem}_{suffix}.jpg"


def apply_motion_blur(img: np.ndarray, max_kernel_size: int = 13) -> np.ndarray:
    """Apply directional motion blur simulating conveyor travel."""
    k_size = random.choice([7, 9, 11, max_kernel_size])
    kernel = np.zeros((k_size, k_size), dtype=np.float32)

    # Random angle between -20 and +20 degrees (conveyor direction)
    angle = random.uniform(-20, 20)
    pt1 = (int(k_size / 2 - np.cos(np.radians(angle)) * k_size / 2),
           int(k_size / 2 - np.sin(np.radians(angle)) * k_size / 2))
    pt2 = (int(k_size / 2 + np.cos(np.radians(angle)) * k_size / 2),
           int(k_size / 2 + np.sin(np.radians(angle)) * k_size / 2))
    cv2.line(kernel, pt1, pt2, 1.0, 1)
    kernel /= max(1.0, kernel.sum())

    return cv2.filter2D(img, -1, kernel)


def apply_gaussian_noise(img: np.ndarray, sigma_range: Tuple[float, float] = (10.0, 22.0)) -> np.ndarray:
    """Add Gaussian sensor noise."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_brightness_drop(img: np.ndarray, factor_range: Tuple[float, float] = (0.50, 0.70)) -> np.ndarray:
    """Reduce brightness by 30% to 50% (factor 0.50 - 0.70)."""
    factor = random.uniform(*factor_range)
    darkened = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return darkened


def apply_perspective_distortion(img: np.ndarray, max_distortion_pct: float = 0.05) -> np.ndarray:
    """Apply slight perspective homography (±5 degrees camera misalignment)."""
    h, w = img.shape[:2]
    dx = w * max_distortion_pct
    dy = h * max_distortion_pct

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), h - random.uniform(0, dy)],
        [random.uniform(0, dx), h - random.uniform(0, dy)],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return warped


def studio_background_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Select bright regions connected to the image border, preserving white objects.

    Gray studio backgrounds are left unchanged. Color alone cannot safely infer
    their boundary against a white straw or carton; an empty mask is preferable
    to erasing a defect cue inside the product.
    """
    white = np.all(img_rgb > 238, axis=2).astype(np.uint8)
    _, components = cv2.connectedComponents(white)
    border = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    border = border[border != 0]
    return (white != 0) & np.isin(components, border)


def replace_studio_bg_with_conveyor(img_rgb: np.ndarray, *, legacy_all_white: bool = False) -> np.ndarray:
    """Replace only supported background pixels; legacy mode is explicit."""
    h, w = img_rgb.shape[:2]
    # Mask studio white background: R,G,B > 238
    is_bg = np.all(img_rgb > 238, axis=2) if legacy_all_white else studio_background_mask(img_rgb)

    # Generate conveyor background: dark gray brushed surface (mean ~ 95)
    base_gray = random.randint(85, 110)
    conveyor = np.full((h, w, 3), base_gray, dtype=np.uint8)
    # Add subtle horizontal striated noise simulating moving belt
    line_noise = np.random.normal(0, 6, (h, 1)).astype(np.int16)
    conveyor = np.clip(conveyor.astype(np.int16) + line_noise, 0, 255).astype(np.uint8)

    result = img_rgb.copy()
    result[is_bg] = conveyor[is_bg]
    return result


def degrade_image(img_rgb: np.ndarray) -> np.ndarray:
    """Apply composite realistic industrial factory degradation pipeline."""
    # 1. Background replacement (conveyor belt)
    out = replace_studio_bg_with_conveyor(img_rgb)
    # 2. Illumination drop (30-50% dimmer)
    out = apply_brightness_drop(out, factor_range=(0.50, 0.70))
    # 3. Motion blur
    out = apply_motion_blur(out, max_kernel_size=11)
    # 4. Gaussian sensor noise
    out = apply_gaussian_noise(out, sigma_range=(10.0, 20.0))
    # 5. Perspective angle misalignment (±5%)
    out = apply_perspective_distortion(out, max_distortion_pct=0.04)
    return out


def build_degraded_val_set(
    val_csv: Path,
    output_dir: Path,
    seed: int = 42,
    exif_transpose: bool = False,
    manifest_name: str = "val_degraded.csv",
) -> pd.DataFrame:
    """Generate degraded validation set from val.csv.

    Args:
        val_csv: Path to original val.csv.
        output_dir: Directory to save degraded images and val_degraded.csv.
        seed: Random seed.
        exif_transpose: Match B1 training by applying EXIF before degradation.
        manifest_name: Output CSV basename, e.g. test_degraded.csv.

    Returns:
        DataFrame pointing to the new degraded images with identical labels.
    """
    random.seed(seed)
    np.random.seed(seed)
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(".csv"):
        raise ValueError("manifest_name must be a CSV basename")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(val_csv)
    print(f"Generating factory-degraded validation images for {len(val_df)} samples...")

    degraded_records = []
    for idx, row in val_df.iterrows():
        orig_path = Path(str(row["image_path"]))
        if not orig_path.exists():
            orig_path = REPO_ROOT / orig_path
        if not orig_path.exists():
            print(f"[WARN] Val image missing: {orig_path}")
            continue

        with Image.open(orig_path) as source:
            source_orientation = int(source.getexif().get(274, 1))
            if exif_transpose:
                source = ImageOps.exif_transpose(source)
            pil_im = source.convert("RGB")
        img_np = np.asarray(pil_im)

        # Apply degradation pipeline
        degraded_np = degrade_image(img_np)

        # Save degraded image
        out_filename = degraded_filename(orig_path, row)
        out_img_path = images_dir / out_filename
        Image.fromarray(degraded_np).save(out_img_path, quality=88)

        # Build duplicate row with updated image path
        record = dict(row)
        record["source_image_path"] = str(orig_path)
        record["source_sha256"] = hashlib.sha256(orig_path.read_bytes()).hexdigest()
        record["sha256"] = hashlib.sha256(out_img_path.read_bytes()).hexdigest()
        record["source_exif_orientation"] = source_orientation
        record["exif_transpose_applied"] = exif_transpose
        record["image_path"] = str(out_img_path)
        record["is_degraded"] = True
        degraded_records.append(record)

    deg_df = pd.DataFrame(degraded_records)
    if len(deg_df) != len(val_df) or deg_df["image_path"].nunique() != len(deg_df):
        raise RuntimeError(
            f"Generated degraded validation set is incomplete or has path collisions: "
            f"rows={len(deg_df)}/{len(val_df)}, unique_paths={deg_df['image_path'].nunique()}"
        )
    out_csv = output_dir / manifest_name
    deg_df.to_csv(out_csv, index=False)
    print(f"[DONE] Degraded validation set generated: {len(deg_df)} images.")
    print(f"Saved manifest to: {out_csv}")
    return deg_df


def main():
    parser = argparse.ArgumentParser(description="Generate industrial degraded validation set.")
    parser.add_argument(
        "--val-csv",
        type=str,
        default=str(REPO_ROOT / "data" / "straw_defects_splits" / "val.csv"),
        help="Path to original val.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "val_degraded_v2"),
        help="Output directory for degraded images",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--exif-transpose", action="store_true",
        help="Apply EXIF orientation before degradation (use for B1-aligned inputs).",
    )
    args = parser.parse_args()

    build_degraded_val_set(
        val_csv=Path(args.val_csv),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        exif_transpose=args.exif_transpose,
    )


if __name__ == "__main__":
    main()
