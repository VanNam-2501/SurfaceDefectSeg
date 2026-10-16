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
import random
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def replace_studio_bg_with_conveyor(img_rgb: np.ndarray) -> np.ndarray:
    """Replace studio white background with simulated factory conveyor belt texture."""
    h, w = img_rgb.shape[:2]
    # Mask studio white background: R,G,B > 238
    is_bg = (img_rgb[:, :, 0] > 238) & (img_rgb[:, :, 1] > 238) & (img_rgb[:, :, 2] > 238)

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
) -> pd.DataFrame:
    """Generate degraded validation set from val.csv.

    Args:
        val_csv: Path to original val.csv.
        output_dir: Directory to save degraded images and val_degraded.csv.
        seed: Random seed.

    Returns:
        DataFrame pointing to the new degraded images with identical labels.
    """
    random.seed(seed)
    np.random.seed(seed)

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

        pil_im = Image.open(orig_path).convert("RGB")
        img_np = np.asarray(pil_im)

        # Apply degradation pipeline
        degraded_np = degrade_image(img_np)

        # Save degraded image
        out_filename = f"degraded_{orig_path.stem}.jpg"
        out_img_path = images_dir / out_filename
        Image.fromarray(degraded_np).save(out_img_path, quality=88)

        # Build duplicate row with updated image path
        record = dict(row)
        record["image_path"] = str(out_img_path)
        record["is_degraded"] = True
        degraded_records.append(record)

    deg_df = pd.DataFrame(degraded_records)
    out_csv = output_dir / "val_degraded.csv"
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
        default=str(REPO_ROOT / "data" / "val_degraded"),
        help="Output directory for degraded images",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_degraded_val_set(
        val_csv=Path(args.val_csv),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
