"""Generate pseudo-masks for juice box straws.

Creates straw masks for ~200 images (40 per class across 5 classes: Good, Bent, Loose, Missing, Pierced).
Methodology:
- For Good, Bent, Loose, Pierced: Adaptive thresholding & morphology in Lab/HSV space
  inside the segmented juice box foreground.
- For Missing: Centered region-of-interest mask representing the expected straw attachment zone.
- Optional fallback: MobileSAM if installed and checkpoint available.

Outputs:
- Binary PNG masks in ``data/masks/*.png``
- Mask manifest in ``data/masks/masks.csv``
- 10 visual inspection overlays in ``data/masks/visual_check/*.jpg``

Usage:
    .venv/Scripts/python.exe scripts/data/make_straw_masks.py
    .venv/Scripts/python.exe scripts/data/make_straw_masks.py --samples-per-class 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def segment_juice_box(img_rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Segment the juice box packaging from white studio background.

    Args:
        img_rgb: RGB image numpy array (H, W, 3).

    Returns:
        Tuple of (binary box mask, (x, y, w, h) bounding box).
    """
    # Studio background is near pure white: R, G, B > 238
    is_not_bg = ~((img_rgb[:, :, 0] > 238) & (img_rgb[:, :, 1] > 238) & (img_rgb[:, :, 2] > 238))
    box_mask = (is_not_bg.astype(np.uint8)) * 255

    # Morphology to fill interior holes inside packaging
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, kernel)

    # Find largest contour (the juice box body)
    contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = img_rgb.shape[:2]
        return box_mask, (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)

    clean_box = np.zeros_like(box_mask)
    cv2.drawContours(clean_box, [largest], -1, 255, -1)
    return clean_box, (bx, by, bw, bh)


def segment_straw_mask(
    img_rgb: np.ndarray,
    class_name: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Generate binary mask for straw or expected straw attachment zone.

    Args:
        img_rgb: RGB image numpy array (H, W, 3).
        class_name: Standardized class name ('good', 'bent', 'loose', 'missing', 'pierced').

    Returns:
        Tuple of (binary uint8 mask where 255=straw, (x, y, w, h) straw bounding box).
    """
    h, w = img_rgb.shape[:2]
    c_name = class_name.lower().strip()
    box_mask, (bx, by, bw, bh) = segment_juice_box(img_rgb)

    straw_mask = np.zeros((h, w), dtype=np.uint8)

    # Case 1: Missing Straw -> Mask expected central zone where straw should be
    if "miss" in c_name:
        mx = bx + int(bw * 0.35)
        my = by + int(bh * 0.15)
        mw = max(10, int(bw * 0.30))
        mh = max(10, int(bh * 0.70))
        straw_mask[my:my + mh, mx:mx + mw] = 255
        return straw_mask, (mx, my, mw, mh)

    # Case 2: Straw present (Good, Bent, Loose, Pierced)
    # Convert to HSV and Lab
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # White plastic straw features: low saturation, high value/lightness
    # Also transparent plastic sleeve reflects bright highlights
    sat = img_hsv[:, :, 1]
    val = img_hsv[:, :, 2]
    lightness = img_lab[:, :, 0]

    # White straw mask within the box
    white_straw = (sat < 70) & (val > 155) & (lightness > 160) & (box_mask > 0)
    straw_candidate = white_straw.astype(np.uint8) * 255

    # Loose or pierced straws can extend outside the box boundary:
    if "loose" in c_name or "pierce" in c_name:
        # Include non-background components with high edge magnitude / white plastic color
        straw_candidate = ((sat < 65) & (val > 150) & (lightness > 150)).astype(np.uint8) * 255

    # Clean morphology: connect straw parts, remove small specks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    straw_cleaned = cv2.morphologyEx(straw_candidate, cv2.MORPH_CLOSE, kernel_close)
    straw_cleaned = cv2.morphologyEx(straw_cleaned, cv2.MORPH_OPEN, kernel_open)

    # Connected component filtering: keep components that match straw scale
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(straw_cleaned)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    min_area = int(bw * bh * 0.005)  # At least 0.5% of box area
    max_area = int(bw * bh * 0.45)   # Less than 45% of box area

    valid_components = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(cw, ch) / max(1, min(cw, ch))

        if min_area <= area <= max_area and (aspect > 1.4 or ch > bh * 0.25):
            valid_components.append(lbl)

    if valid_components:
        for lbl in valid_components:
            final_mask[labels == lbl] = 255
    else:
        # Fallback to largest central elongated component
        straw_mask_fallback = np.zeros((h, w), dtype=np.uint8)
        mx = bx + int(bw * 0.25)
        my = by + int(bh * 0.10)
        mw = int(bw * 0.50)
        mh = int(bh * 0.80)
        straw_mask_fallback[my:my + mh, mx:mx + mw] = straw_cleaned[my:my + mh, mx:mx + mw]
        final_mask = straw_mask_fallback

    # Find final bounding box
    nz_y, nz_x = np.where(final_mask > 0)
    if len(nz_x) > 0:
        sx = int(nz_x.min())
        sy = int(nz_y.min())
        sw = int(nz_x.max() - sx)
        sh = int(nz_y.max() - sy)
        bbox = (sx, sy, sw, sh)
    else:
        # Fallback default bbox
        bbox = (bx + int(bw * 0.3), by + int(bh * 0.2), int(bw * 0.4), int(bh * 0.6))
        final_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 255

    return final_mask, bbox


def create_inspection_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    title: str = "",
) -> np.ndarray:
    """Create visual inspection overlay showing image, mask and bounding box.

    Args:
        img_rgb: Original RGB image.
        mask: Binary mask (255=straw).
        bbox: (x, y, w, h) bounding box.
        title: Title banner text.

    Returns:
        RGB overlay image numpy array.
    """
    overlay = img_rgb.copy()
    # Green translucent mask
    green_layer = np.zeros_like(img_rgb)
    green_layer[:, :, 1] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * green_layer[mask_bool]).astype(np.uint8)

    # Draw yellow bounding box
    x, y, w, h = bbox
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 4)

    # Downscale for visual inspection saving
    vis_h, vis_w = 600, 600
    orig_small = cv2.resize(img_rgb, (vis_w, vis_h))
    over_small = cv2.resize(overlay, (vis_w, vis_h))
    mask_colored = cv2.cvtColor(cv2.resize(mask, (vis_w, vis_h)), cv2.COLOR_GRAY2RGB)

    panel = np.hstack([orig_small, over_small, mask_colored])
    return panel


def generate_straw_masks(
    splits_csv: Path,
    output_dir: Path,
    visual_dir: Path,
    samples_per_class: int = 40,
) -> pd.DataFrame:
    """Generate straw pseudo-masks across stratified subset of images.

    Args:
        splits_csv: Path to train.csv or combined splits CSV.
        output_dir: Directory to save binary mask PNGs.
        visual_dir: Directory to save visual verification overlays.
        samples_per_class: Number of samples per class (default 40 -> 200 total).

    Returns:
        DataFrame manifest of generated masks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(splits_csv)
    brand_col = "brand" if "brand" in df.columns else "brand_name"
    class_col = "raw_class" if "raw_class" in df.columns else "class_name"

    classes = sorted(df[class_col].unique())
    selected_rows = []

    # Stratified selection: pick evenly across brands for each class
    for cls in classes:
        cls_df = df[df[class_col] == cls]
        brands = sorted(cls_df[brand_col].unique())
        per_brand = max(1, samples_per_class // len(brands))

        cls_selected = []
        for b in brands:
            b_df = cls_df[cls_df[brand_col] == b]
            take_n = min(len(b_df), per_brand)
            cls_selected.append(b_df.iloc[:take_n])

        cat = pd.concat(cls_selected, ignore_index=True)
        # Pad to exactly samples_per_class if needed
        if len(cat) < samples_per_class:
            rem = cls_df[~cls_df.index.isin(cat.index)]
            needed = samples_per_class - len(cat)
            cat = pd.concat([cat, rem.iloc[:needed]], ignore_index=True)
        selected_rows.append(cat.iloc[:samples_per_class])

    target_df = pd.concat(selected_rows, ignore_index=True)
    print(f"Generating masks for {len(target_df)} images ({samples_per_class} per class)...")

    manifest = []
    visual_count = {cls: 0 for cls in classes}

    for i, row in target_df.iterrows():
        img_path = Path(str(row["image_path"]))
        if not img_path.exists():
            img_path = REPO_ROOT / img_path
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        c_name = str(row[class_col]).lower().strip()
        brand_name = str(row[brand_col])

        # Read image
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img_pil)

        mask, bbox = segment_straw_mask(img_np, c_name)

        # Save mask PNG
        mask_filename = f"{img_path.stem}_mask.png"
        mask_path = output_dir / mask_filename
        Image.fromarray(mask).save(mask_path)

        manifest.append({
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "class_name": c_name,
            "brand": brand_name,
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
        })

        # Save visual inspection overlay (2 per class -> 10 visual checks)
        if visual_count[c_name] < 2:
            visual_count[c_name] += 1
            vis_panel = create_inspection_overlay(
                img_np, mask, bbox, title=f"{c_name} - {brand_name}"
            )
            vis_out = visual_dir / f"check_{c_name}_{visual_count[c_name]}.jpg"
            Image.fromarray(vis_panel).save(vis_out, quality=90)

    manifest_df = pd.DataFrame(manifest)
    manifest_csv = output_dir / "masks.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\n[DONE] Generated {len(manifest_df)} masks.")
    print(f"Mask manifest saved to: {manifest_csv}")
    print(f"Inspection overlays saved to: {visual_dir}")
    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Generate straw pseudo-masks for 200 images.")
    parser.add_argument(
        "--splits-csv",
        type=str,
        default=str(REPO_ROOT / "data" / "straw_defects_splits" / "train.csv"),
        help="Path to splits train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "masks"),
        help="Directory to save masks and manifest",
    )
    parser.add_argument(
        "--visual-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "masks" / "visual_check"),
        help="Directory to save visual inspection overlays",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=40,
        help="Samples per class (default 40 -> 200 total)",
    )
    args = parser.parse_args()

    generate_straw_masks(
        splits_csv=Path(args.splits_csv),
        output_dir=Path(args.output_dir),
        visual_dir=Path(args.visual_dir),
        samples_per_class=args.samples_per_class,
    )


if __name__ == "__main__":
    main()
