"""Audit Juice Box Straw Defect Dataset.

Performs:
1. Cross-tabulation of 8 brands x 5 classes (total & per-split).
2. Near-duplicate detection across splits using pHash (Hamming distance < threshold).
3. Verification of balance and missing combinations.
4. Export of audit report to ``submission/tables/audit_report.csv``.

Usage:
    .venv/Scripts/python.exe scripts/data/audit_dataset.py
    .venv/Scripts/python.exe scripts/data/audit_dataset.py --splits-dir data/straw_defects_splits --config configs/project.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.fftpack import dct

# Add repo root to import path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: Path) -> dict:
    """Load project YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_phash_fast(img_path: Path, hash_size: int = 8) -> int:
    """Compute 64-bit perceptual hash (pHash) with fast JPEG decoding.

    Args:
        img_path: Path to image file.
        hash_size: Hash grid size (default 8 -> 64-bit).

    Returns:
        64-bit unsigned integer hash.
    """
    with Image.open(img_path) as im:
        # Fast draft downscale for huge 3000x3000 images
        im.draft("L", (128, 128))
        im_gray = im.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(im_gray, dtype=np.float32)

    # 2D DCT
    dct_mat = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_low = dct_mat[:hash_size, :hash_size]

    # Median excluding DC term at (0, 0)
    low_flat = dct_low.flatten()
    med = float(np.median(low_flat[1:]))

    # Bit vector to uint64
    bits = (low_flat > med)
    hash_int = 0
    for b in bits:
        hash_int = (hash_int << 1) | int(b)
    return hash_int


def popcount64_matrix(xor_mat: np.ndarray) -> np.ndarray:
    """Fast vectorized population count (Hamming distance) for 64-bit unsigned integers."""
    x = xor_mat.astype(np.uint64)
    x = (x & np.uint64(0x5555555555555555)) + ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x & np.uint64(0x0F0F0F0F0F0F0F0F)) + ((x >> np.uint64(4)) & np.uint64(0x0F0F0F0F0F0F0F0F))
    x = (x & np.uint64(0x00FF00FF00FF00FF)) + ((x >> np.uint64(8)) & np.uint64(0x00FF00FF00FF00FF))
    x = (x & np.uint64(0x0000FFFF0000FFFF)) + ((x >> np.uint64(16)) & np.uint64(0x0000FFFF0000FFFF))
    x = (x & np.uint64(0x00000000FFFFFFFF)) + ((x >> np.uint64(32)) & np.uint64(0x00000000FFFFFFFF))
    return x.astype(np.int32)


def audit_splits(
    splits_dir: Path,
    hamming_th: int = 6,
    output_csv: Path | None = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Audit splits for balance, missing combinations, and cross-split near-duplicates.

    Args:
        splits_dir: Directory containing train.csv, val.csv, and test.csv.
        hamming_th: Maximum Hamming distance to consider as near-duplicate.
        output_csv: Path to save audit_report.csv.

    Returns:
        Tuple of (cross-tabulation dataframe, list of near-duplicate records).
    """
    split_files = {
        "train": splits_dir / "train.csv",
        "val": splits_dir / "val.csv",
        "test": splits_dir / "test.csv",
    }
    dfs = []
    for split_name, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required split file: {path}")
        df = pd.read_csv(path)
        df["split"] = split_name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    brand_col = "brand" if "brand" in all_df.columns else "brand_name"
    class_col = "raw_class" if "raw_class" in all_df.columns else "class_name"

    # 1. Cross-tabulation: Brand x Class x Split
    print("\n" + "=" * 70, flush=True)
    print("1. CROSS-TABULATION: 8 BRANDS x 5 CLASSES (ALL SPLITS)", flush=True)
    print("=" * 70, flush=True)
    crosstab_total = pd.crosstab(all_df[brand_col], all_df[class_col], margins=True)
    print(crosstab_total, flush=True)

    crosstab_split = pd.crosstab(
        [all_df[brand_col], all_df[class_col]],
        all_df["split"],
        margins=True,
    ).reset_index()

    expected_combos = 8 * 5
    actual_combos = len(all_df.groupby([brand_col, class_col]))
    print(f"\nTotal combinations present: {actual_combos}/{expected_combos}", flush=True)
    if actual_combos < expected_combos:
        print("[WARNING] Some brand x class combinations are completely MISSING!", flush=True)

    # 2. Near-duplicate detection across splits using pHash
    print("\n" + "=" * 70, flush=True)
    print(f"2. CROSS-SPLIT NEAR-DUPLICATE DETECTION (pHash Hamming < {hamming_th})", flush=True)
    print("=" * 70, flush=True)

    print("Computing perceptual hashes for all samples...", flush=True)
    t0 = time.time()
    hashes: List[int] = []
    valid_indices: List[int] = []

    for idx, row in all_df.iterrows():
        p = Path(str(row["image_path"]))
        if not p.exists():
            p = REPO_ROOT / p
        if not p.exists():
            continue
        try:
            h = compute_phash_fast(p)
            hashes.append(h)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}", flush=True)

    t_phash = time.time() - t0
    print(f"Computed {len(hashes)} pHashes in {t_phash:.2f}s", flush=True)

    sub_df = all_df.iloc[valid_indices].copy().reset_index(drop=True)
    hashes_arr = np.array(hashes, dtype=np.uint64)

    near_duplicates: List[Dict[str, Any]] = []

    splits = ["train", "val", "test"]
    split_indices = {s: np.where(sub_df["split"] == s)[0] for s in splits}

    # Cross-split comparisons: train vs val, train vs test, val vs test
    pairs_to_check = [("train", "val"), ("train", "test"), ("val", "test")]

    for s1, s2 in pairs_to_check:
        idx1 = split_indices[s1]
        idx2 = split_indices[s2]
        if len(idx1) == 0 or len(idx2) == 0:
            continue

        h1 = hashes_arr[idx1]
        h2 = hashes_arr[idx2]

        # Fast matrix XOR + popcount
        xor_mat = h1[:, None] ^ h2[None, :]
        dist_mat = popcount64_matrix(xor_mat)

        match_i, match_j = np.where(dist_mat < hamming_th)
        for i_pos, j_pos in zip(match_i, match_j):
            orig_i = idx1[i_pos]
            orig_j = idx2[j_pos]
            r1 = sub_df.iloc[orig_i]
            r2 = sub_df.iloc[orig_j]
            near_duplicates.append({
                "split_a": s1,
                "img_a": r1["image_path"],
                "brand_a": r1[brand_col],
                "class_a": r1[class_col],
                "split_b": s2,
                "img_b": r2["image_path"],
                "brand_b": r2[brand_col],
                "class_b": r2[class_col],
                "hamming_distance": int(dist_mat[i_pos, j_pos]),
            })

    if near_duplicates:
        print(f"[ALERT] Found {len(near_duplicates)} cross-split near-duplicate pairs!", flush=True)
        dup_df = pd.DataFrame(near_duplicates)
        print(dup_df.head(10), flush=True)
    else:
        print(f"[PASSED] Zero cross-split near-duplicates found (Hamming < {hamming_th}). No data leakage.", flush=True)

    # 3. Export audit report
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        audit_rows = []
        for (b, c), group in all_df.groupby([brand_col, class_col]):
            train_count = len(group[group["split"] == "train"])
            val_count = len(group[group["split"] == "val"])
            test_count = len(group[group["split"] == "test"])
            total_count = len(group)
            status = "BALANCED" if total_count == 50 else ("LOW" if total_count < 50 else "HIGH")
            audit_rows.append({
                "brand": b,
                "defect_class": c,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "total_count": total_count,
                "status": status,
            })
        report_df = pd.DataFrame(audit_rows)
        report_df.to_csv(output_csv, index=False)
        print(f"\n[SAVED] Audit report exported to: {output_csv}", flush=True)

        if near_duplicates:
            dup_csv = output_csv.parent / "near_duplicates_report.csv"
            pd.DataFrame(near_duplicates).to_csv(dup_csv, index=False)
            print(f"[SAVED] Near duplicates report exported to: {dup_csv}", flush=True)

    return crosstab_split, near_duplicates


def main():
    parser = argparse.ArgumentParser(description="Audit dataset for balance and cross-split leakage.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "project.yaml"),
        help="Path to project.yaml",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "straw_defects_splits"),
        help="Path to directory containing split CSVs",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(REPO_ROOT / "submission" / "tables" / "audit_report.csv"),
        help="Path to save audit_report.csv",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    hamming_th = int(cfg.get("dataset", {}).get("phash_hamming_threshold", 6))

    audit_splits(
        splits_dir=Path(args.splits_dir),
        hamming_th=hamming_th,
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
