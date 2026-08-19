"""Read-only audit for label/mask consistency and display encoding."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    frames = [
        pd.read_csv(root / "dataset_audit" / "splits" / f"{split}.csv")
        for split in ("train", "val", "test")
    ]
    frame = pd.concat(frames, ignore_index=True)
    defect_rows = frame[frame["label"] == 1]
    good_rows = frame[frame["label"] == 0]
    result: dict[str, object] = {
        "total": int(len(frame)),
        "defect": int(len(defect_rows)),
        "good": int(len(good_rows)),
        "good_with_mask_path": int(
            good_rows["mask_path"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        "missing_defect_mask": 0,
        "empty_defect_mask": 0,
        "dimension_mismatch": 0,
        "declared_pixel_mismatch": 0,
        "mask_max_values": Counter(),
        "examples": [],
    }
    for row in defect_rows.itertuples():
        mask_path = (root / str(row.mask_path)).resolve()
        if not mask_path.is_file():
            result["missing_defect_mask"] += 1
            continue
        with Image.open(mask_path) as image:
            original = np.asarray(image)
        binary = np.any(original != 0, axis=2) if original.ndim == 3 else original != 0
        pixels = int(binary.sum())
        result["mask_max_values"][int(original.max())] += 1
        result["empty_defect_mask"] += int(pixels == 0)
        result["dimension_mismatch"] += int(
            binary.shape != (int(row.height), int(row.width))
        )
        result["declared_pixel_mismatch"] += int(pixels != int(row.defect_pixels))
        if len(result["examples"]) < 5:
            result["examples"].append(
                {
                    "image_id": row.image_id,
                    "pixels": pixels,
                    "max_value": int(original.max()),
                    "unique_values": np.unique(original)[:8].tolist(),
                }
            )
    result["mask_max_values"] = dict(result["mask_max_values"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
