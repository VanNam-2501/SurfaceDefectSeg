"""Fast pre-flight checks before a final evaluation or Kaggle run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "3cad_ani"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_group(value: str) -> str:
    low = str(value).strip().lower()
    mapping = {
        "scratche": "scratches",
        "scratch": "scratches",
        "multiple-defects": "Multiple-defects",
        "multiple_defects": "Multiple-defects",
        "good": "Good",
    }
    return mapping.get(low, str(value).strip())


def group_series(df: pd.DataFrame) -> pd.Series:
    for c in ["defect_group", "defect_type", "group", "class", "category", "folder"]:
        if c in df.columns:
            s = df[c].fillna("").astype(str).map(normalize_group)
            s.loc[df["label"].astype(int) == 0] = "Good"
            return s
    s = df["image_path"].astype(str).map(lambda x: normalize_group(Path(x).parent.name))
    s.loc[df["label"].astype(int) == 0] = "Good"
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    p.add_argument("--train-csv", default=str(DEFAULT_DATASET_ROOT / "dataset_audit" / "splits" / "train.csv"))
    p.add_argument("--val-csv", default=str(DEFAULT_DATASET_ROOT / "dataset_audit" / "splits" / "val.csv"))
    p.add_argument("--test-csv", default=str(DEFAULT_DATASET_ROOT / "dataset_audit" / "splits" / "test.csv"))
    p.add_argument("--save", default=str(REPO_ROOT / "artifacts" / "verification" / "protocol_check.json"))
    args = p.parse_args()

    paths = {"train": Path(args.train_csv), "val": Path(args.val_csv), "test": Path(args.test_csv)}
    dfs = {name: pd.read_csv(path) for name, path in paths.items()}
    report: dict = {"splits": {}, "sha256": {k: sha256(v) for k, v in paths.items()}}

    required = {"image_path", "mask_path", "label"}
    for name, df in dfs.items():
        assert required.issubset(df.columns), f"{name} missing columns: {required - set(df.columns)}"
        assert set(pd.to_numeric(df["label"]).astype(int).unique()).issubset({0, 1})
        assert not df["image_path"].duplicated().any(), f"Duplicate image_path within {name}"
        report["splits"][name] = {
            "n": len(df),
            "good": int((df.label.astype(int) == 0).sum()),
            "defect": int((df.label.astype(int) == 1).sum()),
            "groups": group_series(df).value_counts().to_dict(),
        }
        print(name, report["splits"][name])

    sets = {k: set(v.image_path.astype(str)) for k, v in dfs.items()}
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"

    # Paths alone are insufficient: two differently named files may contain
    # identical pixels. Use hashes emitted by the dataset audit when available.
    if all("sha256" in d.columns for d in dfs.values()):
        hashes = {
            name: set(df["sha256"].dropna().astype(str)) - {""}
            for name, df in dfs.items()
        }
        assert not (hashes["train"] & hashes["val"]), "train/val exact-content duplicate"
        assert not (hashes["train"] & hashes["test"]), "train/test exact-content duplicate"
        assert not (hashes["val"] & hashes["test"]), "val/test exact-content duplicate"
        report["exact_content_overlap_check"] = "PASS"
    else:
        report["exact_content_overlap_check"] = "SKIPPED: sha256 column unavailable"

    if all("image_id" in d.columns for d in dfs.values()):
        ids = {k: set(v.image_id.astype(str)) for k, v in dfs.items()}
        assert not (ids["train"] & ids["val"]), "train/val image_id overlap"
        assert not (ids["train"] & ids["test"]), "train/test image_id overlap"
        assert not (ids["val"] & ids["test"]), "val/test image_id overlap"

    root = Path(args.dataset_root)
    missing_images = []
    missing_masks = []
    for name, df in dfs.items():
        for _, row in df.iterrows():
            ip = Path(str(row.image_path))
            ip = ip if ip.is_absolute() else root / ip
            if not ip.exists():
                missing_images.append(str(ip))
            if int(row.label) == 1:
                mp = Path(str(row.mask_path))
                mp = mp if mp.is_absolute() else root / mp
                if not mp.exists():
                    missing_masks.append(str(mp))

    assert not missing_images, f"Missing {len(missing_images)} images; examples={missing_images[:5]}"
    assert not missing_masks, f"Missing {len(missing_masks)} masks; examples={missing_masks[:5]}"

    total = sum(len(x) for x in dfs.values())
    report["total"] = total
    report["overlap_check"] = "PASS"
    report["files_check"] = "PASS"
    Path(args.save).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SHA256:")
    for k, v in report["sha256"].items():
        print(f"  {k}: {v}")
    print(f"TOTAL: {total}")
    print("PROTOCOL CHECK: PASS")


if __name__ == "__main__":
    main()
