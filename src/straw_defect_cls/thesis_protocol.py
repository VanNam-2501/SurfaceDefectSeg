"""Auditable data preparation for the straw thesis (no model or torch imports)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["good", "bent", "loose", "missing", "pierced"]
ROLES = ("train", "val", "calibration", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_object(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def scan_dataset(root: Path, cache_root: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Byte/pixel duplicate identities; dHash is only a candidate for human review."""
    records, failures = [], []
    root = root.resolve()
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        name = path.parent.name.lower().replace("_straw", "").strip()
        if name not in CLASSES:
            failures.append({"path": str(path), "error": "unrecognized class folder"})
            continue
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                pixel_hash = hashlib.sha256(f"{width}x{height}:RGB".encode() + image.tobytes()).hexdigest()
                gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
                dhash = sum(int(bit) << i for i, bit in enumerate((gray[:, 1:] > gray[:, :-1]).ravel()))
                relative = path.relative_to(root).as_posix()
                sample_id = hashlib.sha256(relative.encode()).hexdigest()[:20]
                stored_path = path
                if cache_root is not None:
                    stored_path = cache_root.resolve() / f"{sample_id}.png"
                    image.resize((384, 384), Image.Resampling.BILINEAR).save(stored_path)
            relative = path.relative_to(root).as_posix()
            records.append({
                "sample_id": sample_id,
                "relative_path": relative, "image_path": str(stored_path), "source_path": str(path),
                "brand": path.parent.parent.name, "class_name": name, "raw_class": name,
                "label": CLASSES.index(name), "width": width, "height": height,
                "sha256": digest_file(stored_path), "source_sha256": digest_file(path), "pixel_sha256": pixel_hash,
                "group_id": pixel_hash, "dhash": f"{dhash:016x}",
            })
            if len(records) % 100 == 0:
                print(f"Scanned {len(records)} images", flush=True)
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
    if not records:
        raise ValueError(f"No recognized images in {root}")
    return pd.DataFrame(records), failures


def split_groups(frame: pd.DataFrame, seed: int, held_out_brand: str | None = None) -> pd.DataFrame:
    """20 group-stratified folds approximate 60/15/10/15; no shared group across roles.

    LOBO uses the held-out brand exclusively for test, and 70/15/15 of the
    remaining brands for train/val/calibration. Fold allocation never uses models.
    """
    frame = frame.copy().reset_index(drop=True)
    frame["role"] = ""
    if held_out_brand is not None:
        test = frame.brand == held_out_brand
        if not test.any():
            raise ValueError(f"Unknown held-out brand: {held_out_brand}")
        # A duplicate from the unseen brand cannot also be in training.
        test_groups = set(frame.loc[test, "group_id"])
        frame.loc[test, "role"] = "test"
        frame.loc[~test & frame.group_id.isin(test_groups), "role"] = "excluded_cross_brand_duplicate"
        remaining = frame[frame.role == ""].copy()
        fold_roles = ["train"] * 14 + ["val"] * 3 + ["calibration"] * 3
    else:
        remaining = frame.copy()
        fold_roles = ["train"] * 12 + ["val"] * 3 + ["calibration"] * 2 + ["test"] * 3
    # Brand x class stratification, rather than class only, exposes brand counts.
    strata = remaining.brand.astype(str) + ":" + remaining.label.astype(str)
    counts = remaining.assign(stratum=strata).groupby("stratum").group_id.nunique()
    if counts.min() < 20:
        raise ValueError("Need >=20 independent groups per brand/class for the fixed 20-fold protocol")
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(remaining, strata, remaining.group_id)):
        frame.loc[remaining.index[indices], "role"] = fold_roles[fold]
    verify_assignments(frame)
    return frame


def verify_assignments(frame: pd.DataFrame) -> None:
    if not set(frame.role).issubset(set(ROLES) | {"excluded_cross_brand_duplicate"}):
        raise ValueError("Unknown split role")
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    exact_column = "pixel_sha256" if "pixel_sha256" in frame else "group_id"
    if (frame.groupby(exact_column).label.nunique() > 1).any():
        raise ValueError("Identical decoded images have contradictory class labels; review first")
    active = frame[frame.role.isin(ROLES)]
    if (active.groupby("group_id").role.nunique() > 1).any():
        raise ValueError("Content/group leakage between roles")
    for role in ROLES:
        if set(active[active.role == role].label) != set(range(5)):
            raise ValueError(f"Missing classes in {role}")


def export_protocol(frame: pd.DataFrame, output: Path, seed: int, source_root: Path,
                    held_out_brand: str | None = None, smoke: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite protocol: {output}. Use a new output directory.")
    verify_assignments(frame)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "inventory.csv", index=False)
    hashes = {}
    for role in ROLES:
        part = frame[frame.role == role].sort_values("sample_id")
        part.to_csv(output / f"{role}.csv", index=False)
        hashes[f"{role}.csv"] = digest_file(output / f"{role}.csv")
    metadata = {
        "schema": 1, "seed": seed, "classes": CLASSES, "source_root": str(source_root.resolve()),
        "held_out_brand": held_out_brand, "smoke_only": smoke,
        "historical_exposure": "Existing dataset already used in development; this is an internal benchmark, not a new external holdout.",
        "grouping": "exact decoded RGB plus conservatively co-assigned dHash<=2 candidate components; capture/session IDs unavailable; visual review still required",
        "counts": {role: int((frame.role == role).sum()) for role in ROLES},
        "manifest_sha256": hashes,
        "content_fingerprint": digest_object(frame[["sample_id", "sha256", "group_id", "role"]].to_dict("records")),
    }
    metadata["fingerprint"] = digest_object(metadata)
    write_json(output / "protocol.json", metadata)
    frame.groupby(["role", "brand", "class_name"]).size().rename("count").to_csv(output / "distribution.csv")
    return metadata


def load_protocol(directory: Path, check_images: bool = False) -> dict:
    metadata = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    saved_fingerprint = metadata.pop("fingerprint")
    if digest_object(metadata) != saved_fingerprint:
        raise ValueError("Protocol metadata was changed")
    metadata["fingerprint"] = saved_fingerprint
    inventory = pd.read_csv(directory / "inventory.csv")
    if digest_object(inventory[["sample_id", "sha256", "group_id", "role"]].to_dict("records")) != metadata["content_fingerprint"]:
        raise ValueError("Inventory changed after freezing")
    for name, expected in metadata["manifest_sha256"].items():
        if digest_file(directory / name) != expected:
            raise ValueError(f"Manifest changed after freezing: {name}")
    frames = [pd.read_csv(directory / f"{role}.csv").assign(role=role) for role in ROLES]
    frame = pd.concat(frames, ignore_index=True)
    verify_assignments(frame)
    if check_images:
        for row in frame.itertuples():
            if digest_file(Path(row.image_path)) != row.sha256:
                raise ValueError(f"Image content changed: {row.image_path}")
    return metadata


def coassign_candidates(frame: pd.DataFrame, pairs: list[dict]) -> pd.DataFrame:
    """Conservatively keep all candidate pairs together, including mixed labels.

    Similar backgrounds can be false matches. This only changes split grouping,
    never class labels or the number of images. Exact-pixel label conflicts still fail.
    """
    frame = frame.copy()
    parent = {sample: sample for sample in frame.sample_id}
    def find(sample):
        while parent[sample] != sample:
            parent[sample] = parent[parent[sample]]
            sample = parent[sample]
        return sample
    def union(a, b):
        first, second = find(a), find(b)
        parent[max(first, second)] = min(first, second)
    for _, part in frame.groupby("pixel_sha256"):
        ids = part.sample_id.tolist()
        for other in ids[1:]:
            union(ids[0], other)
    for pair in pairs:
        union(pair["sample_a"], pair["sample_b"])
    frame["group_id"] = frame.sample_id.map(find)
    return frame


def prepare(root: Path, output: Path, seed: int = 42, dhash_distance: int = 2) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}")
    frame, failures = scan_dataset(root, cache_root=output / "images")
    output.mkdir(parents=True, exist_ok=True)
    # Keep scan evidence even when preparation fails; no corrupt sample is silently skipped.
    frame.to_csv(output / "scan.csv", index=False)
    write_json(output / "scan_errors.json", failures)
    if failures:
        raise ValueError(f"{len(failures)} scan failures; inspect scan_errors.json")
    duplicates = frame[frame.duplicated("pixel_sha256", keep=False)]
    duplicates.to_csv(output / "exact_duplicates.csv", index=False)
    pairs = []
    rows = list(frame.itertuples())
    hashes = [int(row.dhash, 16) for row in rows]
    for i, first in enumerate(rows):
        for j in range(i + 1, len(rows)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= dhash_distance:
                pairs.append({"sample_a": first.sample_id, "sample_b": rows[j].sample_id,
                              "distance": distance, "same_class": first.label == rows[j].label})
    pd.DataFrame(pairs, columns=["sample_a", "sample_b", "distance", "same_class"]).to_csv(output / "near_duplicate_candidates.csv", index=False)
    frame = coassign_candidates(frame, pairs)
    assigned = split_groups(frame, seed)
    result = export_protocol(assigned, output / "iid", seed, root)
    write_json(output / "audit.json", {
        "images": len(frame), "brands": sorted(frame.brand.unique()), "classes": CLASSES,
        "exact_duplicate_images": len(duplicates), "near_duplicate_pairs": len(pairs),
        "near_duplicates_manually_verified": False, "iid": result,
    })
    return result
