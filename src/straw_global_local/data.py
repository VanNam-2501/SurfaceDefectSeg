"""Immutable three-way manifests and label-independent views from original images."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

CLASSES = ["good", "bent", "loose", "missing", "pierced"]
ROLES = ("train", "val", "test")
VARIANTS = tuple(f"B{i}" for i in range(7))
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
VIEW_SPEC = {"schema": 1, "sizes": [224, 512], "crop_fraction": .6,
             "crop_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
             "resize": "PIL bilinear", "exif": True}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def validate_frames(frames):
    combined = pd.concat([f.assign(role=role) for role, f in frames.items()], ignore_index=True)
    required = {"sample_id", "group_id", "image_path", "label", "class_name", "sha256"}
    if not required.issubset(combined.columns) or combined[list(required)].isna().any().any():
        raise ValueError("Incomplete manifest")
    if combined.sample_id.duplicated().any():
        raise ValueError("Sample IDs overlap")
    if combined.groupby("group_id").role.nunique().max() > 1:
        raise ValueError("Related group crosses data roles")
    if combined.groupby("sha256").role.nunique().max() > 1:
        raise ValueError("Exact file duplicate crosses roles")
    if not all(int(y) == y and 0 <= y < 5 for y in combined.label):
        raise ValueError("Expected integer labels 0..4")
    if any(CLASSES[int(row.label)] != row.class_name for row in combined.itertuples()):
        raise ValueError("Label/name mismatch")
    if any(set(f.label) != set(range(5)) for f in frames.values()):
        raise ValueError("All five classes required in each role")


def save_protocol(frames, output, context):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new protocol directory: {output}")
    validate_frames(frames)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for role in ROLES:
        frames[role] = frames[role].copy().assign(role=role)
        path = output / f"{role}.csv"
        frames[role].sort_values("sample_id").to_csv(path, index=False)
        hashes[path.name] = file_hash(path)
    metadata = {"schema": 1, "classes": CLASSES, "context": context,
                "counts": {r: len(f) for r, f in frames.items()}, "manifests": hashes}
    metadata["fingerprint"] = digest(metadata)
    write_json(output / "protocol.json", metadata)
    write_json(output / "review.json", {
        "labels_reviewed": False, "related_groups_reviewed": False,
        "dataset_rights_recorded": False, "related_work_reviewed": False,
        "reviewer": "", "notes": "Record actual review and evidence; never mark automatically.",
    })
    return metadata


def prepare(source, output, image_root=None, smoke_per_class=None):
    # Reuse the audited grouping/split, not its resized image cache.
    from src.straw_defect_cls.thesis_protocol import load_protocol as load_old
    source = Path(source)
    old = load_old(source)
    if old.get("held_out_brand"):
        raise ValueError("This experiment uses the IID group split, not LOBO")
    frames = {r: pd.read_csv(source / f"{r}.csv") for r in ("train", "val", "calibration", "test")}
    frames["train"] = pd.concat([frames["train"], frames.pop("calibration")], ignore_index=True)
    for role, frame in frames.items():
        if smoke_per_class:
            # Keep complete related groups selected by a deterministic per-class seed list.
            groups = set(frame.groupby("label", sort=True).head(smoke_per_class).group_id)
            frame = frame[frame.group_id.isin(groups)].copy()
        paths, hashes = [], []
        for row in frame.to_dict("records"):
            relative = Path(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source relative path")
            path = Path(image_root) / relative if image_root else Path(row.get("source_path", row["image_path"]))
            paths.append(str(path.resolve()))
            hashes.append(row.get("source_sha256", row["sha256"]))
        frame["image_path"], frame["sha256"] = paths, hashes
        frames[role] = frame
    return save_protocol(frames, output, {"source_protocol": old["fingerprint"],
        "split": "original train + calibration -> train; original val/test retained",
        "smoke_only": bool(smoke_per_class) or bool(old.get("smoke_only")),
        "historical_exposure": old["historical_exposure"],
        "grouping": "inherited heuristic groups; manual review remains required"})


def load_protocol(directory):
    directory = Path(directory)
    metadata = read_json(directory / "protocol.json")
    saved = metadata.pop("fingerprint")
    if digest(metadata) != saved:
        raise ValueError("Protocol metadata changed")
    metadata["fingerprint"] = saved
    frames = {}
    for role in ROLES:
        path = directory / f"{role}.csv"
        if file_hash(path) != metadata["manifests"][path.name]:
            raise ValueError("Frozen manifest changed")
        frames[role] = pd.read_csv(path)
    validate_frames(frames)
    return metadata, frames


def crop_boxes(width, height):
    cw, ch = max(1, round(width * .6)), max(1, round(height * .6))
    return [(x, y, x + cw, y + ch) for y in (0, height - ch) for x in (0, width - cw)]


def render_arrays(image, variant=None):
    """All view geometry depends only on pixels/size, never class or brand."""
    result = {}
    for size in (224, 512):
        if variant is None or variant == "B1" and size == 512 or variant != "B1" and size == 224:
            result[f"global{size}"] = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)).copy()
    if variant is None or variant not in ("B0", "B1"):
        boxes = crop_boxes(*image.size)
        result["local"] = np.stack([np.asarray(image.crop(box).resize((224, 224), Image.Resampling.BILINEAR)) for box in boxes])
        width, height = image.size
        result["coords"] = np.array([[(l+r)/(2*width), (t+b)/(2*height), (r-l)/width, (b-t)/height]
                                    for l,t,r,b in boxes], dtype=np.float32)
    return result


def create_cache(protocol_dir, cache_dir, workers=2):
    metadata, frames = load_protocol(protocol_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = {"protocol": metadata["fingerprint"], "views": VIEW_SPEC, "data_code": file_hash(__file__)}
    marker = cache_dir / "cache.json"
    if marker.exists() and read_json(marker)["spec"] != spec:
        raise ValueError("Cache source/view implementation changed; use a new directory")
    previous = read_json(marker).get("files", {}) if marker.exists() else {}
    rows = pd.concat(list(frames.values())).to_dict("records")
    def one(row):
        sample = str(row["sample_id"])
        target = cache_dir / f"{sample}.npz"
        if sample in previous and target.exists() and file_hash(target) == previous[sample]:
            return sample, previous[sample]
        source = Path(row["image_path"])
        if file_hash(source) != row["sha256"]:
            raise ValueError(f"Original image changed or resized cache supplied: {source}")
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError("Expected original image dimensions")
            values = render_arrays(image)
        temp = target.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temp.replace(target)
        return sample, file_hash(target)
    files = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (sample, checksum) in enumerate(pool.map(one, rows), 1):
            files[sample] = checksum
            if index % 100 == 0:
                print(f"Cached original-image views {index}/{len(rows)}", flush=True)
                partial = {"spec": spec, "files": files}
                partial["fingerprint"] = digest(partial)
                write_json(marker, partial)
    result = {"spec": spec, "files": files}
    result["fingerprint"] = digest(result)
    write_json(marker, result)
    return result


def load_cache(protocol_dir, cache_dir, verify=True):
    metadata, frames = load_protocol(protocol_dir)
    cached = read_json(Path(cache_dir) / "cache.json")
    fingerprint = cached.pop("fingerprint")
    if digest(cached) != fingerprint:
        raise ValueError("Cache metadata changed")
    cached["fingerprint"] = fingerprint
    expected = {"protocol": metadata["fingerprint"], "views": VIEW_SPEC, "data_code": file_hash(__file__)}
    if cached["spec"] != expected:
        raise ValueError("Cache/protocol/view mismatch")
    ids = set(pd.concat(list(frames.values())).sample_id.astype(str))
    if ids != set(cached["files"]):
        raise ValueError("Cache inventory mismatch")
    if verify:
        for sample, checksum in cached["files"].items():
            if file_hash(Path(cache_dir) / f"{sample}.npz") != checksum:
                raise ValueError(f"Cached image views changed: {sample}")
    return metadata, frames, cached


def tensor_views(arrays, variant, augment=False):
    size = 512 if variant == "B1" else 224
    global_image = torch.from_numpy(arrays[f"global{size}"].copy()).permute(2, 0, 1).float() / 255
    result = {"global": global_image}
    if variant not in ("B0", "B1"):
        result["local"] = torch.from_numpy(arrays["local"].copy()).permute(0, 3, 1, 2).float() / 255
        result["coords"] = torch.from_numpy(arrays["coords"].copy())
    if augment:
        brightness = .9 + .2 * float(torch.rand(()))
        result["global"] = (result["global"] * brightness).clamp(0, 1)
        if "local" in result:
            result["local"] = (result["local"] * brightness).clamp(0, 1)
        if torch.rand(()) < .5:
            result["global"] = result["global"].flip(-1)
            if "local" in result:
                # Exactly the corner geometry of cropping after a whole-image flip.
                order = torch.tensor([1, 0, 3, 2])
                result["local"] = result["local"][order].flip(-1)
                coords = result["coords"][order].clone()
                coords[:, 0] = 1 - coords[:, 0]
                result["coords"] = coords
    result["global"] = (result["global"] - MEAN) / STD
    if "local" in result:
        result["local"] = (result["local"] - MEAN) / STD
    return result


class ViewDataset(Dataset):
    def __init__(self, frame, cache_dir, variant, augment=False):
        if variant not in VARIANTS:
            raise ValueError("Unknown model variant")
        self.frame, self.cache_dir, self.variant, self.augment = frame.reset_index(drop=True), Path(cache_dir), variant, augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(self.cache_dir / f"{row.sample_id}.npz", allow_pickle=False) as cached:
            keys = ["global512" if self.variant == "B1" else "global224"]
            if self.variant not in ("B0", "B1"):
                keys += ["local", "coords"]
            arrays = {key: cached[key] for key in keys}
        return tensor_views(arrays, self.variant, self.augment), int(row.label), str(row.sample_id)
