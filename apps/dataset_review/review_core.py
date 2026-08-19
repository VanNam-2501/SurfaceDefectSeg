"""Core data, review, and export services for the dataset review tool.

The source dataset is treated as immutable. Reviews, edited masks, audit events,
and exports live under ``apps/dataset_review`` unless explicitly configured.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


DECISIONS = {
    "approved",
    "acceptable_mark",
    "relabel_good",
    "relabel_defect",
    "fix_mask",
    "uncertain",
    "exclude",
}

ISSUE_TAGS = {
    "hidden_defect",
    "false_alarm",
    "mask_misaligned",
    "mask_incomplete",
    "border_issue",
    "annotation_ambiguous",
    "duplicate",
    "hard_negative",
    "lighting",
    "texture",
    "model_disagreement",
    "model_mask_accepted",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(value: str, default: str = "review_export") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return text[:80] or default


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "" or pd.isna(value):
        return default
    return int(float(value))


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "" or pd.isna(value):
        return default
    return float(value)


def binary_mask(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = np.any(array != 0, axis=2)
    elif array.ndim == 2:
        array = array != 0
    else:
        raise ValueError(f"Unsupported mask shape {array.shape} at {path}")
    result = array.astype(np.uint8)
    if expected_size is not None and result.shape != expected_size:
        raise ValueError(
            f"Mask size {result.shape} does not match image size {expected_size}"
        )
    return result


def mask_statistics(mask: np.ndarray) -> dict[str, Any]:
    h, w = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    touches_border = False
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        border = x == 0 or y == 0 or x + width >= w or y + height >= h
        touches_border = touches_border or border
        components.append(
            {
                "area": area,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "touches_border": border,
            }
        )
    pixels = int(mask.sum())
    single_pixel_components = sum(item["area"] == 1 for item in components)
    return {
        "height": h,
        "width": w,
        "pixels": pixels,
        "ratio": pixels / float(h * w) if h and w else 0.0,
        "component_count": len(components),
        "smallest_component": min((item["area"] for item in components), default=0),
        "largest_component": max((item["area"] for item in components), default=0),
        "single_pixel_components": single_pixel_components,
        "touches_border": touches_border,
        "components": components,
    }


def remove_singleton_spray(
    mask: np.ndarray,
    minimum_singletons: int = 32,
) -> tuple[np.ndarray, int]:
    """Remove a clearly invalid spray of isolated one-pixel components.

    A small number of single pixels is left untouched. The cleanup only
    activates for the systematic editor artifact observed in saved masks,
    where dozens or hundreds of isolated pixels are scattered over the image.
    """
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary, 0
    singleton_ids = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] == 1) + 1
    if len(singleton_ids) < minimum_singletons:
        return binary, 0
    cleaned = binary.copy()
    cleaned[np.isin(labels, singleton_ids)] = 0
    return cleaned, int(len(singleton_ids))


@dataclass(frozen=True)
class ReviewConfig:
    tool_dir: Path
    dataset_root: Path
    results_roots: tuple[Path, ...] = ()
    database_path: Path | None = None
    edits_dir: Path | None = None
    exports_dir: Path | None = None

    def normalized(self) -> "ReviewConfig":
        tool_dir = self.tool_dir.resolve()
        return ReviewConfig(
            tool_dir=tool_dir,
            dataset_root=self.dataset_root.resolve(),
            results_roots=tuple(path.resolve() for path in self.results_roots),
            database_path=(self.database_path or tool_dir / "review_state.sqlite3").resolve(),
            edits_dir=(self.edits_dir or tool_dir / "edits").resolve(),
            exports_dir=(self.exports_dir or tool_dir / "exports").resolve(),
        )


class ReviewStore:
    """Small SQLite store with an append-only event history."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def connection(self):
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    image_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL,
                    corrected_label INTEGER,
                    corrected_group TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    note TEXT NOT NULL DEFAULT '',
                    hard_negative INTEGER NOT NULL DEFAULT 0,
                    excluded INTEGER NOT NULL DEFAULT 0,
                    edited_mask_path TEXT NOT NULL DEFAULT '',
                    reviewer TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    before_json TEXT NOT NULL,
                    after_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_events_image_id
                    ON review_events(image_id, id);
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["tags"] = json.loads(result.pop("tags_json") or "[]")
        result["hard_negative"] = bool(result["hard_negative"])
        result["excluded"] = bool(result["excluded"])
        return result

    def get(self, image_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE image_id = ?", (image_id,)
            ).fetchone()
        return self._decode(row)

    def all(self) -> dict[str, dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM reviews").fetchall()
        return {str(row["image_id"]): self._decode(row) or {} for row in rows}

    def save(self, image_id: str, payload: dict[str, Any], action: str = "review") -> dict[str, Any]:
        decision = str(payload.get("decision", "")).strip()
        if decision not in DECISIONS:
            raise ValueError(f"Unsupported decision: {decision}")
        corrected_label = payload.get("corrected_label")
        if corrected_label not in (None, 0, 1):
            raise ValueError("corrected_label must be null, 0, or 1")
        tags = sorted({str(tag) for tag in payload.get("tags", []) if str(tag)})
        invalid_tags = sorted(set(tags) - ISSUE_TAGS)
        if invalid_tags:
            raise ValueError(f"Unsupported tags: {invalid_tags}")

        with self._lock:
            before = self.get(image_id)
            edited_mask_path = str(
                payload.get("edited_mask_path")
                or (before or {}).get("edited_mask_path", "")
            )
            current = {
                "image_id": image_id,
                "decision": decision,
                "corrected_label": corrected_label,
                "corrected_group": str(payload.get("corrected_group", "")).strip(),
                "tags": tags,
                "note": str(payload.get("note", "")).strip(),
                "hard_negative": bool(payload.get("hard_negative", False)),
                "excluded": bool(payload.get("excluded", False)) or decision == "exclude",
                "edited_mask_path": edited_mask_path,
                "reviewer": str(payload.get("reviewer", "")).strip(),
                "updated_at": utc_now(),
            }
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO reviews (
                        image_id, decision, corrected_label, corrected_group,
                        tags_json, note, hard_negative, excluded,
                        edited_mask_path, reviewer, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                        decision=excluded.decision,
                        corrected_label=excluded.corrected_label,
                        corrected_group=excluded.corrected_group,
                        tags_json=excluded.tags_json,
                        note=excluded.note,
                        hard_negative=excluded.hard_negative,
                        excluded=excluded.excluded,
                        edited_mask_path=excluded.edited_mask_path,
                        reviewer=excluded.reviewer,
                        updated_at=excluded.updated_at
                    """,
                    (
                        image_id,
                        current["decision"],
                        current["corrected_label"],
                        current["corrected_group"],
                        json.dumps(current["tags"], ensure_ascii=False),
                        current["note"],
                        int(current["hard_negative"]),
                        int(current["excluded"]),
                        current["edited_mask_path"],
                        current["reviewer"],
                        current["updated_at"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO review_events (
                        image_id, action, before_json, after_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        image_id,
                        action,
                        json.dumps(before or {}, ensure_ascii=False),
                        json.dumps(current, ensure_ascii=False),
                        current["updated_at"],
                    ),
                )
        return current

    def set_mask_path(self, image_id: str, mask_path: str, fallback: dict[str, Any]) -> dict[str, Any]:
        payload = dict(self.get(image_id) or fallback)
        payload["edited_mask_path"] = mask_path
        tags = set(payload.get("tags", []))
        tags.add("mask_incomplete")
        payload["tags"] = sorted(tags)
        if payload.get("decision") not in DECISIONS:
            payload["decision"] = "fix_mask"
        return self.save(image_id, payload, action="mask_saved")

    def events(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM review_events ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]


class DatasetReviewService:
    def __init__(self, config: ReviewConfig) -> None:
        self.config = config.normalized()
        if not self.config.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.config.dataset_root}")
        assert self.config.database_path is not None
        assert self.config.edits_dir is not None
        assert self.config.exports_dir is not None
        self.config.edits_dir.mkdir(parents=True, exist_ok=True)
        self.config.exports_dir.mkdir(parents=True, exist_ok=True)
        self.store = ReviewStore(self.config.database_path)
        self.records = self._load_records()
        self.models: list[str] = []
        # Qualitative previews are model-specific. The same image can have a
        # different preview for UNet, SegFormer, Mamba, ...
        self.qualitative_assets: dict[tuple[str, str], Path] = {}
        self.prediction_assets: dict[tuple[str, str], Path] = {}
        self._load_predictions()
        self._diagnostics_cache: dict[str, dict[str, Any]] = {}

    def _contained_source(self, raw_path: str) -> Path:
        path = Path(raw_path)
        resolved = path.resolve() if path.is_absolute() else (self.config.dataset_root / path).resolve()
        if resolved != self.config.dataset_root and self.config.dataset_root not in resolved.parents:
            raise ValueError(f"Source path escapes dataset root: {raw_path}")
        return resolved

    def _load_records(self) -> dict[str, dict[str, Any]]:
        split_dir = self.config.dataset_root / "dataset_audit" / "splits"
        records: dict[str, dict[str, Any]] = {}
        for split in ("train", "val", "test"):
            path = split_dir / f"{split}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing split CSV: {path}")
            frame = pd.read_csv(path).fillna("")
            required = {"image_id", "image_path", "mask_path", "label"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in frame.to_dict("records"):
                image_id = str(row["image_id"])
                if image_id in records:
                    raise ValueError(f"Duplicate image_id across splits: {image_id}")
                label = as_int(row["label"])
                image_path = self._contained_source(str(row["image_path"]))
                mask_text = str(row.get("mask_path", "")).strip()
                mask_path = self._contained_source(mask_text) if mask_text else None
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image: {image_path}")
                if label == 1 and (mask_path is None or not mask_path.is_file()):
                    raise FileNotFoundError(f"Missing defect mask for {image_id}: {mask_path}")
                height = as_int(row.get("height"))
                width = as_int(row.get("width"))
                if not height or not width:
                    with Image.open(image_path) as image:
                        width, height = image.size
                records[image_id] = {
                    "image_id": image_id,
                    "split": split,
                    "label": label,
                    "label_name": "Defect" if label else "Good",
                    "defect_group": str(row.get("defect_group", "good") or "good"),
                    "source_split": str(row.get("source_split", "")),
                    "height": height,
                    "width": width,
                    "defect_pixels": as_int(row.get("defect_pixels")),
                    "defect_ratio": as_float(row.get("defect_ratio")),
                    "num_components": as_int(row.get("num_components")),
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "source_row": dict(row),
                    "predictions": {},
                    "candidate_reasons": [],
                }
        return records

    @staticmethod
    def _model_name(results_root: Path, csv_path: Path) -> str:
        relative = csv_path.relative_to(results_root)
        parts = list(relative.parts[:-2])
        return "/".join(parts[-2:] or [results_root.name])

    def _load_predictions(self) -> None:
        discovered_models: set[str] = set()
        model_run_dirs: dict[str, Path] = {}
        suffix_map = {image_id.rsplit("_", 1)[-1]: image_id for image_id in self.records}
        for results_root in self.config.results_roots:
            if not results_root.exists():
                continue
            csv_paths = list(results_root.rglob("per_image_metrics.csv"))
            for csv_path in csv_paths:
                split = csv_path.parent.name.lower()
                if split not in {"train", "validation", "val", "test"}:
                    continue
                split = "val" if split == "validation" else split
                model = self._model_name(results_root, csv_path)
                discovered_models.add(model)
                model_run_dirs.setdefault(model, csv_path.parent.parent)
                frame = pd.read_csv(csv_path).fillna("")
                for row in frame.to_dict("records"):
                    image_id = str(row.get("image_id", ""))
                    record = self.records.get(image_id)
                    if record is None or record["split"] != split:
                        continue
                    record["predictions"][model] = {
                        "image_score": as_float(row.get("image_score")),
                        "image_pred": as_int(row.get("image_pred")),
                        "threshold": as_float(row.get("threshold"), 0.5),
                        "predicted_positive_pixels": as_int(row.get("predicted_positive_pixels")),
                        "pixel_tp": as_int(row.get("pixel_tp")),
                        "pixel_fp": as_int(row.get("pixel_fp")),
                        "pixel_fn": as_int(row.get("pixel_fn")),
                        "positive_dice": (
                            None if row.get("positive_dice", "") == "" else as_float(row.get("positive_dice"))
                        ),
                    }

                qualitative_dir = csv_path.parent / "qualitative"
                if qualitative_dir.is_dir():
                    for asset in qualitative_dir.glob("*.png"):
                        suffix = asset.stem.rsplit("_", 1)[-1]
                        image_id = suffix_map.get(suffix)
                        if image_id and asset.stem.endswith(image_id):
                            self.qualitative_assets.setdefault((model, image_id), asset)

        # Full-dataset prediction exports supplement validation/test metrics and
        # also add model results for train images.
        for model, run_dir in model_run_dirs.items():
            predictions_root = run_dir / "predictions"
            for split in ("train", "val", "test"):
                manifest_path = predictions_root / split / "manifest.csv"
                if not manifest_path.is_file():
                    continue
                frame = pd.read_csv(manifest_path).fillna("")
                for row in frame.to_dict("records"):
                    image_id = str(row.get("image_id", ""))
                    record = self.records.get(image_id)
                    if record is None or record["split"] != split:
                        continue
                    probability_value = str(row.get("probability_path", "")).strip()
                    probability_path = (
                        (run_dir / probability_value).resolve()
                        if probability_value
                        else predictions_root / split / "probability" / f"{image_id}.png"
                    )
                    run_resolved = run_dir.resolve()
                    if probability_path.is_file() and (
                        probability_path == run_resolved or run_resolved in probability_path.parents
                    ):
                        self.prediction_assets[(model, image_id)] = probability_path

                    current = record["predictions"].get(model, {})
                    current.update(
                        {
                            "image_score": as_float(row.get("image_score")),
                            "image_pred": as_int(row.get("image_pred")),
                            "threshold": as_float(row.get("threshold"), 0.5),
                            "predicted_positive_pixels": as_int(row.get("predicted_positive_pixels")),
                            "pixel_tp": (
                                None if row.get("pixel_tp", "") == "" else as_int(row.get("pixel_tp"))
                            ),
                            "pixel_fp": (
                                None if row.get("pixel_fp", "") == "" else as_int(row.get("pixel_fp"))
                            ),
                            "pixel_fn": (
                                None if row.get("pixel_fn", "") == "" else as_int(row.get("pixel_fn"))
                            ),
                            "positive_dice": (
                                None if row.get("positive_dice", "") == "" else as_float(row.get("positive_dice"))
                            ),
                        }
                    )
                    record["predictions"][model] = current

        # Portable caches may be stored independently from the original model
        # run (for example under artifacts/experiments/decision/predictions).
        # Discover them from metadata.json instead of requiring a neighboring
        # per_image_metrics.csv. The canonical local PNG path is preferred so
        # manifests remain portable even when they contain a historical path.
        for results_root in self.config.results_roots:
            if not results_root.exists():
                continue
            for metadata_path in results_root.rglob("metadata.json"):
                predictions_root = metadata_path.parent.resolve()
                if not any((predictions_root / split / "manifest.csv").is_file() for split in ("train", "val", "test")):
                    continue
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
                model = str(metadata.get("model") or predictions_root.name).strip()
                if not model:
                    continue
                discovered_models.add(model)
                for split in ("train", "val", "test"):
                    manifest_path = predictions_root / split / "manifest.csv"
                    if not manifest_path.is_file():
                        continue
                    frame = pd.read_csv(manifest_path).fillna("")
                    for row in frame.to_dict("records"):
                        image_id = str(row.get("image_id", ""))
                        record = self.records.get(image_id)
                        if record is None or record["split"] != split:
                            continue
                        probability_path = (
                            predictions_root / split / "probability" / f"{image_id}.png"
                        ).resolve()
                        if probability_path.is_file() and predictions_root in probability_path.parents:
                            self.prediction_assets[(model, image_id)] = probability_path
                        current = record["predictions"].get(model, {})
                        current.update(
                            {
                                "image_score": as_float(row.get("image_score")),
                                "image_pred": as_int(row.get("image_pred")),
                                "threshold": as_float(row.get("threshold"), 0.5),
                                "predicted_positive_pixels": as_int(row.get("predicted_positive_pixels")),
                                "pixel_tp": None if row.get("pixel_tp", "") == "" else as_int(row.get("pixel_tp")),
                                "pixel_fp": None if row.get("pixel_fp", "") == "" else as_int(row.get("pixel_fp")),
                                "pixel_fn": None if row.get("pixel_fn", "") == "" else as_int(row.get("pixel_fn")),
                                "positive_dice": None if row.get("positive_dice", "") == "" else as_float(row.get("positive_dice")),
                            }
                        )
                        record["predictions"][model] = current

        self.models = sorted(discovered_models)
        for record in self.records.values():
            reasons: set[str] = set()
            predictions = list(record["predictions"].values())
            disagreement = len({item["image_pred"] for item in predictions}) > 1
            for prediction in predictions:
                model_reasons: set[str] = set()
                if record["label"] == 0:
                    if prediction["image_pred"]:
                        model_reasons.add("false_positive")
                    if prediction["image_score"] >= 0.99:
                        model_reasons.add("high_score_good")
                else:
                    if not prediction["image_pred"]:
                        model_reasons.add("false_negative")
                    if prediction.get("pixel_tp") is not None and prediction["pixel_tp"] == 0:
                        model_reasons.add("zero_overlap")
                if disagreement:
                    model_reasons.add("model_disagreement")
                prediction["candidate_reasons"] = sorted(model_reasons)
                reasons.update(model_reasons)
            record["candidate_reasons"] = sorted(reasons)

    def current_mask_path(self, image_id: str) -> Path | None:
        record = self.get_record(image_id)
        review = self.store.get(image_id)
        if review and review.get("edited_mask_path"):
            path = Path(review["edited_mask_path"]).resolve()
            if self.config.edits_dir == path or self.config.edits_dir in path.parents:
                if path.is_file():
                    return path
        return record["mask_path"]

    def get_record(self, image_id: str) -> dict[str, Any]:
        try:
            return self.records[image_id]
        except KeyError as exc:
            raise KeyError(f"Unknown image_id: {image_id}") from exc

    def diagnostics(self, image_id: str, refresh: bool = False) -> dict[str, Any]:
        if not refresh and image_id in self._diagnostics_cache:
            return self._diagnostics_cache[image_id]
        record = self.get_record(image_id)
        with Image.open(record["image_path"]) as image:
            image_size = (image.height, image.width)
        mask_path = self.current_mask_path(image_id)
        if mask_path is None:
            mask = np.zeros(image_size, dtype=np.uint8)
        else:
            mask = binary_mask(mask_path)
        size_match = mask.shape == image_size
        stats = mask_statistics(mask) if size_match else {
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "pixels": int(mask.sum()),
            "ratio": 0.0,
            "component_count": 0,
            "smallest_component": 0,
            "largest_component": 0,
            "single_pixel_components": 0,
            "touches_border": False,
            "components": [],
        }
        issues = []
        if not size_match:
            issues.append("mask_size_mismatch")
        if record["label"] == 1 and stats["pixels"] == 0:
            issues.append("empty_defect_mask")
        if record["label"] == 0 and stats["pixels"] > 0:
            issues.append("nonempty_good_mask")
        if stats["touches_border"]:
            issues.append("mask_touches_border")
        if stats["single_pixel_components"] >= 32:
            issues.append("singleton_speckle_noise")
        result = {
            "image_height": image_size[0],
            "image_width": image_size[1],
            "mask_size_match": size_match,
            "mask": stats,
            "issues": issues,
        }
        self._diagnostics_cache[image_id] = result
        return result

    def _public_record(self, record: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
        predictions = record["predictions"]
        max_score = max((item["image_score"] for item in predictions.values()), default=None)
        return {
            "image_id": record["image_id"],
            "split": record["split"],
            "source_split": record["source_split"],
            "label": record["label"],
            "label_name": record["label_name"],
            "defect_group": record["defect_group"],
            "height": record["height"],
            "width": record["width"],
            "defect_pixels": record["defect_pixels"],
            "defect_ratio": record["defect_ratio"],
            "num_components": record["num_components"],
            "predictions": predictions,
            "max_score": max_score,
            "candidate_reasons": record["candidate_reasons"],
            "qualitative_models": sorted(
                model
                for model in predictions
                if (model, record["image_id"]) in self.qualitative_assets
            ),
            "has_qualitative": any(
                (model, record["image_id"]) in self.qualitative_assets
                for model in predictions
            ),
            "prediction_models": sorted(
                model
                for model in predictions
                if (model, record["image_id"]) in self.prediction_assets
            ),
            "review": review,
        }

    def details(self, image_id: str) -> dict[str, Any]:
        record = self.get_record(image_id)
        review = self.store.get(image_id)
        result = self._public_record(record, review)
        result["diagnostics"] = self.diagnostics(image_id)
        result["source_image_path"] = str(record["image_path"])
        result["source_mask_path"] = str(record["mask_path"] or "")
        return result

    def summary(self) -> dict[str, Any]:
        reviews = self.store.all()
        decisions: dict[str, int] = {}
        for review in reviews.values():
            decisions[review["decision"]] = decisions.get(review["decision"], 0) + 1
        candidate_count = sum(bool(record["candidate_reasons"]) for record in self.records.values())
        return {
            "total": len(self.records),
            "reviewed": len(reviews),
            "remaining": len(self.records) - len(reviews),
            "candidates": candidate_count,
            "hard_negatives": sum(bool(item["hard_negative"]) for item in reviews.values()),
            "excluded": sum(bool(item["excluded"]) for item in reviews.values()),
            "decisions": decisions,
        }

    def list_items(
        self,
        *,
        split: str = "all",
        label: str = "all",
        group: str = "all",
        review_status: str = "all",
        decision: str = "all",
        candidate: str = "all",
        model: str = "all",
        min_score: float | None = None,
        search: str = "",
        sort: str = "priority",
        descending: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        reviews = self.store.all()
        query = search.strip().lower()
        values = []
        for record in self.records.values():
            review = reviews.get(record["image_id"])
            if split != "all" and record["split"] != split:
                continue
            if label != "all" and record["label"] != int(label):
                continue
            if group != "all" and record["defect_group"] != group:
                continue
            if review_status == "reviewed" and review is None:
                continue
            if review_status == "unreviewed" and review is not None:
                continue
            if decision != "all" and (review or {}).get("decision") != decision:
                continue
            if model != "all" and model not in record["predictions"]:
                continue
            active_reasons = (
                record["predictions"][model].get("candidate_reasons", [])
                if model != "all"
                else record["candidate_reasons"]
            )
            if candidate == "any" and not active_reasons:
                continue
            if candidate == "none" and active_reasons:
                continue
            if candidate not in {"all", "any", "none"} and candidate not in active_reasons:
                continue
            if min_score is not None:
                scores = (
                    [record["predictions"][model]["image_score"]]
                    if model != "all" and model in record["predictions"]
                    else [item["image_score"] for item in record["predictions"].values()]
                )
                if not scores or max(scores) < min_score:
                    continue
            if query and query not in record["image_id"].lower() and query not in record["defect_group"].lower():
                continue
            public = self._public_record(record, review)
            if model != "all":
                public["active_model"] = model
                public["candidate_reasons"] = active_reasons
                public["max_score"] = record["predictions"][model]["image_score"]
            values.append(public)

        def key(item: dict[str, Any]) -> Any:
            if sort == "score":
                return -1.0 if item["max_score"] is None else item["max_score"]
            if sort == "area":
                return max(
                    (p["predicted_positive_pixels"] for p in item["predictions"].values()),
                    default=0,
                )
            if sort == "id":
                return item["image_id"]
            if sort == "updated":
                return (item["review"] or {}).get("updated_at", "")
            reason_weight = {
                "high_score_good": 5,
                "zero_overlap": 4,
                "model_disagreement": 3,
                "false_positive": 2,
                "false_negative": 2,
            }
            return (
                max((reason_weight.get(value, 0) for value in item["candidate_reasons"]), default=0),
                -1.0 if item["max_score"] is None else item["max_score"],
            )

        values.sort(key=key, reverse=descending)
        total = len(values)
        offset = max(0, min(offset, max(0, total - 1))) if total else 0
        limit = max(1, min(limit, 500))
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": values[offset : offset + limit],
        }

    def save_review(self, image_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.get_record(image_id)
        decision = payload.get("decision")
        if decision in {"relabel_good", "acceptable_mark"}:
            payload["corrected_label"] = 0
        elif decision == "relabel_defect":
            payload["corrected_label"] = 1
        elif payload.get("corrected_label") == "":
            payload["corrected_label"] = None
        return self.store.save(image_id, payload)

    def save_mask_data_url(self, image_id: str, data_url: str) -> dict[str, Any]:
        record = self.get_record(image_id)
        match = re.fullmatch(r"data:image/png;base64,([A-Za-z0-9+/=\r\n]+)", data_url)
        if not match:
            raise ValueError("Expected a base64 PNG data URL")
        raw = base64.b64decode(match.group(1), validate=True)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("Mask PNG is too large")
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image)
        if array.ndim == 3:
            array = np.any(array[..., :3] != 0, axis=2)
        elif array.ndim == 2:
            array = array != 0
        else:
            raise ValueError("Unsupported mask image")
        expected = (record["height"], record["width"])
        if array.shape != expected:
            raise ValueError(f"Edited mask size {array.shape} must equal image size {expected}")
        array, removed_noise_pixels = remove_singleton_spray(array)
        destination = self._save_binary_mask(image_id, array)
        self._diagnostics_cache.pop(image_id, None)
        fallback = {
            "decision": "fix_mask",
            "corrected_label": None,
            "corrected_group": "",
            "tags": ["mask_incomplete"],
            "note": "",
            "hard_negative": False,
            "excluded": False,
            "reviewer": "",
        }
        review = self.store.set_mask_path(image_id, str(destination), fallback)
        return {
            "review": review,
            "diagnostics": self.diagnostics(image_id, refresh=True),
            "removed_noise_pixels": removed_noise_pixels,
        }

    def _save_binary_mask(self, image_id: str, array: np.ndarray, suffix: str = "") -> Path:
        record = self.get_record(image_id)
        expected = (record["height"], record["width"])
        if array.shape != expected:
            raise ValueError(f"Mask size {array.shape} must equal image size {expected}")
        mask = (array.astype(bool).astype(np.uint8) * 255)
        mask_dir = self.config.edits_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        destination = (mask_dir / f"{safe_slug(image_id)}{suffix}.png").resolve()
        if self.config.edits_dir not in destination.parents:
            raise ValueError("Invalid edited mask path")
        Image.fromarray(mask, mode="L").save(destination, format="PNG", optimize=True)
        return destination

    def apply_prediction_mask(self, image_id: str, model: str) -> dict[str, Any]:
        """Use the selected model's stored binary prediction as an editable GT mask."""
        record = self.get_record(image_id)
        asset = self.prediction_assets.get((model, image_id))
        prediction = record["predictions"].get(model)
        if asset is None or not asset.is_file() or prediction is None:
            raise ValueError(f"No stored prediction map for model '{model}' and this sample")
        with Image.open(asset) as image:
            encoded = np.asarray(image)
        if encoded.ndim == 3:
            encoded = encoded[..., 0]
        scale = 65535.0 if encoded.dtype == np.uint16 else 255.0
        probability = np.clip(encoded.astype(np.float32) / scale, 0.0, 1.0)
        binary = probability >= float(prediction.get("threshold", 0.5))
        if not np.any(binary):
            raise ValueError("The selected model predicts no defect pixels, so it cannot become a defect mask")

        destination = self._save_binary_mask(
            image_id,
            binary,
            suffix=f"__model_{safe_slug(model, default='model')}",
        )
        self._diagnostics_cache.pop(image_id, None)
        previous = self.store.get(image_id) or {}
        tags = set(previous.get("tags", []))
        tags.add("model_mask_accepted")
        note = str(previous.get("note", "")).strip()
        provenance = (
            f"Mask accepted from {model} at threshold {float(prediction.get('threshold', 0.5)):.4f}."
        )
        if provenance not in note:
            note = f"{note}\n{provenance}".strip()
        review_payload = {
            "decision": "fix_mask",
            "corrected_label": 1,
            "corrected_group": (
                str(previous.get("corrected_group", "")).strip()
                or ("unclassified" if record["label"] == 0 else "")
            ),
            "tags": sorted(tags),
            "note": note,
            "hard_negative": False,
            "excluded": False,
            "edited_mask_path": str(destination),
            "reviewer": str(previous.get("reviewer", "")),
        }
        review = self.store.save(image_id, review_payload, action="model_mask_applied")
        return {
            "review": review,
            "diagnostics": self.diagnostics(image_id, refresh=True),
            "model": model,
            "mask_pixels": int(np.count_nonzero(binary)),
        }

    def _final_row(self, record: dict[str, Any], review: dict[str, Any] | None, export_dir: Path) -> tuple[dict[str, Any], list[str]]:
        row = dict(record["source_row"])
        issues: list[str] = []
        corrected_label = None if review is None else review.get("corrected_label")
        final_label = record["label"] if corrected_label is None else int(corrected_label)
        final_group = (
            str((review or {}).get("corrected_group", "")).strip()
            or ("good" if final_label == 0 else record["defect_group"])
        )
        row["label"] = final_label
        row["label_name"] = "Defect" if final_label else "Good"
        row["defect_group"] = final_group

        if final_label == 0:
            row["mask_path"] = ""
            row["defect_pixels"] = 0
            row["defect_ratio"] = 0.0
            row["num_components"] = 0
        else:
            mask_path = self.current_mask_path(record["image_id"])
            if mask_path is None:
                issues.append("defect_without_mask")
            else:
                try:
                    mask = binary_mask(mask_path, (record["height"], record["width"]))
                    stats = mask_statistics(mask)
                except Exception as exc:  # export must report, not partially guess
                    issues.append(f"invalid_mask:{exc}")
                else:
                    if stats["pixels"] == 0:
                        issues.append("empty_defect_mask")
                    elif stats["single_pixel_components"] >= 32:
                        issues.append("singleton_speckle_noise")
                    elif review and review.get("edited_mask_path"):
                        destination = export_dir / "corrected_masks" / safe_slug(final_group) / f"{safe_slug(record['image_id'])}.png"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(mask_path, destination)
                        row["mask_path"] = str(destination.resolve())
                    row["defect_pixels"] = stats["pixels"]
                    row["defect_ratio"] = stats["ratio"]
                    row["num_components"] = stats["component_count"]
        return row, issues

    @staticmethod
    def _write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> None:
        rows = list(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows and not columns:
            path.write_text("", encoding="utf-8-sig")
            return
        fieldnames = columns or list(rows[0].keys())
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _hardlink_or_copy(source: Path, destination: Path) -> str:
        """Materialize a source image without ever modifying the source file."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"

    def _write_training_dataset(
        self,
        export_dir: Path,
        records_by_split: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
        columns: list[str],
    ) -> dict[str, Any]:
        """Create a portable, train-ready dataset alongside the audit export.

        Unlike the legacy manifests, every path in this bundle is relative to
        ``training_dataset`` and every record has an explicit 0/255 PNG mask.
        Good samples receive a zero mask; reviewed samples use their current
        edited mask when one exists.
        """
        dataset_dir = export_dir / "training_dataset"
        all_rows: list[dict[str, Any]] = []
        split_counts: dict[str, int] = {}
        materialization = {"hardlink": 0, "copy": 0, "masks_written": 0}

        for split, pairs in records_by_split.items():
            rows: list[dict[str, Any]] = []
            for record, final_row in pairs:
                row = dict(final_row)
                image_suffix = record["image_path"].suffix.lower() or ".png"
                image_relative = Path("images") / split / f"{safe_slug(record['image_id'])}{image_suffix}"
                image_destination = dataset_dir / image_relative
                mode = self._hardlink_or_copy(record["image_path"], image_destination)
                materialization[mode] += 1

                mask_relative = Path("masks") / split / f"{safe_slug(record['image_id'])}.png"
                mask_destination = dataset_dir / mask_relative
                if int(row["label"]) == 1:
                    source_mask = self.current_mask_path(record["image_id"])
                    if source_mask is None:
                        raise ValueError(f"Cannot package defect without mask: {record['image_id']}")
                    mask = binary_mask(source_mask, (record["height"], record["width"]))
                else:
                    mask = np.zeros((record["height"], record["width"]), dtype=np.uint8)
                mask_destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                    mask_destination, format="PNG", optimize=True
                )
                materialization["masks_written"] += 1

                row["image_path"] = image_relative.as_posix()
                row["mask_path"] = mask_relative.as_posix()
                rows.append(row)
                all_rows.append(row)

            self._write_csv(dataset_dir / "dataset_audit" / "splits" / f"{split}.csv", rows, columns)
            split_counts[split] = len(rows)

        self._write_csv(dataset_dir / "cleaned_manifest.csv", all_rows, columns)
        info = {
            "dataset_root": str(dataset_dir.resolve()),
            "manifest_dir": "dataset_audit/splits",
            "split_manifests": {
                split: f"dataset_audit/splits/{split}.csv" for split in records_by_split
            },
            "path_contract": "image_path and mask_path are relative to this training_dataset directory",
            "mask_encoding": "8-bit PNG: 0 for background, 255 for defect",
            "good_samples_have_blank_mask": True,
            "split_counts": split_counts,
            "materialization": materialization,
        }
        (dataset_dir / "dataset_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (dataset_dir / "TRAINING_READY.md").write_text(
            "# Training-ready dataset\n\n"
            "Use this `training_dataset` folder as the dataset root for training.\n\n"
            "- Train manifest: `dataset_audit/splits/train.csv`\n"
            "- Validation manifest: `dataset_audit/splits/val.csv`\n"
            "- Test manifest: `dataset_audit/splits/test.csv`\n\n"
            "All `image_path` and `mask_path` values are relative to this folder. "
            "Every sample has a PNG mask: Good samples have an all-black mask; "
            "Defect samples use the final reviewed mask.\n",
            encoding="utf-8",
        )
        return {
            "path": str(dataset_dir.resolve()),
            "split_counts": split_counts,
            "materialization": materialization,
        }

    def export(self, name: str = "") -> dict[str, Any]:
        reviews = self.store.all()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = safe_slug(name, f"review_export_{timestamp}")
        export_dir = (self.config.exports_dir / export_name).resolve()
        if self.config.exports_dir not in export_dir.parents:
            raise ValueError("Invalid export path")
        if export_dir.exists():
            export_dir = (self.config.exports_dir / f"{export_name}_{timestamp}").resolve()
        export_dir.mkdir(parents=True)

        clean_rows: list[dict[str, Any]] = []
        unresolved_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        hard_negative_rows: list[dict[str, Any]] = []
        split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        package_records: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
            "train": [], "val": [], "test": []
        }

        for record in self.records.values():
            review = reviews.get(record["image_id"])
            final_row, issues = self._final_row(record, review, export_dir)
            excluded = bool((review or {}).get("excluded")) or (review or {}).get("decision") in {"exclude", "uncertain"}
            if issues:
                unresolved_rows.append(
                    {
                        "image_id": record["image_id"],
                        "split": record["split"],
                        "issues": "|".join(issues),
                        "decision": (review or {}).get("decision", "unreviewed"),
                        "note": (review or {}).get("note", ""),
                    }
                )
                excluded = True
            if not excluded:
                clean_rows.append(final_row)
                split_rows[record["split"]].append(final_row)
                package_records[record["split"]].append((record, final_row))
            if review:
                audit_rows.append(
                    {
                        "image_id": record["image_id"],
                        "split": record["split"],
                        "original_label": record["label"],
                        "final_label": final_row["label"],
                        "original_group": record["defect_group"],
                        "final_group": final_row["defect_group"],
                        "decision": review["decision"],
                        "tags": "|".join(review["tags"]),
                        "hard_negative": int(review["hard_negative"]),
                        "excluded": int(excluded),
                        "edited_mask": int(bool(review.get("edited_mask_path"))),
                        "reviewer": review["reviewer"],
                        "note": review["note"],
                        "updated_at": review["updated_at"],
                    }
                )
                if review["hard_negative"] and int(final_row["label"]) == 0:
                    hard_negative_rows.append(final_row)

        columns = list(next(iter(self.records.values()))["source_row"].keys())
        self._write_csv(export_dir / "cleaned_manifest.csv", clean_rows, columns)
        for split, rows in split_rows.items():
            self._write_csv(export_dir / "splits" / f"{split}.csv", rows, columns)
        self._write_csv(export_dir / "audit_log.csv", audit_rows)
        self._write_csv(export_dir / "hard_negatives.csv", hard_negative_rows, columns)
        self._write_csv(export_dir / "unresolved.csv", unresolved_rows)

        training_dataset = self._write_training_dataset(export_dir, package_records, columns)

        event_path = export_dir / "review_events.jsonl"
        with event_path.open("w", encoding="utf-8") as handle:
            for event in self.store.events():
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        summary = {
            "created_at": utc_now(),
            "dataset_root": str(self.config.dataset_root),
            "total_source_records": len(self.records),
            "reviewed_records": len(reviews),
            "exported_records": len(clean_rows),
            "excluded_or_unresolved": len(self.records) - len(clean_rows),
            "unresolved_records": len(unresolved_rows),
            "hard_negatives": len(hard_negative_rows),
            "split_counts": {key: len(value) for key, value in split_rows.items()},
            "training_dataset_path": training_dataset["path"],
            "training_dataset_split_counts": training_dataset["split_counts"],
            "training_dataset_materialization": training_dataset["materialization"],
            "source_dataset_modified": False,
        }
        (export_dir / "export_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"path": str(export_dir), "summary": summary}
