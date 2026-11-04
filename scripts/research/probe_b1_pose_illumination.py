"""Measure B1 sensitivity to in-plane rotation and low light on frozen validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.data import CLASSES
from src.straw_global_local.model import GlobalLocalCNN
from src.straw_global_local.runner import classification_metrics

PROTOCOL = ROOT / "artifacts/straw_global_local/data_v1"
CHECKPOINT = ROOT / "artifacts/local_b1_check/B1_s42/best.pt"
OUT = ROOT / "artifacts/b1_pose_illumination/probe"
SIZE = 512
MEAN = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
STD = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]


def source_rgb(path: str) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return np.asarray(image)


def rotate_keep_canvas(rgb: np.ndarray, angle: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = np.median(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0), axis=0)
    return cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(x) for x in border))


def estimate_deskew(rgb: np.ndarray) -> tuple[float | None, float, np.ndarray | None]:
    """Prototype silhouette/rotated-box estimate; returns correction, confidence, quad."""
    h, w = rgb.shape[:2]
    scale = min(1.0, 700.0 / max(h, w))
    small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = max(4, round(min(sh, sw) * .025))
    border_pixels = np.concatenate([lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3),
                                    lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3)])
    background = np.median(border_pixels, axis=0)
    delta = np.linalg.norm(lab - background, axis=2)
    mask = (delta > 13).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, (round(sw * .015) | 1)),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour) / float(sw * sh)
        if not .025 <= area <= .65:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) <= 0:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if ratio < 1.35 or ratio > 8.0:
            continue
        candidates.append((area, ratio, contour, rect))
    if not candidates:
        return None, 0.0, None
    area, ratio, contour, rect = max(candidates, key=lambda row: row[0])
    points = cv2.boxPoints(rect)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    vector = edges[int(np.argmax(lengths))]
    axis_angle = np.degrees(np.arctan2(vector[1], vector[0]))
    # Keep the long carton axis vertical; a 180-degree ambiguity remains by design.
    correction = (90.0 - axis_angle + 90.0) % 180.0 - 90.0
    fill = area / max(1e-6, cv2.contourArea(cv2.convexHull(contour)) / float(sw * sh))
    confidence = float(np.clip((min(ratio, 4.0) - 1.2) / 1.8, 0, 1) * np.clip(fill, 0, 1))
    return correction, confidence, points / scale


def gamma_correct(rgb: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(rgb, lut)


def clahe_luma(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def infer(model, images: list[np.ndarray]) -> np.ndarray:
    tensors = []
    for rgb in images:
        small = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(small.copy()).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    x = (torch.stack(tensors).to("cuda") - MEAN) / STD
    with torch.inference_mode():
        return model({"global": x}).float().cpu().numpy()


def report(labels: np.ndarray, logits: np.ndarray) -> dict:
    result = classification_metrics(labels, logits)
    return {k: result[k] for k in ("n", "accuracy", "macro_f1", "fn", "fp", "wrong_defect_type", "confusion")}


def main():
    torch.set_num_threads(4)
    model = GlobalLocalCNN("B1", pretrained=False, freeze_bn=True).eval().cuda()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    frame = (pd.read_csv(PROTOCOL / "val.csv").groupby("label", group_keys=False)
             .sample(n=20, random_state=20261021).sort_values("sample_id").reset_index(drop=True))
    labels = frame.label.to_numpy(dtype=int)
    originals, rotated, corrected, gamma05, gamma07, clahe05 = [], [], [], [], [], []
    rotation_rows = []
    for row in frame.itertuples():
        rgb = source_rgb(row.image_path)
        seed = int(hashlib.sha256(str(row.sample_id).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        angle = float(rng.uniform(-35.0, 35.0))
        tilted = rotate_keep_canvas(rgb, angle)
        correction, confidence, quad = estimate_deskew(tilted)
        if correction is None:
            rectified = tilted
        else:
            rectified = rotate_keep_canvas(tilted, correction)
        originals.append(rgb)
        rotated.append(tilted)
        corrected.append(rectified)
        dark05 = np.clip(rgb.astype(np.float32) * .50, 0, 255).astype(np.uint8)
        dark07 = np.clip(rgb.astype(np.float32) * .70, 0, 255).astype(np.uint8)
        gamma05.append(gamma_correct(dark05, .70))
        gamma07.append(gamma_correct(dark07, .82))
        clahe05.append(clahe_luma(dark05))
        rotation_rows.append({"sample_id": row.sample_id, "label": CLASSES[int(row.label)],
            "synthetic_angle_degrees": angle, "estimated_correction_degrees": correction,
            "deskew_confidence": confidence, "detected": correction is not None})
        if len(originals) % 50 == 0:
            print(f"Prepared {len(originals)}/{len(frame)} validation images", flush=True)
    scenarios = {
        "clean_original": originals,
        "rotated_baseline": rotated,
        "rotated_auto_deskew": corrected,
        "dark_50_baseline": [np.clip(a.astype(np.float32)*.5,0,255).astype(np.uint8) for a in originals],
        "dark_50_gamma_070": gamma05,
        "dark_50_clahe": clahe05,
        "dark_70_baseline": [np.clip(a.astype(np.float32)*.7,0,255).astype(np.uint8) for a in originals],
        "dark_70_gamma_082": gamma07,
    }
    metrics = {}
    predictions = {}
    for name, images in scenarios.items():
        logits = infer(model, images)
        metrics[name] = report(labels, logits)
        predictions[name] = logits.argmax(axis=1)
        print(name, metrics[name], flush=True)
    pred_frame = frame[["sample_id", "label", "class_name"]].copy()
    for name, values in predictions.items():
        pred_frame[name] = values
    OUT.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(OUT / "validation_predictions.csv", index=False)
    pd.DataFrame(rotation_rows).to_csv(OUT / "deskew_detections.csv", index=False)
    output = {"checkpoint": "B1_s42 best.pt", "split": "balanced frozen validation diagnostic subset n=100 (20/class)",
        "rotation_generator": "per-sample fixed seed; uniform in-plane angle [-35,35] degrees",
        "deskew": "border Lab median background mask + largest rotated rectangle; 180-degree ambiguity remains",
        "lighting": "synthetic multiplicative darkening; gamma and CLAHE are diagnostic-only",
        "metrics": metrics}
    (OUT / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved probe to {OUT}", flush=True)


if __name__ == "__main__":
    main()
