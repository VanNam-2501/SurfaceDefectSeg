"""Deterministic whole-image preprocessing and synthetic stress tests."""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor, normalize

from .models import build_model


def transform_image(image: Image.Image, resolution: int, condition: str, sample_id: str) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
    if condition != "clean":
        family, severity = condition.rsplit("_", 1)
        level = int(severity) - 1
        if level not in range(3):
            raise ValueError(f"Invalid corruption: {condition}")
        if family == "blur":
            image = image.filter(ImageFilter.GaussianBlur([0.7, 1.5, 2.5][level]))
        elif family == "dark":
            image = ImageEnhance.Brightness(image).enhance([0.8, 0.6, 0.4][level])
        elif family == "noise":
            seed = int(hashlib.sha256(f"{sample_id}:{condition}:42".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            pixels = np.asarray(image, dtype=float) / 255
            pixels = np.clip(pixels + rng.normal(0, [0.02, 0.05, 0.10][level], pixels.shape), 0, 1)
            image = Image.fromarray(np.uint8(np.round(pixels * 255)))
        elif family == "jpeg":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=[70, 40, 15][level])
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        else:
            raise ValueError(f"Unknown condition: {condition}")
    return normalize(to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class InferenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, resolution: int, condition: str):
        self.rows = list(frame.itertuples())
        self.resolution, self.condition = resolution, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row.image_path) as image:
            tensor = transform_image(image, self.resolution, self.condition, row.sample_id)
        return tensor


@torch.inference_mode()
def infer(checkpoint: Path, frame: pd.DataFrame, condition: str, device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    model = build_model(cfg["model_name"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    loader = DataLoader(InferenceDataset(frame, cfg["image_size"], condition), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    values = [model(batch.to(device)).float().cpu().numpy() for batch in loader]
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(values)


@torch.inference_mode()
def benchmark(checkpoint: Path, device: str = "cuda", repeats: int = 100) -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    model = build_model(cfg["model_name"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    tensor = torch.zeros(1, 3, cfg["image_size"], cfg["image_size"], device=device)
    for _ in range(10):
        model(tensor)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(tensor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return {"device": device, "batch_size": 1, "precision": "float32", "warmup": 10,
            "repeats": repeats, "scope": "model forward only; excludes image decoding and preprocessing",
            "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(np.mean(times)), "parameters": sum(p.numel() for p in model.parameters()),
            "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.startswith("cuda") else None}
