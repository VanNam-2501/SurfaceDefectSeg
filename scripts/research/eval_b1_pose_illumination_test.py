"""One-shot held-out test comparison for baseline B1 and its fine-tuned candidate.

The frozen test split is used only here, after candidate selection is complete.
No parameters, thresholds, augmentation ranges or checkpoints are selected from it.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.data import CLASSES, file_hash
from src.straw_global_local.model import GlobalLocalCNN
from src.straw_global_local.runner import classification_metrics
from scripts.research.finetune_b1_pose_illumination import RawCache, rotate_batch, normalized

PROTOCOL = ROOT / "artifacts/straw_global_local/data_v1"
CACHE = ROOT / "artifacts/straw_global_local/cache_v1"
BASE = ROOT / "artifacts/local_b1_check/B1_s42/best.pt"
CANDIDATE = ROOT / "artifacts/b1_pose_illumination/candidate_v1/best.pt"
OUT = ROOT / "artifacts/b1_pose_illumination/test_comparison.json"
DEVICE = torch.device("cuda")


def load_model(path):
    model = GlobalLocalCNN("B1", pretrained=False, freeze_bn=True).to(DEVICE)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.eval()


def infer(model, images):
    values = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            x = normalized(images[start:start + 8].to(DEVICE))
            values.append(model({"global": x}).float().cpu().numpy())
    return np.concatenate(values)


def summarize(labels, logits):
    result = classification_metrics(labels, logits)
    return {k: result[k] for k in ("n", "accuracy", "macro_f1", "fn", "fp", "wrong_defect_type", "confusion")}


def main():
    print("loading held-out test inputs", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the configured B1 model")
    torch.set_num_threads(4)
    frame = pd.read_csv(PROTOCOL / "test.csv").sort_values("sample_id").reset_index(drop=True)
    dataset = RawCache(frame)
    images, labels = [], []
    for batch_index, (x, y) in enumerate(DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0), 1):
        images.append(x)
        labels.extend(y.tolist())
        if batch_index % 10 == 0:
            print(f"loaded {min(batch_index * 8, len(frame))}/{len(frame)}", flush=True)
    images = torch.cat(images).to(DEVICE)
    labels = np.asarray(labels, dtype=int)
    # Deterministic, paired synthetic challenge: rotation in the image plane, uniform exposure loss.
    rng = np.random.default_rng(20261021)
    angles = torch.tensor(rng.uniform(-35, 35, len(images)), dtype=torch.float32, device=DEVICE)
    factors = torch.tensor(rng.uniform(.50, .70, len(images)), dtype=torch.float32, device=DEVICE)[:, None, None, None]
    rotated = rotate_batch(images, angles)
    scenarios = {
        "clean": images,
        "rotated_pm35deg": rotated,
        "dark_x050_070": (images * factors).clamp(0, 1),
        "rotated_and_dark": (rotated * factors).clamp(0, 1),
    }
    all_logits = {}
    results = {}
    for name, path in (("baseline", BASE), ("finetuned_candidate", CANDIDATE)):
        print(f"evaluating {name}", flush=True)
        model = load_model(path)
        results[name] = {}
        for scenario, batch in scenarios.items():
            logits = infer(model, batch)
            results[name][scenario] = summarize(labels, logits)
            all_logits[f"{name}_{scenario}"] = logits
            print(f"  {scenario}: {results[name][scenario]['accuracy']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    # Paired prediction counts make the impact auditable beyond aggregate accuracy.
    for scenario, batch in scenarios.items():
        a = all_logits[f"baseline_{scenario}"].argmax(1)
        b = all_logits[f"finetuned_candidate_{scenario}"].argmax(1)
        results.setdefault("paired", {})[scenario] = {
            "candidate_fixed_baseline_errors": int(np.sum((a != labels) & (b == labels))),
            "candidate_introduced_errors": int(np.sum((a == labels) & (b != labels))),
            "predictions_changed": int(np.sum(a != b)),
        }
    output = {
        "role": "one-shot frozen test evaluation after candidate selection",
        "test_manifest_sha256": file_hash(PROTOCOL / "test.csv"),
        "baseline_checkpoint_sha256": file_hash(BASE),
        "candidate_checkpoint_sha256": file_hash(CANDIDATE),
        "classes": CLASSES,
        "n": len(frame),
        "synthetic_transform_note": "paired in-plane rotations uniformly sampled from -35 to +35 degrees and uniform image intensity factor 0.50 to 0.70; diagnostic only, not camera data",
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
