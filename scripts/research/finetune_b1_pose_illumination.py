"""Fine-tune only the selected B1 checkpoint for in-plane rotation and exposure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.data import CLASSES, file_hash
from src.straw_global_local.model import GlobalLocalCNN
from src.straw_global_local.runner import classification_metrics

PROTOCOL = ROOT / "artifacts/straw_global_local/data_v1"
CACHE = ROOT / "artifacts/straw_global_local/cache_v1"
BASE_CHECKPOINT = ROOT / "artifacts/local_b1_check/B1_s42/best.pt"
OUT = ROOT / "artifacts/b1_pose_illumination/candidate_v1"
SIZE = 512
DEVICE = torch.device("cuda")
MEAN = torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]
STD = torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]


class RawCache(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(CACHE / f"{row.sample_id}.npz", allow_pickle=False) as cached:
            image = torch.from_numpy(cached["global512"].copy()).permute(2, 0, 1).float() / 255.0
        return image, int(row.label)


def rotate_batch(images: torch.Tensor, angles: torch.Tensor, fill: float = .91) -> torch.Tensor:
    radians = torch.deg2rad(angles)
    cos, sin = torch.cos(radians), torch.sin(radians)
    theta = torch.zeros((len(images), 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    pad = torch.full((1, images.shape[1], 1, 1), fill, device=images.device, dtype=images.dtype)
    return F.grid_sample(images - pad, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + pad


def random_augment(images: torch.Tensor) -> torch.Tensor:
    batch = len(images)
    angles = torch.rand(batch, device=DEVICE) * 360.0 - 180.0
    result = rotate_batch(images, angles)
    brightness = .50 + .70 * torch.rand(batch, 1, 1, 1, device=DEVICE)
    gamma = .72 + .56 * torch.rand(batch, 1, 1, 1, device=DEVICE)
    contrast = .85 + .30 * torch.rand(batch, 1, 1, 1, device=DEVICE)
    result = (result * brightness).clamp(0, 1).pow(gamma)
    mean = result.mean(dim=(1, 2, 3), keepdim=True)
    result = ((result - mean) * contrast + mean).clamp(0, 1)
    if torch.rand((), device=DEVICE) < .20:
        result = (result + torch.randn_like(result) * (.008 + .012 * torch.rand((), device=DEVICE))).clamp(0, 1)
    return (result - MEAN) / STD


def normalized(images: torch.Tensor) -> torch.Tensor:
    return (images - MEAN) / STD


def evaluate(model, images, labels, batch_size=8):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = normalized(images[start:start+batch_size].to(DEVICE))
            outputs.append(model({"global": batch}).float().cpu().numpy())
    logits = np.concatenate(outputs)
    metrics = classification_metrics(labels, logits)
    return logits, {key: metrics[key] for key in ("n", "accuracy", "macro_f1", "fn", "fp", "wrong_defect_type", "confusion")}


def make_validation(frame):
    selected = (frame.groupby("label", group_keys=False)
                .sample(n=20, random_state=20261021).sort_values("sample_id").reset_index(drop=True))
    dataset = RawCache(selected)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    images, labels = [], []
    for x, y in loader:
        images.append(x)
        labels.extend(y.tolist())
    images = torch.cat(images).to(DEVICE)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(20261021)
    angles = torch.tensor(rng.uniform(-35, 35, len(images)), dtype=torch.float32, device=DEVICE)
    factors = torch.tensor(rng.uniform(.50, .70, len(images)), dtype=torch.float32, device=DEVICE)[:, None, None, None]
    rotated = rotate_batch(images, angles)
    dark = (images * factors).clamp(0, 1)
    rotated_dark = (rotated * factors).clamp(0, 1)
    return selected, labels, {"clean": images, "rotated": rotated, "dark": dark, "rotated_dark": rotated_dark}


def score(model, labels, scenarios):
    results = {}
    logits = {}
    for name, images in scenarios.items():
        logits[name], results[name] = evaluate(model, images, labels)
    rank = .4*results["clean"]["macro_f1"] + .2*results["rotated"]["macro_f1"] + \
           .2*results["dark"]["macro_f1"] + .2*results["rotated_dark"]["macro_f1"]
    return rank, results, logits


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Use a new output folder: {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("This B1 fine-tune requires the configured CUDA GPU")
    torch.set_num_threads(4)
    random.seed(20261021)
    np.random.seed(20261021)
    torch.manual_seed(20261021)
    torch.cuda.manual_seed_all(20261021)
    OUT.mkdir(parents=True)
    train_frame = pd.read_csv(PROTOCOL / "train.csv")
    val_frame = pd.read_csv(PROTOCOL / "val.csv")
    selected, val_labels, scenarios = make_validation(val_frame)
    model = GlobalLocalCNN("B1", pretrained=False, freeze_bn=True).to(DEVICE)
    checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    base_rank, base_metrics, _ = score(model, val_labels, scenarios)
    print("baseline", json.dumps({"rank": base_rank, "metrics": base_metrics}), flush=True)
    loader = DataLoader(RawCache(train_frame), batch_size=4, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(20261021), pin_memory=True)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": 1.0e-5},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": 5.0e-5},
    ], weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda")
    epochs, patience, bad, best_rank = 8, 3, 0, base_rank
    best_path = OUT / "best.pt"
    torch.save({"model": model.state_dict(), "epoch": 0, "rank": base_rank,
                "metrics": base_metrics, "source_checkpoint_sha256": file_hash(BASE_CHECKPOINT)}, best_path)
    history = [{"epoch": 0, "train_loss": None, "rank": base_rank, "metrics": base_metrics}]
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        for step, (images, labels) in enumerate(loader, 1):
            images = random_augment(images.to(DEVICE, non_blocking=True))
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = F.cross_entropy(model({"global": images}), labels)
            scaler.scale(loss / 2.0).backward()
            if step % 2 == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
        rank, metrics, logits = score(model, val_labels, scenarios)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "rank": rank, "metrics": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if rank > best_rank + 1e-5:
            best_rank, bad = rank, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "rank": rank,
                "metrics": metrics, "source_checkpoint_sha256": file_hash(BASE_CHECKPOINT),
                "training_protocol": "frozen B1 train only; full in-plane rotation, brightness/gamma/contrast/noise augmentation"}, best_path)
            np.savez_compressed(OUT / "best_validation_logits.npz", labels=val_labels,
                sample_ids=selected.sample_id.astype(str).to_numpy(), **logits)
        else:
            bad += 1
        (OUT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if bad >= patience:
            break
    summary = {"base_checkpoint_sha256": file_hash(BASE_CHECKPOINT), "train_manifest_sha256": file_hash(PROTOCOL / "train.csv"),
        "val_manifest_sha256": file_hash(PROTOCOL / "val.csv"), "test_manifest_sha256": file_hash(PROTOCOL / "test.csv"),
        "best_epoch": torch.load(best_path, map_location="cpu", weights_only=False)["epoch"],
        "baseline_rank": base_rank, "best_rank": best_rank, "baseline_metrics": base_metrics,
        "selected_validation_n": len(selected), "classes": CLASSES,
        "limitation": "synthetic balanced validation diagnostic; final frozen test has not been read by this script"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
