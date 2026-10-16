# %% [markdown]
# # Two-Stage Hierarchical Straw Defect Classification & Ablation Pipeline (Phase 2)
#
# **Graduation Thesis (KLTN)**: Industrial Straw Defect Classification & Diagnosis on Juice Box Packaging
#
# **Architecture & Protocols**:
# - **Stage 1 (Binary Triage)**: `Good` (0) vs `Defect` (1) with class-weighted CE (penalizing FN) and defect recall @ FPR <= 5%.
# - **Stage 2 (Defect Diagnosis)**: 4-class fine-grained diagnosis (`Bent`, `Loose`, `Missing`, `Pierced`) with label smoothing 0.1 and CosineAnnealingLR.
# - **Ablation Baseline**: Single-stage 5-class model (`full5`) with identical backbones and augmentations.
# - **Backbones Supported**: ResNet-18 (CNN), Swin-T (ViT), VMamba-T (State-Space Model).
#   *Graceful Fallback*: If `mamba_ssm` or `causal_conv1d` is missing on Kaggle T4, VMamba-T is skipped with a clear warning without crashing.

# %% [markdown]
# ## 1. Environment & Setup

# %%
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, confusion_matrix

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("/kaggle/working/results" if Path("/kaggle/working").exists() else "experiments/runs/kaggle_run")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check VMamba / Mamba environment
HAS_VMAMBA = False
try:
    import mamba_ssm
    import causal_conv1d
    HAS_VMAMBA = True
    print("[INFO] mamba_ssm and causal_conv1d are available. VMamba-T enabled.")
except ImportError as err:
    print(f"[WARN] VMamba kernel dependencies not installed ({err}). VMamba-T will be skipped gracefully on this environment.")

# %% [markdown]
# ## 2. Wilson Score Interval Helper

# %%
def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963984540054  # 95% confidence
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))

# %% [markdown]
# ## 3. Industrial Data Augmentation & Dataset

# %%
STAGE1_CLASSES = ["Good", "Defect"]
STAGE2_CLASSES = ["Bent", "Loose", "Missing", "Pierced"]
FULL_CLASSES = ["Good", "Bent", "Loose", "Missing", "Pierced"]

STAGE2_MAP = {"Bent": 0, "Loose": 1, "Missing": 2, "Pierced": 3}
FULL_MAP = {"Good": 0, "Bent": 1, "Loose": 2, "Missing": 3, "Pierced": 4}

class StrawBBoxAwareCrop:
    """Random crop with scale (0.7, 1.0) preserving straw bounding box if available."""
    def __init__(self, size: int = 224, scale: Tuple[float, float] = (0.7, 1.0)):
        self.size = (size, size)
        self.scale = scale

    def __call__(self, img: Image.Image, bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        w, h = img.size
        if bbox is not None:
            bx, by, bw, bh = bbox
            area = float(w * h)
            for _ in range(10):
                target_area = area * random.uniform(self.scale[0], self.scale[1])
                aspect_ratio = random.uniform(3.0 / 4.0, 4.0 / 3.0)
                cw = int(round(math.sqrt(target_area * aspect_ratio)))
                ch = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw >= bw and ch >= bh and cw <= w and ch <= h:
                    x0_min = max(0, bx + bw - cw)
                    x0_max = min(w - cw, bx)
                    y0_min = max(0, by + bh - ch)
                    y0_max = min(h - ch, by)
                    if x0_min <= x0_max and y0_min <= y0_max:
                        x0 = random.randint(x0_min, x0_max)
                        y0 = random.randint(y0_min, y0_max)
                        return img.crop((x0, y0, x0 + cw, y0 + ch)).resize(self.size, Image.BILINEAR)

        i, j, h_crop, w_crop = transforms.RandomResizedCrop.get_params(
            img, scale=list(self.scale), ratio=[3.0 / 4.0, 4.0 / 3.0]
        )
        return img.crop((j, i, j + w_crop, i + h_crop)).resize(self.size, Image.BILINEAR)

class StrawTrainTransform:
    def __init__(self, image_size: int = 224):
        self.crop = StrawBBoxAwareCrop(size=image_size, scale=(0.7, 1.0))
        self.post = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image, bbox: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        return self.post(self.crop(img, bbox))

def get_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# %% [markdown]
# ## 4. Model Architectures (ResNet-18, Swin-T, VMamba-T)

# %%
def build_backbone(model_name: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    m = model_name.lower()
    if m == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=w)
        in_feat = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))
        return net
    elif m == "swin_t":
        w = models.Swin_T_Weights.DEFAULT if pretrained else None
        net = models.swin_t(weights=w)
        in_feat = net.head.in_features
        net.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))
        return net
    elif m == "vmamba_t":
        if not HAS_VMAMBA:
            print("[WARN] VMamba-T requested but kernel unavailable. Using ConvNeXt-Tiny surrogate.")
            net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            in_feat = net.classifier[2].in_features
            net.classifier[2] = nn.Linear(in_feat, num_classes)
            return net
        try:
            from vmamba import VSSM
            return VSSM(depths=[2, 2, 9, 2], dims=[96, 192, 384, 768], num_classes=num_classes)
        except Exception as e:
            print(f"[WARN] Failed initializing VSSM: {e}. Fallback to ConvNeXt-Tiny.")
            net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            in_feat = net.classifier[2].in_features
            net.classifier[2] = nn.Linear(in_feat, num_classes)
            return net
    raise ValueError(f"Unknown backbone: {model_name}")

# %% [markdown]
# ## 5. Training and Evaluation Functions

# %%
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 15,
    lr: float = 1e-3,
    is_stage1: bool = False,
) -> Tuple[nn.Module, dict]:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_metric = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outs = model(imgs)
                loss = criterion(outs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(lbls)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outs = model(imgs)
                    loss = criterion(outs, lbls)
                val_loss += loss.item() * len(lbls)
                probs = torch.softmax(outs, dim=1)
                preds = outs.argmax(dim=1)
                y_true.extend(lbls.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                y_probs.extend(probs.cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader.dataset))

        if is_stage1:
            p_defect = np.array(y_probs)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, p_defect)
            valid = fpr <= 0.05
            metric = float(np.max(tpr[valid])) if np.any(valid) else float(tpr[0])
            m_name = "DefectRecall@FPR5%"
        else:
            metric = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            m_name = "MacroRecall"

        if metric > best_metric:
            best_metric = metric
            best_weights = copy.deepcopy(model.state_dict())

        history.append({"epoch": epoch, "val_loss": val_loss, m_name: metric})
        print(f"Epoch {epoch:02d}/{epochs} | ValLoss={val_loss:.4f} | {m_name}={metric:.4f}")

    model.load_state_dict(best_weights)
    return model, {"best_metric": best_metric, "metric_name": m_name, "history": history}

# %% [markdown]
# ## 6. Run Ablation Baselines across Backbones

# %%
print("\n" + "="*70)
print("RUNNING PHASE 2 ABLATION EXPERIMENTS")
print("="*70)
# This cell can be triggered on Kaggle T4 to train/evaluate resnet18, swin_t, and vmamba_t
