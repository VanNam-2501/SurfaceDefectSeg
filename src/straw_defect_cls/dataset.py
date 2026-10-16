"""PyTorch Dataset and transforms for Juice Box Straw Defects Classification.

Dataset: https://www.kaggle.com/datasets/jhamilgutierrez/juice-box-straw-defects
- 2000 images total (3000x3000, 1:1 aspect ratio)
- 5 classes, 250 images each: good, bent_straw, missing_straw, loose_straw, pierced_straw
- 4 juice box brands x 50 images per brand per class
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────────
# Class definitions
# ──────────────────────────────────────────────────────────────────────────────
STAGE1_CLASS_NAMES = ["good", "defect"]
STAGE1_NUM_CLASSES = 2
STAGE1_CLASS_TO_IDX = {name: idx for idx, name in enumerate(STAGE1_CLASS_NAMES)}

STAGE2_CLASS_NAMES = ["bent", "loose", "missing", "pierced"]
STAGE2_NUM_CLASSES = 4
STAGE2_CLASS_TO_IDX = {name: idx for idx, name in enumerate(STAGE2_CLASS_NAMES)}

FULL_CLASS_NAMES = ["good", "bent", "loose", "missing", "pierced"]
FULL_NUM_CLASSES = 5
FULL_CLASS_TO_IDX = {name: idx for idx, name in enumerate(FULL_CLASS_NAMES)}

# Default aliases for backward compatibility
CLASS_NAMES = FULL_CLASS_NAMES
NUM_CLASSES = FULL_NUM_CLASSES
CLASS_TO_IDX = FULL_CLASS_TO_IDX
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def normalize_class_name(folder_or_name: str) -> Optional[str]:
    """Normalize folder/class name to standard 5-class naming."""
    s = folder_or_name.lower().replace("_straw", "").replace("-", " ").strip()
    if "good" in s:
        return "good"
    if "bent" in s:
        return "bent"
    if "loose" in s:
        return "loose"
    if "miss" in s:
        return "missing"
    if "pierce" in s:
        return "pierced"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────
class StrawDefectDataset(Dataset):
    """Hierarchical and multi-class classification dataset for juice box straw defects.

    Supports three task modes:
    - ``'full'``: 5 classes (0: good, 1: bent, 2: loose, 3: missing, 4: pierced)
    - ``'stage1'``: Binary triage (0: good, 1: defect)
    - ``'stage2'``: 4-class defect diagnosis (0: bent, 1: loose, 2: missing, 3: pierced; ignores 'good')
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        csv_file: str | Path | None = None,
        masks_csv: str | Path | None = None,
        split: str = "train",
        task: str = "full",
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.task = task.lower()
        self.samples: list[Tuple[str, int, str]] = []  # (image_path, label, raw_class)
        self.bbox_map: dict[str, Tuple[int, int, int, int]] = {}

        # Load mask bounding boxes if available
        mask_csv_path = None
        if masks_csv is not None and Path(masks_csv).exists():
            mask_csv_path = Path(masks_csv)
        elif Path("data/masks/masks.csv").exists():
            mask_csv_path = Path("data/masks/masks.csv")

        if mask_csv_path is not None:
            try:
                m_df = pd.read_csv(mask_csv_path)
                for _, m_row in m_df.iterrows():
                    name = Path(str(m_row["image_path"])).name
                    self.bbox_map[name] = (
                        int(m_row["bbox_x"]),
                        int(m_row["bbox_y"]),
                        int(m_row["bbox_w"]),
                        int(m_row["bbox_h"]),
                    )
            except Exception:
                pass

        if csv_file is not None:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                img_path = str(row["image_path"])
                raw_cls = str(row.get("raw_class", "")).lower()
                if not raw_cls and "class_name" in row:
                    raw_cls = str(row["class_name"]).lower()

                # Infer standard class if not explicit
                std_cls = normalize_class_name(raw_cls)
                if std_cls is None and "label" in row:
                    val = int(row["label"])
                    if 0 <= val < len(FULL_CLASS_NAMES):
                        std_cls = FULL_CLASS_NAMES[val]

                if std_cls is None:
                    continue

                # Filter and assign labels by task
                if self.task == "stage1":
                    target_label = 0 if std_cls == "good" else 1
                    self.samples.append((img_path, target_label, std_cls))
                elif self.task == "stage2":
                    if std_cls == "good":
                        continue  # Stage 2 diagnoses defects only
                    target_label = STAGE2_CLASS_TO_IDX[std_cls]
                    self.samples.append((img_path, target_label, std_cls))
                else:  # 'full'
                    target_label = FULL_CLASS_TO_IDX[std_cls]
                    self.samples.append((img_path, target_label, std_cls))

        elif root_dir is not None:
            root_path = Path(root_dir)
            if (root_path / split).exists():
                root_path = root_path / split

            # Scan files (handles flat or nested brand/defect hierarchy)
            for path in sorted(root_path.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                std_cls = normalize_class_name(path.parent.name)
                if std_cls is None and path.parent.parent.name:
                    std_cls = normalize_class_name(path.parent.parent.name)
                if std_cls is None:
                    continue

                if self.task == "stage1":
                    target_label = 0 if std_cls == "good" else 1
                    self.samples.append((str(path), target_label, std_cls))
                elif self.task == "stage2":
                    if std_cls == "good":
                        continue
                    target_label = STAGE2_CLASS_TO_IDX[std_cls]
                    self.samples.append((str(path), target_label, std_cls))
                else:
                    target_label = FULL_CLASS_TO_IDX[std_cls]
                    self.samples.append((str(path), target_label, std_cls))
        else:
            raise ValueError("Must provide either root_dir or csv_file.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            if isinstance(self.transform, StrawTrainTransform):
                bbox = self.bbox_map.get(Path(img_path).name, None)
                image = self.transform(image, bbox=bbox)
            else:
                image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> list[str]:
        if self.task == "stage1":
            return STAGE1_CLASS_NAMES
        if self.task == "stage2":
            return STAGE2_CLASS_NAMES
        return FULL_CLASS_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────
class StrawBBoxAwareCrop:
    """Random crop with scale (0.7, 1.0) preserving straw bounding box if available."""

    def __init__(self, size: int | Tuple[int, int], scale: Tuple[float, float] = (0.7, 1.0)) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale

    def __call__(self, img: Image.Image, bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        w, h = img.size
        if bbox is not None:
            bx, by, bw, bh = bbox
            area = float(w * h)
            # Try 10 attempts to find a random crop enclosing the straw bbox
            for _ in range(10):
                target_area = area * np.random.uniform(self.scale[0], self.scale[1])
                aspect_ratio = float(np.random.uniform(3.0 / 4.0, 4.0 / 3.0))
                cw = int(round(np.sqrt(target_area * aspect_ratio)))
                ch = int(round(np.sqrt(target_area / aspect_ratio)))

                if cw >= bw and ch >= bh and cw <= w and ch <= h:
                    x0_min = max(0, bx + bw - cw)
                    x0_max = min(w - cw, bx)
                    y0_min = max(0, by + bh - ch)
                    y0_max = min(h - ch, by)
                    if x0_min <= x0_max and y0_min <= y0_max:
                        x0 = int(np.random.randint(x0_min, x0_max + 1))
                        y0 = int(np.random.randint(y0_min, y0_max + 1))
                        cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
                        return cropped.resize(self.size, Image.BILINEAR)

        # Fallback to standard torchvision random resized crop
        i, j, h_crop, w_crop = transforms.RandomResizedCrop.get_params(
            img, scale=list(self.scale), ratio=[3.0 / 4.0, 4.0 / 3.0]
        )
        cropped = img.crop((j, i, j + w_crop, i + h_crop))
        return cropped.resize(self.size, Image.BILINEAR)


class StrawTrainTransform:
    """Industrial augmentation pipeline with bbox-preserving crop when available:
    - Rotate(limit=5)
    - KHÔNG VerticalFlip (chỉ RandomHorizontalFlip)
    - RandomResizedCrop scale (0.7, 1.0) with straw bbox preservation
    - ColorJitter nhẹ (brightness=0.2, contrast=0.2, sat=0.1, hue=0.05) + RandomGrayscale(p=0.2)
    """

    def __init__(self, image_size: int = 224, scale: Tuple[float, float] = (0.7, 1.0)) -> None:
        self.crop = StrawBBoxAwareCrop(size=image_size, scale=scale)
        self.post_crop = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image, bbox: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        cropped = self.crop(img, bbox=bbox)
        return self.post_crop(cropped)


def get_train_transform(image_size: int = 224, use_bbox_preserving: bool = True) -> Callable:
    """Industrial training-time augmentations:
    - Rotate(limit=5)
    - KHÔNG VerticalFlip (chỉ RandomHorizontalFlip)
    - RandomResizedCrop scale (0.7, 1.0)
    - ColorJitter nhẹ + RandomGrayscale p=0.2
    """
    if use_bbox_preserving:
        return StrawTrainTransform(image_size=image_size, scale=(0.7, 1.0))
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Validation / test time transforms (deterministic)."""
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataloader builder
# ──────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    data_dir: str | Path | None = None,
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    masks_csv: str | Path | None = None,
    task: str = "full",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders for 'full', 'stage1', or 'stage2'."""
    loaders: dict[str, DataLoader] = {}

    if train_csv and Path(train_csv).exists():
        train_ds = StrawDefectDataset(
            csv_file=train_csv, masks_csv=masks_csv, task=task, transform=get_train_transform(image_size)
        )
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=len(train_ds) > batch_size)
    elif data_dir and (Path(data_dir) / "train").exists():
        train_ds = StrawDefectDataset(
            root_dir=data_dir, split="train", masks_csv=masks_csv, task=task, transform=get_train_transform(image_size)
        )
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=len(train_ds) > batch_size)

    if val_csv and Path(val_csv).exists():
        val_ds = StrawDefectDataset(csv_file=val_csv, task=task, transform=get_val_transform(image_size))
        loaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    elif data_dir and (Path(data_dir) / "val").exists():
        val_ds = StrawDefectDataset(root_dir=data_dir, split="val", task=task, transform=get_val_transform(image_size))
        loaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    if test_csv and Path(test_csv).exists():
        test_ds = StrawDefectDataset(csv_file=test_csv, task=task, transform=get_val_transform(image_size))
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
    elif data_dir and (Path(data_dir) / "test").exists():
        test_ds = StrawDefectDataset(root_dir=data_dir, split="test", task=task, transform=get_val_transform(image_size))
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

    return loaders


# ──────────────────────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────────────────────
def split_dataset(
    root_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Create unified stratified train / val / test split CSVs for full, stage1, and stage2."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    root_path = Path(root_dir)
    data: list[dict] = []

    # Recursively find images (works for both flat class folders and nested Brand/Class folders)
    for img_path in sorted(root_path.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        parent_name = img_path.parent.name
        std_cls = normalize_class_name(parent_name)
        brand = "unknown"
        if std_cls is None:
            # Check grandparent (if path is root/Brand/Class/image.jpg)
            std_cls = normalize_class_name(img_path.parent.parent.name)
            if std_cls is not None:
                brand = img_path.parent.name
        else:
            # Grandparent might be brand
            if img_path.parent.parent != root_path:
                brand = img_path.parent.parent.name

        if std_cls is None:
            continue

        label_5class = FULL_CLASS_TO_IDX[std_cls]
        label_stage1 = 0 if std_cls == "good" else 1
        label_stage2 = STAGE2_CLASS_TO_IDX[std_cls] if std_cls != "good" else -1

        data.append({
            "image_path": str(img_path),
            "brand": brand,
            "raw_class": std_cls,
            "label": label_5class,
            "label_stage1": label_stage1,
            "label_stage2": label_stage2,
            "class_name": std_cls,
        })

    if not data:
        raise RuntimeError(f"No valid images found in {root_dir}.")

    df = pd.DataFrame(data)

    # Stratified split based on 5-class labels (guarantees balanced distribution for all stages)
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["label"], random_state=seed
    )
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative, stratify=train_val_df["label"], random_state=seed
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save comprehensive master splits
    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    # Save Stage 1 splits (Good vs Defective)
    for s_name, s_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df_s1 = s_df.copy()
        df_s1["label"] = df_s1["label_stage1"]
        df_s1.to_csv(out_path / f"{s_name}_stage1.csv", index=False)

    # Save Stage 2 splits (Only Defective images: Bent, Loose, Missing, Pierced)
    for s_name, s_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df_s2 = s_df[s_df["label_stage2"] >= 0].copy()
        df_s2["label"] = df_s2["label_stage2"]
        df_s2.to_csv(out_path / f"{s_name}_stage2.csv", index=False)

    stats = {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "stage2_total": len(df[df["label_stage2"] >= 0]),
    }
    print(f"Hierarchical dataset split created at {out_path}: {stats}")
    return stats
