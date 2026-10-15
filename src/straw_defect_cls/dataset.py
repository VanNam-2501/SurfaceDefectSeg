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
CLASS_NAMES = ["good", "bent_straw", "missing_straw", "loose_straw", "pierced_straw"]
NUM_CLASSES = 5
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────
class StrawDefectDataset(Dataset):
    """Multi-class classification dataset for juice box straw defects.

    Supports two loading modes:
    1. **Folder mode**: ``root_dir`` points to a directory with subfolders
       per class (e.g. ``good/``, ``bent_straw/``, …).  If a ``split``
       sub-directory exists (e.g. ``root_dir/train/``), images are loaded
       from there.
    2. **CSV mode**: ``csv_file`` with columns ``image_path`` and ``label``.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        csv_file: str | Path | None = None,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.samples: list[Tuple[str, int]] = []

        if csv_file is not None:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                self.samples.append((str(row["image_path"]), int(row["label"])))
        elif root_dir is not None:
            root_path = Path(root_dir)
            # If there is a split subfolder (train / val / test), use it
            if (root_path / split).exists():
                root_path = root_path / split
            for class_name in sorted(root_path.iterdir()):
                if not class_name.is_dir():
                    continue
                # Map folder name to label index
                folder_name = class_name.name.lower()
                if folder_name in CLASS_TO_IDX:
                    label = CLASS_TO_IDX[folder_name]
                else:
                    # Try to find a matching class name loosely
                    label = len(CLASS_NAMES)  # unknown; keep scanning
                    for cname, cidx in CLASS_TO_IDX.items():
                        if cname in folder_name or folder_name in cname:
                            label = cidx
                            break
                    if label == len(CLASS_NAMES):
                        continue  # skip unrecognised folders

                for img_path in sorted(class_name.iterdir()):
                    if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                        self.samples.append((str(img_path), label))
        else:
            raise ValueError("Must provide either root_dir or csv_file.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────
def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Training-time augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Validation / test time transforms (deterministic)."""
    return transforms.Compose([
        transforms.Resize(image_size + 32),
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
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders.

    Supports both folder-based and CSV-based datasets.
    Returns a dict mapping split name -> DataLoader.
    """
    loaders: dict[str, DataLoader] = {}

    if train_csv and Path(train_csv).exists():
        train_ds = StrawDefectDataset(csv_file=train_csv, transform=get_train_transform(image_size))
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)
    elif data_dir and (Path(data_dir) / "train").exists():
        train_ds = StrawDefectDataset(root_dir=data_dir, split="train", transform=get_train_transform(image_size))
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)

    if val_csv and Path(val_csv).exists():
        val_ds = StrawDefectDataset(csv_file=val_csv, transform=get_val_transform(image_size))
        loaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    elif data_dir and (Path(data_dir) / "val").exists():
        val_ds = StrawDefectDataset(root_dir=data_dir, split="val", transform=get_val_transform(image_size))
        loaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    if test_csv and Path(test_csv).exists():
        test_ds = StrawDefectDataset(csv_file=test_csv, transform=get_val_transform(image_size))
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
    elif data_dir and (Path(data_dir) / "test").exists():
        test_ds = StrawDefectDataset(root_dir=data_dir, split="test", transform=get_val_transform(image_size))
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
    """Create stratified train / val / test split CSVs from a class-folder dataset.

    Returns split statistics.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    root_path = Path(root_dir)
    data: list[dict] = []

    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        folder_name = class_dir.name.lower()
        if folder_name not in CLASS_TO_IDX:
            continue
        label = CLASS_TO_IDX[folder_name]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                data.append({
                    "image_path": str(img_path),
                    "label": label,
                    "class_name": folder_name,
                })

    if not data:
        raise RuntimeError(f"No images found in {root_dir}. Expected subfolders: {CLASS_NAMES}")

    df = pd.DataFrame(data)

    # Stratified two-stage split
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["label"], random_state=seed
    )
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative, stratify=train_val_df["label"], random_state=seed
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    stats = {"total": len(df), "train": len(train_df), "val": len(val_df), "test": len(test_df)}
    print(f"Dataset split: {stats}")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["class_name"].value_counts().to_dict()
        print(f"  {split_name}: {dict(sorted(dist.items()))}")
    return stats
