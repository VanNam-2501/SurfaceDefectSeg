"""Training datasets and augmentation helpers.

Final main protocol:
- native-resolution source images
- 512x512 defect-aware patches for E2-E6
- optional aspect-ratio resize+pad dataset for E1 only
- mild photometric augmentation is the default because it does not assume that
  product orientation is physically irrelevant
- geometric flips/90-degree rotations remain opt-in only
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ani_dataset import AluminumPatchDataset, IMAGENET_MEAN, IMAGENET_STD


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


def _to_unit_range(image: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(device=image.device, dtype=image.dtype)
    std = _STD.to(device=image.device, dtype=image.dtype)
    return (image * std + mean).clamp(0.0, 1.0)


def _normalize(image: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(device=image.device, dtype=image.dtype)
    std = _STD.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std


class MildPhotometricAugmentDataset(Dataset):
    """Mild brightness/contrast/gamma augmentation on image only."""

    def __init__(
        self,
        base: Dataset,
        brightness: float = 0.10,
        contrast: float = 0.10,
        gamma: float = 0.10,
    ) -> None:
        self.base = base
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.gamma = float(gamma)

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return float(torch.empty(1).uniform_(low, high).item())

    def __getitem__(self, index: int):
        image, mask = self.base[index]
        x = _to_unit_range(image)

        if self.brightness > 0:
            factor = self._uniform(1.0 - self.brightness, 1.0 + self.brightness)
            x = x * factor

        if self.contrast > 0:
            factor = self._uniform(1.0 - self.contrast, 1.0 + self.contrast)
            channel_mean = x.mean(dim=(1, 2), keepdim=True)
            x = (x - channel_mean) * factor + channel_mean

        if self.gamma > 0:
            gamma_value = self._uniform(1.0 - self.gamma, 1.0 + self.gamma)
            x = x.clamp(1e-6, 1.0).pow(gamma_value)

        x = _normalize(x.clamp(0.0, 1.0))
        return x.contiguous(), mask.contiguous()


class GeometricAugmentDataset(Dataset):
    """Optional aligned H/V flips and 90-degree rotations."""

    def __init__(self, base: Dataset, enabled: bool = True) -> None:
        self.base = base
        self.enabled = bool(enabled)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, mask = self.base[index]
        if not self.enabled:
            return image, mask
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(2,))
            mask = torch.flip(mask, dims=(2,))
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(1,))
            mask = torch.flip(mask, dims=(1,))
        if torch.rand(()) < 0.5:
            k = int(torch.randint(1, 4, (1,)).item())
            image = torch.rot90(image, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(1, 2))
        return image.contiguous(), mask.contiguous()


class ResizePadDataset(Dataset):
    """E1 baseline: resize full image with preserved aspect ratio, then center-pad."""

    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path,
        size: int = 512,
        normalize: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.records = pd.read_csv(self.csv_path)
        self.size = int(size)
        self.normalize = bool(normalize)
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, value: object) -> Path:
        path = Path(_text(value))
        return path.resolve() if path.is_absolute() else (self.dataset_root / path).resolve()

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            arr = np.asarray(im)
        if arr.ndim == 2:
            return (arr != 0).astype(np.uint8)
        return np.any(arr != 0, axis=2).astype(np.uint8)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        label = int(row["label"])
        image = self._read_rgb(self._resolve(row["image_path"]))
        if label == 1:
            mask = self._read_mask(self._resolve(row["mask_path"]))
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        h, w = image.shape[:2]
        scale = min(self.size / h, self.size / w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))

        image_r = np.asarray(Image.fromarray(image).resize((nw, nh), Image.Resampling.BILINEAR))
        mask_r = np.asarray(
            Image.fromarray(mask).resize((nw, nh), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )

        top = (self.size - nh) // 2
        left = (self.size - nw) // 2
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        mcanvas = np.zeros((self.size, self.size), dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = image_r
        mcanvas[top : top + nh, left : left + nw] = mask_r

        image_t = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.0)
        if self.normalize:
            image_t = (image_t - self._mean) / self._std
        mask_t = torch.from_numpy(mcanvas[None].copy()).float()
        return image_t, mask_t


def build_train_dataset(
    csv_path: str | Path,
    dataset_root: str | Path,
    patch_size: int,
    seed: int,
    data_mode: str = "patch",
    augmentation: str = "photometric",
):
    if data_mode == "patch":
        base: Dataset = AluminumPatchDataset(
            csv_path=csv_path,
            dataset_root=dataset_root,
            patch_size=patch_size,
            normalize=True,
            seed=seed,
        )
    elif data_mode == "resize":
        base = ResizePadDataset(
            csv_path=csv_path,
            dataset_root=dataset_root,
            size=patch_size,
            normalize=True,
        )
    else:
        raise ValueError(f"Unsupported data_mode={data_mode!r}")

    if augmentation == "none":
        return base
    if augmentation == "photometric":
        return MildPhotometricAugmentDataset(base)
    if augmentation == "geometric":
        return GeometricAugmentDataset(base, enabled=True)
    if augmentation == "photometric_geometric":
        return GeometricAugmentDataset(MildPhotometricAugmentDataset(base), enabled=True)
    raise ValueError(f"Unsupported augmentation={augmentation!r}")
