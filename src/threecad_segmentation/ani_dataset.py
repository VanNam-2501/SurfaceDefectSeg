"""PyTorch Dataset for on-the-fly 3CAD-ANI patch sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, get_worker_info

from patch_sampler import PatchSampler


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


class AluminumPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read native-resolution images and sample aligned patches on demand.

    Good samples receive an all-zero mask and a random crop. Defect samples use
    their GT mask to select a positive crop. No source image is resized.
    """

    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path,
        patch_size: int = 512,
        normalize: bool = True,
        seed: int = 42,
        sampler: PatchSampler | None = None,
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.records = pd.read_csv(self.csv_path)
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.sampler = sampler or PatchSampler(patch_size=patch_size)
        self.patch_size = self.sampler.patch_size
        self._rng: np.random.Generator | None = None
        self._rng_worker_id: int | None = None

        required = {"image_path", "mask_path", "label"}
        missing = sorted(required - set(self.records.columns))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        self.records["label"] = pd.to_numeric(self.records["label"], errors="raise").astype(int)
        invalid_labels = sorted(set(self.records["label"]) - {0, 1})
        if invalid_labels:
            raise ValueError(f"label must contain only 0/1, got {invalid_labels}")
        if self.records.empty:
            raise ValueError(f"CSV contains no samples: {self.csv_path}")

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Each DataLoader worker must create its own independent RNG stream.
        state["_rng"] = None
        state["_rng_worker_id"] = None
        return state

    def _get_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self._rng is None or self._rng_worker_id != worker_id:
            torch_seed = int(torch.initial_seed())
            combined_seed = (torch_seed + self.seed) % (2**32)
            self._rng = np.random.default_rng(combined_seed)
            self._rng_worker_id = worker_id
        return self._rng

    def _resolve(self, value: object) -> Path:
        text = _text(value)
        if not text:
            raise ValueError("Encountered an empty required path")
        path = Path(text)
        return path.resolve() if path.is_absolute() else (self.dataset_root / path).resolve()

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read image {path}: {exc}") from exc

    @staticmethod
    def _read_binary_mask(path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                array = np.asarray(image)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read mask {path}: {exc}") from exc
        if array.ndim == 2:
            return (array != 0).astype(np.uint8)
        if array.ndim == 3:
            return np.any(array != 0, axis=2).astype(np.uint8)
        raise ValueError(f"Unsupported mask shape {array.shape} at {path}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        label = int(row["label"])
        image_path = self._resolve(row["image_path"])
        image = self._read_rgb(image_path)

        if label == 1:
            mask_path = self._resolve(row["mask_path"])
            mask = self._read_binary_mask(mask_path)
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Image/mask mismatch for {image_path}: {image.shape[:2]} versus {mask.shape}"
                )
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        image_patch, mask_patch = self.sampler(
            image=image,
            mask=mask,
            is_defect=(label == 1),
            rng=self._get_rng(),
        )

        image_tensor = torch.from_numpy(image_patch.transpose(2, 0, 1).copy()).float().div_(255.0)
        if self.normalize:
            image_tensor = (image_tensor - self._mean) / self._std
        mask_tensor = torch.from_numpy(mask_patch[None].copy()).float()

        expected_image = (3, self.patch_size, self.patch_size)
        expected_mask = (1, self.patch_size, self.patch_size)
        if tuple(image_tensor.shape) != expected_image or tuple(mask_tensor.shape) != expected_mask:
            raise RuntimeError(
                f"Invalid tensor shapes: image={tuple(image_tensor.shape)}, mask={tuple(mask_tensor.shape)}"
            )
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
            raise RuntimeError("Output mask contains values other than 0 and 1")
        return image_tensor, mask_tensor


def denormalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """Convert a normalized CHW tensor back to the [0, 1] range for display."""
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)
