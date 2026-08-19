"""On-the-fly 512x512 patch sampling without resizing source images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PatchCoordinates:
    top: int
    left: int
    height: int
    width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


class PatchSampler:
    """Pad if needed, then sample aligned image/mask patches.

    Positive sampling first chooses one connected component uniformly. If its
    bounding box fits inside the patch, the crop contains the whole component.
    Otherwise, the crop is guaranteed to contain at least one pixel belonging
    to that component.
    """

    def __init__(self, patch_size: int = 512, image_pad_value: int = 0) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)
        self.image_pad_value = int(image_pad_value)

    def _validate(self, image: np.ndarray, mask: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {image.shape}")
        if mask.ndim != 2:
            raise ValueError(f"Expected binary mask [H,W], got {mask.shape}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {image.shape[:2]} versus {mask.shape}")
        values = np.unique(mask)
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError(f"Mask must contain only 0/1, got {values.tolist()}")

    def _pad(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        height, width = mask.shape
        missing_height = max(0, self.patch_size - height)
        missing_width = max(0, self.patch_size - width)
        pad_top = missing_height // 2
        pad_bottom = missing_height - pad_top
        pad_left = missing_width // 2
        pad_right = missing_width - pad_left

        if missing_height or missing_width:
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=self.image_pad_value,
            )
            mask = np.pad(
                mask,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
        return image, mask, (pad_top, pad_bottom, pad_left, pad_right)

    @staticmethod
    def _randint_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
        if high < low:
            raise ValueError(f"Invalid integer interval [{low}, {high}]")
        return int(rng.integers(low, high + 1))

    def _random_crop_start(
        self, height: int, width: int, rng: np.random.Generator
    ) -> tuple[int, int]:
        max_top = height - self.patch_size
        max_left = width - self.patch_size
        top = self._randint_inclusive(rng, 0, max_top)
        left = self._randint_inclusive(rng, 0, max_left)
        return top, left

    def _axis_start_containing_interval(
        self,
        interval_start: int,
        interval_end: int,
        axis_length: int,
        rng: np.random.Generator,
    ) -> int | None:
        """Return a crop start that contains [start, end), or None if impossible."""
        lowest = max(0, interval_end - self.patch_size)
        highest = min(interval_start, axis_length - self.patch_size)
        if lowest > highest:
            return None
        return self._randint_inclusive(rng, lowest, highest)

    def _axis_start_containing_point(
        self, point: int, axis_length: int, rng: np.random.Generator
    ) -> int:
        lowest = max(0, point - self.patch_size + 1)
        highest = min(point, axis_length - self.patch_size)
        return self._randint_inclusive(rng, lowest, highest)

    def _positive_crop_start(
        self, mask: np.ndarray, rng: np.random.Generator
    ) -> tuple[int, int]:
        component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8, copy=False), connectivity=8
        )
        if component_count <= 1:
            raise ValueError("Defect sample has an empty mask")

        component_id = int(rng.integers(1, component_count))
        left, top, width, height, _ = map(int, stats[component_id])
        image_height, image_width = mask.shape

        crop_top = self._axis_start_containing_interval(
            top, top + height, image_height, rng
        )
        crop_left = self._axis_start_containing_interval(
            left, left + width, image_width, rng
        )
        if crop_top is not None and crop_left is not None:
            return crop_top, crop_left

        # The selected component is larger than the patch on at least one axis.
        # Select one of its pixels so the resulting patch is still positive.
        component_pixels = np.argwhere(component_map == component_id)
        selected_y, selected_x = component_pixels[int(rng.integers(0, len(component_pixels)))]
        crop_top = self._axis_start_containing_point(int(selected_y), image_height, rng)
        crop_left = self._axis_start_containing_point(int(selected_x), image_width, rng)
        return crop_top, crop_left

    def sample(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        is_defect: bool,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, PatchCoordinates]:
        mask = (mask != 0).astype(np.uint8, copy=False)
        self._validate(image, mask)
        image, mask, pads = self._pad(image, mask)

        if is_defect:
            top, left = self._positive_crop_start(mask, rng)
        else:
            top, left = self._random_crop_start(*mask.shape, rng)

        bottom = top + self.patch_size
        right = left + self.patch_size
        image_patch = np.ascontiguousarray(image[top:bottom, left:right])
        mask_patch = np.ascontiguousarray(mask[top:bottom, left:right])

        expected_image_shape = (self.patch_size, self.patch_size, 3)
        expected_mask_shape = (self.patch_size, self.patch_size)
        if image_patch.shape != expected_image_shape or mask_patch.shape != expected_mask_shape:
            raise RuntimeError(
                f"Sampler returned invalid shapes: image={image_patch.shape}, mask={mask_patch.shape}"
            )
        if is_defect and int(mask_patch.sum()) == 0:
            raise RuntimeError("Positive sampling invariant failed: sampled mask is empty")

        coordinates = PatchCoordinates(
            top=top,
            left=left,
            height=self.patch_size,
            width=self.patch_size,
            pad_top=pads[0],
            pad_bottom=pads[1],
            pad_left=pads[2],
            pad_right=pads[3],
        )
        return image_patch, mask_patch, coordinates

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        is_defect: bool,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_patch, mask_patch, _ = self.sample(image, mask, is_defect, rng)
        return image_patch, mask_patch
