"""Image quality checks with explicit, configurable engineering thresholds."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class QualityConfig:
    min_brightness: float = 30
    max_brightness: float = 240
    min_contrast: float = 10
    min_sharpness: float = 18
    min_side: int = 48

    @classmethod
    def parse(cls, value=None):
        value = value or {}
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown quality setting")
        result = cls(**value)
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for v in asdict(result).values()):
            raise ValueError("Quality settings must be finite numbers")
        if not (0 <= result.min_brightness < result.max_brightness <= 255 and
                0 <= result.min_contrast <= 128 and 0 <= result.min_sharpness <= 10000 and
                8 <= result.min_side <= 1024):
            raise ValueError("Quality settings are outside supported ranges")
        return result


def assess(image: Image.Image, config=None, clipped=False):
    config = config or QualityConfig()
    image = ImageOps.exif_transpose(image).convert("RGB")
    gray = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32)
    gray = gray[16:-16, 16:-16]  # Ignore crop/background borders when estimating blur.
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    brightness, contrast, sharpness = float(gray.mean()), float(gray.std()), float(lap.var())
    reasons = []
    for failed, reason in ((brightness < config.min_brightness, "too_dark"),
                           (brightness > config.max_brightness, "too_bright"),
                           (contrast < config.min_contrast, "low_contrast"),
                           (sharpness < config.min_sharpness, "blurred"),
                           (min(image.size) < config.min_side, "too_small"),
                           (clipped, "clipped_product")):
        if failed:
            reasons.append(reason)
    return {"acceptable": not reasons, "reasons": reasons, "brightness": brightness,
            "contrast": contrast, "sharpness": sharpness, "width": image.width, "height": image.height,
            "score": sharpness * min(1., contrast / 35.) / (1 + abs(brightness - 128) / 128),
            "thresholds": asdict(config),
            "measurement": {"method": "interior_laplacian_variance_v1", "resize": [256, 256],
                            "border_excluded": 16, "input_size": list(image.size)},
            "scope": "engineering image checks; not a calibrated probability or validated defect detector"}
