"""Small, dependency-free helpers for training checkpoint compatibility."""
from __future__ import annotations

import math
from typing import Any


def recover_best_epoch(
    checkpoint: dict[str, Any],
    history_rows: list[dict[str, Any]],
) -> int:
    """Recover the selected epoch, including from older checkpoint formats."""
    if "best_epoch" in checkpoint:
        return int(checkpoint["best_epoch"])

    candidates: list[tuple[float, int]] = []
    for row in history_rows:
        try:
            metric = float(row["val_positive_dice_0.5"])
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(metric):
            candidates.append((metric, epoch))

    if not candidates:
        return 0

    try:
        saved_metric = float(checkpoint["best_metric"])
    except (KeyError, TypeError, ValueError):
        saved_metric = math.nan
    if math.isfinite(saved_metric):
        tolerance = max(1e-12, abs(saved_metric) * 1e-9)
        matching_epochs = [
            epoch for metric, epoch in candidates if abs(metric - saved_metric) <= tolerance
        ]
        if matching_epochs:
            return min(matching_epochs)

    # Match the training rule's preference for the earliest epoch on a tie.
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]
