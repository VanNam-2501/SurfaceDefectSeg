"""Losses for binary industrial defect segmentation.

Final thesis protocol:
- BCEWithLogitsLoss is computed on every sample (Good + Defect).
- Dice loss is computed only on samples whose GT mask contains defect pixels.

This avoids asking Dice to optimize all-zero Good masks while keeping BCE as the
main false-positive suppressor on normal regions.
"""
from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss operating on raw logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be greater than zero")
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logit/target shape mismatch: {tuple(logits.shape)} versus {tuple(targets.shape)}"
            )
        targets = targets.to(dtype=logits.dtype)
        probabilities = torch.sigmoid(logits)
        reduce_dims = tuple(range(1, logits.ndim))
        intersection = (probabilities * targets).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """0.5 * BCE(all samples) + 0.5 * Dice(positive samples only) by default."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_smooth: float = 1.0,
        pos_weight: torch.Tensor | None = None,
        dice_positive_only: bool = True,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_positive_only = bool(dice_positive_only)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=dice_smooth)

    def components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logit/target shape mismatch: {tuple(logits.shape)} versus {tuple(targets.shape)}"
            )
        targets = targets.to(dtype=logits.dtype)
        bce_loss = self.bce(logits, targets)

        if self.dice_positive_only:
            positive = targets.flatten(1).sum(1) > 0
            positive_count = int(positive.sum().item())
            if positive_count:
                dice_loss = self.dice(logits[positive], targets[positive])
            else:
                # Keep the zero attached to the graph so AMP/backward stay simple.
                dice_loss = logits.sum() * 0.0
        else:
            positive_count = int((targets.flatten(1).sum(1) > 0).sum().item())
            dice_loss = self.dice(logits, targets)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total, bce_loss, dice_loss, positive_count

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total, _, _, _ = self.components(logits, targets)
        return total
