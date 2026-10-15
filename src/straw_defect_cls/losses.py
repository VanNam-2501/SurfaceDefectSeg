"""Loss functions for multi-class classification.

Provides:
- Standard CrossEntropyLoss
- CrossEntropy with label smoothing
- Focal Loss (for handling class imbalance)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import NUM_CLASSES


class CrossEntropyWithLabelSmoothing(nn.Module):
    """Cross-entropy with manual label smoothing (works on all PyTorch versions)."""

    def __init__(self, num_classes: int = NUM_CLASSES, smoothing: float = 0.1) -> None:
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Lin et al., 2017)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        num_classes: int = NUM_CLASSES,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", torch.ones(num_classes))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        at = self.alpha.gather(0, targets)
        focal_loss = at * (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_criterion(loss_name: str = "ce", **kwargs) -> nn.Module:
    """Factory function for loss functions.

    Args:
        loss_name: ``'ce'`` (CrossEntropy), ``'ls_ce'`` (label smoothing),
                   ``'focal'`` (Focal Loss).
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == "ls_ce":
        return CrossEntropyWithLabelSmoothing(**kwargs)
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: '{loss_name}'. Available: ce, ls_ce, focal")
