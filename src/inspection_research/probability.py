"""Consistent binary/conditional factorization for any number of defect classes."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def joint_log_probabilities(binary_logits: torch.Tensor, type_logits: torch.Tensor) -> torch.Tensor:
    """Y=0 is good, Y=1..K are defects; K=1 is ordinary binary inspection."""
    binary_logits = binary_logits.reshape(-1)
    if type_logits.ndim != 2 or type_logits.shape[0] != len(binary_logits) or type_logits.shape[1] < 1:
        raise ValueError("Expected N binary logits and NxK conditional logits, K>=1")
    good = F.logsigmoid(-binary_logits).unsqueeze(1)
    defective = F.logsigmoid(binary_logits).unsqueeze(1) + F.log_softmax(type_logits, dim=1)
    return torch.cat((good, defective), dim=1)


def hierarchical_nll(binary_logits: torch.Tensor, type_logits: torch.Tensor,
                     labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if labels.ndim != 1 or labels.dtype != torch.long or len(labels) != len(binary_logits):
        raise ValueError("Labels must be a length-N torch.long vector")
    if len(labels) == 0 or torch.any(labels < 0) or torch.any(labels > type_logits.shape[1]):
        raise ValueError("Labels must be in 0..K and nonempty")
    return F.nll_loss(joint_log_probabilities(binary_logits, type_logits), labels, reduction=reduction)


class HierarchicalHead(nn.Module):
    """Two heads on a shared representation, rather than two mandatory encoders."""
    def __init__(self, in_features: int, num_defects: int):
        super().__init__()
        if in_features < 1 or num_defects < 1:
            raise ValueError("in_features and num_defects must be positive")
        self.binary = nn.Linear(in_features, 1)
        self.types = nn.Linear(in_features, num_defects)

    def forward(self, features):
        return self.binary(features).squeeze(-1), self.types(features)


def entropic_group_risk(losses: torch.Tensor, groups: torch.Tensor, num_groups: int,
                        temperature: float = 0.2, prior: torch.Tensor | None = None):
    """tau log sum_g prior_g exp(mean_loss_g/tau), with exact group gradients.

    Every declared group must be present. An outer trainer must use an appropriate
    sampler/full-group pass; silently dropping groups changes the objective.
    Minibatch nonlinear estimates need not be unbiased population risk estimates.
    """
    if losses.ndim != 1 or groups.shape != losses.shape or not len(losses):
        raise ValueError("Expected aligned, nonempty per-example losses and group IDs")
    if groups.dtype != torch.long or temperature <= 0 or num_groups < 1:
        raise ValueError("Need integer groups, positive temperature and group count")
    if torch.any(groups < 0) or torch.any(groups >= num_groups) or not torch.isfinite(losses).all():
        raise ValueError("Invalid group IDs or nonfinite losses")
    counts = torch.bincount(groups, minlength=num_groups)
    if torch.any(counts == 0):
        raise ValueError("A declared group is absent; use a group-aware sampler")
    risks = torch.stack([losses[groups == group].mean() for group in range(num_groups)])
    if prior is None:
        prior = torch.ones(num_groups, device=losses.device, dtype=losses.dtype) / num_groups
    else:
        prior = torch.as_tensor(prior, device=losses.device, dtype=losses.dtype)
        if prior.shape != (num_groups,) or not torch.isfinite(prior).all() or torch.any(prior <= 0):
            raise ValueError("Group prior must be finite and strictly positive")
        prior = prior / prior.sum()
    log_weights = prior.log() + risks / temperature
    objective = temperature * torch.logsumexp(log_weights, dim=0)
    return objective, risks, torch.softmax(log_weights, dim=0)
