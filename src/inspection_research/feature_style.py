"""Training-time feature augmentation and consistency, independent of product labels.

MixStyle is due to Zhou et al. (ICLR 2021). Same-class pairing and the paired
consistency objective below are research variants, not claims of a new method
or of improved results. Their value requires controlled image experiments.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


def style_partners(labels, batch_size, policy="random", domains=None, device=None):
    """Pair within a batch. Invalid same-class/cross-domain pairs stay unchanged."""
    if policy not in {"random", "same_class", "cross_domain_same_class"}:
        raise ValueError("Unknown style pairing policy")
    if policy == "random":
        return torch.randperm(batch_size, device=device)
    if labels is None or labels.shape != (batch_size,) or labels.dtype != torch.long:
        raise ValueError("Conditional training needs length-B integer labels")
    if policy == "cross_domain_same_class" and (domains is None or domains.shape != (batch_size,)):
        raise ValueError("Cross-domain pairing requires observed domain IDs")
    labels = labels.to(device)
    if domains is not None:
        domains = domains.to(device)
    indices = torch.arange(batch_size, device=device)
    partners = indices.clone()
    for index in range(batch_size):
        allowed = (labels == labels[index]) & (indices != index)
        if policy == "cross_domain_same_class":
            allowed &= domains != domains[index]
        choices = indices[allowed]
        if len(choices):
            partners[index] = choices[torch.randint(len(choices), (), device=device)]
    return partners


class FeatureStyleMix(nn.Module):
    """Mix per-channel spatial mean/std; retain recipient normalized spatial content.

    Statistics are detached, as in MixStyle: gradients flow through recipient
    content, not through donor statistics. This is an explicit stop-gradient
    convention, not an assertion that style and defect content are separable.
    """
    def __init__(self, alpha=.1, strength=1., probability=.5, eps=1e-6):
        super().__init__()
        if alpha <= 0 or eps <= 0 or not 0 <= strength <= 1 or not 0 <= probability <= 1:
            raise ValueError("Invalid style augmentation parameters")
        self.alpha, self.strength, self.probability, self.eps = alpha, strength, probability, eps

    def forward(self, features, partners=None, mixing=None, force=False):
        if not self.training or self.strength == 0:
            return features
        if features.ndim != 4 or not features.is_floating_point():
            raise ValueError("Expected BxCxHxW floating point features")
        if not force and torch.rand((), device=features.device) >= self.probability:
            return features
        batch = features.shape[0]
        if partners is None or partners.shape != (batch,) or partners.dtype != torch.long:
            raise ValueError("Explicit partner indices are required")
        partners = partners.to(features.device)
        if torch.any(partners < 0) or torch.any(partners >= batch):
            raise ValueError("Partner index out of range")
        # Float32 statistics avoid low-precision variance under AMP.
        source = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
        mean = source.mean(dim=(2, 3), keepdim=True).detach()
        std = (source.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt().detach()
        normalized = (source - mean) / std
        if mixing is None:
            parameter = torch.tensor(self.alpha, device=features.device, dtype=source.dtype)
            mixing = torch.distributions.Beta(parameter, parameter).sample((batch,))
        mixing = torch.as_tensor(mixing, device=features.device, dtype=source.dtype)
        if mixing.shape != (batch,) or not torch.isfinite(mixing).all() or torch.any((mixing < 0) | (mixing > 1)):
            raise ValueError("Mixing coefficients must be length-B values in [0,1]")
        donor_weight = self.strength * (1 - mixing.reshape(batch, 1, 1, 1))
        mixed_mean = mean + donor_weight * (mean[partners] - mean)
        mixed_std = std + donor_weight * (std[partners] - std)
        result = (normalized * mixed_std + mixed_mean).to(features.dtype)
        # No partner must mean exact identity, including low-precision rounding.
        unchanged = partners == torch.arange(batch, device=features.device)
        return torch.where(unchanged.reshape(batch, 1, 1, 1), features, result)


def jensen_shannon_from_logits(first, second, reduction="mean"):
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] < 2:
        raise ValueError("Expected aligned NxC logits, C>=2")
    # Preserve float64 for algebra tests; use float32 under AMP.
    dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    log_p = F.log_softmax(first.to(dtype), dim=1)
    log_q = F.log_softmax(second.to(dtype), dim=1)
    log_m = torch.logsumexp(torch.stack((log_p, log_q)), dim=0) - torch.log(first.new_tensor(2., dtype=dtype))
    divergence = .5 * ((log_p.exp() * (log_p - log_m)).sum(1) + (log_q.exp() * (log_q - log_m)).sum(1))
    # Roundoff can produce a tiny negative value at identical distributions.
    divergence = divergence.clamp_min(0)
    if reduction == "none":
        return divergence
    if reduction == "mean":
        return divergence.mean()
    raise ValueError("Unknown reduction")


def paired_training_loss(clean_logits, mixed_logits, labels, consistency_weight=1., smoothing=0.):
    """1/2 CE(clean)+1/2 CE(mixed)+lambda JS, no pseudo-label or GT at inference."""
    if consistency_weight < 0 or not 0 <= smoothing < 1:
        raise ValueError("Invalid loss parameters")
    clean_ce = F.cross_entropy(clean_logits, labels, label_smoothing=smoothing)
    mixed_ce = F.cross_entropy(mixed_logits, labels, label_smoothing=smoothing)
    consistency = jensen_shannon_from_logits(clean_logits, mixed_logits)
    total = .5 * (clean_ce + mixed_ce) + consistency_weight * consistency
    return total, {"clean_ce": clean_ce.detach(), "mixed_ce": mixed_ce.detach(),
                   "js": consistency.detach(), "total": total.detach()}


class StyleResNet18(nn.Module):
    """A standard ResNet18 with parameter-free style mixing after layer1/layer2.

    Evaluation is ordinary ResNet inference with no labels, domains or donor
    image. Number of output classes is configured separately for each product.
    """
    def __init__(self, num_classes, pretrained=False, alpha=.1, strength=1., probability=.5):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Need at least Good and one defect class")
        self.network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        self.style = FeatureStyleMix(alpha, strength, probability)

    def forward(self, images, labels=None, domains=None, augment=False, pairing="random"):
        net = self.network
        partners = None
        if self.training and augment:
            partners = style_partners(labels, len(images), pairing, domains, images.device)
        features = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        features = net.layer1(features)
        if self.training and augment:
            features = self.style(features, partners)
        features = net.layer2(features)
        if self.training and augment:
            features = self.style(features, partners)
        features = net.layer3(features)
        features = net.layer4(features)
        return net.fc(torch.flatten(net.avgpool(features), 1))
