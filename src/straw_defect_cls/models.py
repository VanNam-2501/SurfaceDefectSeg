"""Model architectures for Juice Box Straw Defect Classification.

Three lightweight CNN backbones adapted for 5-class classification:
- ResNet-18 (ImageNet pretrained)
- EfficientNet-B0 (ImageNet pretrained)
- MobileNetV3-Small (ImageNet pretrained)

All models expose ``encoder_parameters()`` and ``classifier_parameters()``
for differential learning-rate optimisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights, resnet18,
    EfficientNet_B0_Weights, efficientnet_b0,
    MobileNet_V3_Small_Weights, mobilenet_v3_small,
)

from .dataset import NUM_CLASSES


# ──────────────────────────────────────────────────────────────────────────────
# ResNet-18
# ──────────────────────────────────────────────────────────────────────────────
class ResNet18Classifier(nn.Module):
    """ResNet-18 backbone with a single linear classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def encoder_parameters(self):
        return self.backbone.parameters()

    def classifier_parameters(self):
        return self.classifier.parameters()


# ──────────────────────────────────────────────────────────────────────────────
# EfficientNet-B0
# ──────────────────────────────────────────────────────────────────────────────
class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 backbone with custom classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def encoder_parameters(self):
        return self.backbone.parameters()

    def classifier_parameters(self):
        return self.classifier.parameters()


# ──────────────────────────────────────────────────────────────────────────────
# MobileNetV3-Small
# ──────────────────────────────────────────────────────────────────────────────
class MobileNetV3Classifier(nn.Module):
    """MobileNetV3-Small backbone with custom classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def encoder_parameters(self):
        return self.backbone.parameters()

    def classifier_parameters(self):
        return self.classifier.parameters()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[nn.Module]] = {
    "resnet18": ResNet18Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
    "mobilenet_v3": MobileNetV3Classifier,
}


def build_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate a classifier by name.

    Supported: ``resnet18``, ``efficientnet_b0``, ``mobilenet_v3``.
    """
    key = model_name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](num_classes=num_classes, pretrained=pretrained)
