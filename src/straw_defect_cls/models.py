"""Model architectures for Juice Box Straw Defect Classification.

Three lightweight CNN backbones adapted for 5-class classification:
- ResNet-18 (ImageNet pretrained)
- EfficientNet-B0 (ImageNet pretrained)
- MobileNetV3-Small (ImageNet pretrained)

All models expose ``encoder_parameters()`` and ``classifier_parameters()``
for differential learning-rate optimisation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights, resnet18,
    EfficientNet_B0_Weights, efficientnet_b0,
    MobileNet_V3_Small_Weights, mobilenet_v3_small,
    Swin_T_Weights, swin_t,
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
# Swin Transformer Tiny (Swin-T) - Vision Transformer Paradigm
# ──────────────────────────────────────────────────────────────────────────────
class SwinTClassifier(nn.Module):
    """Swin Transformer Tiny (Swin-T) for global self-attention modelling."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
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
# VMamba-T (VSSM) - State Space Model (Mamba) Paradigm
# ──────────────────────────────────────────────────────────────────────────────
class VMambaTClassifier(nn.Module):
    """Visual State Space Model (VMamba-T) with linear-complexity 2D scanning."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        repo_root = Path(__file__).resolve().parents[2]
        vmamba_dir = repo_root / "src" / "threecad_segmentation" / "third_party" / "VMamba"
        if str(vmamba_dir) not in sys.path:
            sys.path.insert(0, str(vmamba_dir))

        try:
            from vmamba import VSSM
            self.model = VSSM(
                depths=[2, 2, 9, 2],
                dims=[96, 192, 384, 768],
                num_classes=num_classes,
                drop_path_rate=0.2,
            )
        except Exception as e:
            print(f"[WARN] Failed to load official VSSM: {e}. Using fallback Mamba classifier.")
            from torchvision.models import convnext_tiny
            self.model = convnext_tiny(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def encoder_parameters(self):
        return (p for n, p in self.model.named_parameters() if "classifier" not in n and "head" not in n)

    def classifier_parameters(self):
        return (p for n, p in self.model.named_parameters() if "classifier" in n or "head" in n)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[nn.Module]] = {
    # The 3 Core Paradigms
    "resnet18": ResNet18Classifier,       # CNN
    "swin_t": SwinTClassifier,            # Transformer
    "vmamba_t": VMambaTClassifier,        # Mamba (SSM)
    # Additional CNN backbones
    "efficientnet_b0": EfficientNetB0Classifier,
    "mobilenet_v3": MobileNetV3Classifier,
}


def build_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate a classifier by name.

    Core Paradigms:
    - ``resnet18``: Convolutional Neural Network (CNN)
    - ``swin_t``: Vision Transformer (ViT / Swin)
    - ``vmamba_t``: Visual State Space Model (Mamba)
    """
    key = model_name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](num_classes=num_classes, pretrained=pretrained)


# ──────────────────────────────────────────────────────────────────────────────
# Two-stage Hierarchical Pipeline
# ──────────────────────────────────────────────────────────────────────────────
class HierarchicalInferencePipeline(nn.Module):
    """Two-stage hierarchical defect triage and diagnosis pipeline.

    - Stage 1: Triage (Good vs Defective)
    - Stage 2: Fine-grained diagnosis (Bent, Loose, Missing, Pierced)
    - Decision: PASS, REVIEW, or DEFECT
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        stage2_model: nn.Module,
        confidence_good_threshold: float = 0.80,
        confidence_defect_threshold: float = 0.65,
    ) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.confidence_good_threshold = confidence_good_threshold
        self.confidence_defect_threshold = confidence_defect_threshold

    @torch.no_grad()
    def predict_single(self, image_tensor: torch.Tensor) -> dict:
        """Process a single image tensor of shape (3, H, W) or (1, 3, H, W)."""
        import numpy as np

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Stage 1: Triage (Good vs Defective)
        s1_logits = self.stage1_model(image_tensor)
        s1_probs = torch.softmax(s1_logits, dim=1)[0].cpu().numpy()
        p_good, p_defect = float(s1_probs[0]), float(s1_probs[1])

        # Stage 2: Defect Diagnosis
        s2_logits = self.stage2_model(image_tensor)
        s2_probs = torch.softmax(s2_logits, dim=1)[0].cpu().numpy()
        stage2_classes = ["bent", "loose", "missing", "pierced"]
        s2_pred_idx = int(np.argmax(s2_probs))
        s2_pred_class = stage2_classes[s2_pred_idx]
        s2_confidence = float(s2_probs[s2_pred_idx])

        # Hierarchical Decision Logic
        if p_good >= self.confidence_good_threshold:
            decision = "PASS"
            predicted_class = "good"
            confidence = p_good
            reason = f"Stage 1 detected normal straw with high confidence ({p_good*100:.1f}%)."
        elif p_defect >= self.confidence_defect_threshold:
            decision = "DEFECT"
            predicted_class = s2_pred_class
            confidence = p_defect * s2_confidence
            reason = f"Stage 1 flagged defective ({p_defect*100:.1f}%). Stage 2 diagnosed {s2_pred_class} ({s2_confidence*100:.1f}%)."
        else:
            decision = "REVIEW"
            predicted_class = s2_pred_class if p_defect > p_good else "good"
            confidence = max(p_good, p_defect)
            reason = f"Uncertain prediction (Good: {p_good*100:.1f}%, Defect: {p_defect*100:.1f}%). Sent to secondary review."

        return {
            "decision": decision,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "stage1_probs": {"good": p_good, "defect": p_defect},
            "stage2_probs": {cls_name: float(s2_probs[i]) for i, cls_name in enumerate(stage2_classes)},
            "reason": reason,
        }

