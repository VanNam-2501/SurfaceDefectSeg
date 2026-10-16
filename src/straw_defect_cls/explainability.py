"""Grad-CAM and Grad-CAM++ (Appendix Baseline for Straw Defect Classification).

IMPORTANT ARCHITECTURAL & METHODOLOGICAL NOTE:
----------------------------------------------
Grad-CAM / Grad-CAM++ is maintained exclusively as an APPENDIX BASELINE
(evaluated on ResNet-18 for the 'Bent' defect class).

Known Limitations on Low-Contrast Geometric Cues:
1. Low-Contrast Cue Diffusion: Thin transparent plastic straws, puncture holes,
   and acute kink folds exhibit subtle intensity contrasts compared to vibrant,
   high-contrast brand graphics printed on the beverage carton. Coarse convolutional
   feature maps (e.g. 7x7 at layer4) tend to bleed across adjacent packaging boundaries.
2. Non-Causal Nature: Saliency heatmaps visualize where gradients concentrate, but
   fail to prove counterfactual causality (i.e. whether removing the straw alone flips
   the model decision vs removing the brand logo).
3. Primary Industrial Standard: For formal quality assurance and certification, the
   thesis establishes Object-Level Occlusion Attribution (`src.straw_defect_cls.explain`)
   as the primary causal proof.
"""
from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAMPlusPlus:
    """Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks (Appendix)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradient: Optional[torch.Tensor] = None
        self.activation: Optional[torch.Tensor] = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate normalized Grad-CAM++ heatmap of shape (H, W) in [0, 1]."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradient  # (1, C, H, W)
        activations = self.activation  # (1, C, H, W)

        # Grad-CAM++ higher-order weighting
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        eps = 1e-7

        # Alpha weights
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha = grad_2 / (2.0 * grad_2 + sum_activations * grad_3 + eps)
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        # Positive gradient contributions
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay(
        self,
        original_image: Image.Image | np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """Overlay heatmap onto original image."""
        if isinstance(original_image, Image.Image):
            orig_np = np.array(original_image.convert("RGB"))
        else:
            orig_np = original_image.copy()

        h, w = orig_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * orig_np)
        return Image.fromarray(blended)

    def remove_hooks(self):
        """Clean up hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradient: Optional[torch.Tensor] = None
        self.activation: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate normalized heatmap of shape (H, W) with values in [0, 1]."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Global average pooling on gradients
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activation, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay(
        self,
        original_image: Image.Image | np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """Overlay heatmap onto original image."""
        if isinstance(original_image, Image.Image):
            orig_np = np.array(original_image.convert("RGB"))
        else:
            orig_np = original_image.copy()

        h, w = orig_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * orig_np)
        return Image.fromarray(blended)

    def remove_hooks(self):
        """Clean up hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_default_target_layer(model: nn.Module) -> nn.Module:
    """Find the default last conv layer for supported architectures."""
    # ResNet-18
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4[-1]
    # EfficientNet-B0
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        return model.backbone.features[-1]
    # MobileNetV3
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        return model.backbone.features[-1]
    # Fallback to direct attributes
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Could not automatically locate the target convolutional layer for Grad-CAM.")
