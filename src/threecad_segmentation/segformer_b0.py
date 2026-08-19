"""SegFormer-B0 binary segmentation with ImageNet-only pretrained MiT-B0 encoder.

For fair architecture comparison, the main experiment does NOT start from an
ADE20K semantic-segmentation checkpoint. U-Net/ResNet18 and VMamba-T also use
ImageNet classification pretraining, so SegFormer uses nvidia/mit-b0 encoder
pretraining and a freshly initialized binary decode head.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel


class SegFormerB0Binary(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            config = SegformerConfig.from_pretrained(
                "nvidia/mit-b0",
                num_labels=1,
                id2label={0: "defect"},
                label2id={"defect": 0},
            )
            self.net = SegformerForSemanticSegmentation(config)
            self.net.segformer = SegformerModel.from_pretrained("nvidia/mit-b0")
        else:
            config = SegformerConfig(
                num_labels=1,
                id2label={0: "defect"},
                label2id={"defect": 0},
            )
            self.net = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def encoder_parameters(self):
        return self.net.segformer.parameters()

    def decoder_parameters(self):
        return self.net.decode_head.parameters()


def build_segformer_b0(pretrained: bool = True) -> SegFormerB0Binary:
    return SegFormerB0Binary(pretrained=pretrained)
