"""U-Net decoder with an ImageNet-pretrained ResNet18 encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.refine = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


class UNetResNet18(nn.Module):
    """Binary segmentation model returning raw logits, never probabilities."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # 512 -> 256. Keep the pre-maxpool stem as the highest-resolution skip.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.encoder1 = backbone.layer1  # 64 channels, 128x128
        self.encoder2 = backbone.layer2  # 128 channels, 64x64
        self.encoder3 = backbone.layer3  # 256 channels, 32x32
        self.encoder4 = backbone.layer4  # 512 channels, 16x16

        self.decoder4 = DecoderBlock(512, 256, 256)  # 16 -> 32
        self.decoder3 = DecoderBlock(256, 128, 128)  # 32 -> 64
        self.decoder2 = DecoderBlock(128, 64, 64)    # 64 -> 128
        self.decoder1 = DecoderBlock(64, 64, 64)     # 128 -> 256
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_refine = DoubleConv(32, 32)
        self.segmentation_head = nn.Conv2d(32, 1, kernel_size=1)

        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        decoder_modules = (
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.final_up,
            self.final_refine,
            self.segmentation_head,
        )
        for root in decoder_modules:
            for module in root.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected input [B,3,H,W], got {tuple(images.shape)}")
        input_size = images.shape[-2:]

        skip_256 = self.stem(images)
        skip_128 = self.encoder1(self.maxpool(skip_256))
        skip_64 = self.encoder2(skip_128)
        skip_32 = self.encoder3(skip_64)
        bottleneck = self.encoder4(skip_32)

        x = self.decoder4(bottleneck, skip_32)
        x = self.decoder3(x, skip_64)
        x = self.decoder2(x, skip_128)
        x = self.decoder1(x, skip_256)
        x = self.final_refine(self.final_up(x))
        logits = self.segmentation_head(x)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_unet_resnet18(pretrained: bool = True) -> UNetResNet18:
    return UNetResNet18(pretrained=pretrained)
