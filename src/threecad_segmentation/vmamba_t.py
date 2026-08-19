"""Binary segmentation model: official VMamba-T (s2l5) backbone + lightweight FPN decoder.

Requires the official MzeroMiko/VMamba repository to be cloned so that:
    third_party/VMamba/vmamba.py
is importable.

The backbone configuration mirrors the official `vmamba_tiny_s2l5` definition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

_CANDIDATES = []
if os.environ.get("VMAMBA_REPO"):
    _CANDIDATES.append(Path(os.environ["VMAMBA_REPO"]).expanduser())
_CANDIDATES.extend([
    Path(__file__).resolve().parent / "third_party" / "VMamba",
    Path("/content/TTTN/third_party/VMamba"),
    Path("/content/third_party/VMamba"),
])
_REPO = next((p for p in _CANDIDATES if (p / "vmamba.py").exists()), None)
if _REPO is None:
    searched = "\n".join(str(p / "vmamba.py") for p in _CANDIDATES)
    raise ImportError(
        "Official VMamba repo not found. Searched:\n" + searched +
        "\nRun setup_vmamba_colab.sh or setup_vmamba_kaggle.sh from the project root."
    )
sys.path.insert(0, str(_REPO))

from vmamba import Backbone_VSSM  # type: ignore  # noqa: E402


OFFICIAL_VMAMBA_T_S2L5_CKPT = (
    "https://github.com/MzeroMiko/VMamba/releases/download/"
    "%23v2cls/vssm_tiny_0230_ckpt_epoch_262.pth"
)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        groups = 8 if out_ch % 8 == 0 else 1
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )


class LightweightFPNDecoder(nn.Module):
    """Top-down FPN using all 4 VMamba scales."""

    def __init__(
        self,
        in_channels=(96, 192, 384, 768),
        fpn_channels: int = 128,
    ) -> None:
        super().__init__()

        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, fpn_channels, kernel_size=1) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvGNAct(fpn_channels, fpn_channels, 3) for _ in in_channels]
        )

        self.fuse = nn.Sequential(
            ConvGNAct(fpn_channels * 4, fpn_channels, 3),
            ConvGNAct(fpn_channels, 64, 3),
        )
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features, output_size):
        f1, f2, f3, f4 = features

        p4 = self.lateral[3](f4)
        p3 = self.lateral[2](f3) + F.interpolate(
            p4, size=f3.shape[-2:], mode="bilinear", align_corners=False
        )
        p2 = self.lateral[1](f2) + F.interpolate(
            p3, size=f2.shape[-2:], mode="bilinear", align_corners=False
        )
        p1 = self.lateral[0](f1) + F.interpolate(
            p2, size=f1.shape[-2:], mode="bilinear", align_corners=False
        )

        p1 = self.smooth[0](p1)
        p2 = self.smooth[1](p2)
        p3 = self.smooth[2](p3)
        p4 = self.smooth[3](p4)

        target = p1.shape[-2:]
        multi = torch.cat(
            [
                p1,
                F.interpolate(p2, size=target, mode="bilinear", align_corners=False),
                F.interpolate(p3, size=target, mode="bilinear", align_corners=False),
                F.interpolate(p4, size=target, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        x = self.fuse(multi)
        logits = self.head(x)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


class VMambaTBinary(nn.Module):
    """VMamba-T s2l5 backbone + multi-scale binary segmentation decoder."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # Mirrors official vmamba_tiny_s2l5(channel_first=True).
        self.backbone = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=None,
            norm_layer="ln2d",
            depths=[2, 2, 5, 2],
            dims=96,
            drop_path_rate=0.2,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
        )

        self.decoder = LightweightFPNDecoder(
            in_channels=(96, 192, 384, 768),
            fpn_channels=128,
        )

        if pretrained:
            self._load_official_pretrained()

    def _load_official_pretrained(self) -> None:
        checkpoint = torch.hub.load_state_dict_from_url(
            OFFICIAL_VMAMBA_T_S2L5_CKPT,
            map_location="cpu",
            check_hash=False,
            progress=True,
        )
        state = checkpoint.get("model", checkpoint)
        incompatible = self.backbone.load_state_dict(state, strict=False)

        # Classification-only keys may be unexpected/missing because Backbone_VSSM
        # removes the classifier. That is expected; backbone weights should load.
        print(
            "[VMamba] pretrained loaded | "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        if incompatible.missing_keys:
            print("[VMamba] missing keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("[VMamba] unexpected keys:", incompatible.unexpected_keys)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")

        output_size = images.shape[-2:]
        features = self.backbone(images)

        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise RuntimeError(
                f"Expected 4 VMamba feature maps, got {type(features)} / "
                f"{len(features) if isinstance(features, (list, tuple)) else 'N/A'}"
            )

        return self.decoder(features, output_size)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.decoder.parameters()


def build_vmamba_t_binary(pretrained: bool = True) -> VMambaTBinary:
    return VMambaTBinary(pretrained=pretrained)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vmamba_t_binary(pretrained=False).to(device)
    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
