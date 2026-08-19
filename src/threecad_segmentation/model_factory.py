from __future__ import annotations


def build_model(model_name: str, pretrained: bool = True):
    key = model_name.lower().strip()
    if key in {"unet", "unet_r18", "unet-resnet18"}:
        from unet_r18 import UNetResNet18
        return UNetResNet18(pretrained=pretrained)
    if key in {"segformer", "segformer_b0", "segformer-b0"}:
        from segformer_b0 import SegFormerB0Binary
        return SegFormerB0Binary(pretrained=pretrained)
    if key in {"vmamba", "vmamba_t", "vmamba_t_s2l5", "vmamba-t"}:
        from vmamba_t import build_vmamba_t_binary
        return build_vmamba_t_binary(pretrained=pretrained)
    raise ValueError(f"Unknown model_name={model_name!r}")
