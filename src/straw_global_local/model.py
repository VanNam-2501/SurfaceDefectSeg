"""B0-B6 share a ResNet18; only the declared aggregation varies."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from .data import VARIANTS


class GlobalLocalCNN(nn.Module):
    def __init__(self, variant="B0", pretrained=True, freeze_bn=True, num_classes=5):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown variant")
        self.variant, self.freeze_bn = variant, freeze_bn
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512 if variant in ("B0", "B1", "B2") else 1024, num_classes)
        if variant in ("B4", "B5", "B6"):
            self.local_projection = nn.Linear(512, 128)
            self.score = nn.Linear(128, 1, bias=False)
        if variant in ("B5", "B6"):
            self.global_projection = nn.Linear(512, 128, bias=False)
        if variant == "B6":
            self.position_projection = nn.Linear(4, 128, bias=False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for layer in self.encoder.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()  # Running stats frozen; affine weights still learn.
        return self

    def forward(self, batch, return_attention=False):
        variant = self.variant
        if variant in ("B0", "B1"):
            logits = self.classifier(self.encoder(batch["global"]))
            return (logits, None) if return_attention else logits
        local = batch["local"]
        n, k, channels, height, width = local.shape
        if variant == "B2":
            features = self.encoder(local.reshape(n*k, channels, height, width)).reshape(n, k, -1)
            weights = features.new_full((n, k), 1/k)
            logits = self.classifier(features.mean(1))
        else:
            views = torch.cat([batch["global"][:, None], local], dim=1)
            encoded = self.encoder(views.reshape(n*(k+1), channels, height, width)).reshape(n, k+1, -1)
            global_feature, features = encoded[:, 0], encoded[:, 1:]
            weights = features.new_full((n, k), 1/k)
            if variant in ("B4", "B5", "B6"):
                score_features = self.local_projection(features)
                if variant in ("B5", "B6"):
                    score_features = score_features + self.global_projection(global_feature)[:, None]
                if variant == "B6":
                    score_features = score_features + self.position_projection(batch["coords"])
                weights = self.score(torch.tanh(score_features)).squeeze(-1).softmax(1)
            pooled = (weights[..., None] * features).sum(1)
            logits = self.classifier(torch.cat([global_feature, pooled], dim=1))
        return (logits, weights) if return_attention else logits
