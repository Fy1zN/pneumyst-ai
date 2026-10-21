"""DenseNet-121 with a new output layer."""
from __future__ import annotations

from torch import nn
from torchvision import models


def build_model(num_outputs: int, pretrained: bool = True, memory_efficient: bool = True) -> nn.Module:
    """ImageNet DenseNet-121; memory_efficient trades ~15-20% speed for much less GPU memory (4 GB cards)."""
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights, memory_efficient=memory_efficient)
    model.classifier = nn.Linear(model.classifier.in_features, num_outputs)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable