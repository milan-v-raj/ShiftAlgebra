"""Two lightweight architecture families absent from Phase-2 development."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVGG(nn.Module):
    def __init__(self, width: int = 12, classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1), nn.BatchNorm2d(4 * width), nn.ReLU(),
        )
        self.classifier = nn.Linear(4 * width, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))


class DenseLayer(nn.Module):
    def __init__(self, channels: int, growth: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, growth, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.layer(x)], dim=1)


class TinyDenseNet(nn.Module):
    def __init__(self, width: int = 12, growth: int = 8, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, padding=1, bias=False)
        channels = width
        blocks = []
        for stage in range(3):
            for _ in range(2):
                blocks.append(DenseLayer(channels, growth))
                channels += growth
            if stage < 2:
                reduced = max(width, channels // 2)
                blocks.append(nn.Sequential(
                    nn.BatchNorm2d(channels), nn.ReLU(),
                    nn.Conv2d(channels, reduced, 1, bias=False), nn.AvgPool2d(2),
                ))
                channels = reduced
        self.features = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm(self.features(self.stem(x))))
        return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))


FAMILIES = ("vgg_holdout", "dense_holdout")


def make_model(family: str) -> nn.Module:
    if family == "vgg_holdout":
        return TinyVGG()
    if family == "dense_holdout":
        return TinyDenseNet()
    raise KeyError(family)
