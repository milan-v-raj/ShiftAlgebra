"""Five lightweight CIFAR architecture families for the Phase-2 CPU bank."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _pooled(x: Tensor) -> Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


class ShallowCNN(nn.Module):
    def __init__(self, width: int = 16, classes: int = 10) -> None:
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, width, 5, padding=2), nn.BatchNorm2d(width), nn.ReLU()),
            nn.Sequential(nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU()),
            nn.Sequential(nn.Conv2d(2 * width, 4 * width, 3, padding=1), nn.BatchNorm2d(4 * width), nn.ReLU()),
        ])
        self.classifier = nn.Linear(4 * width, classes)

    def forward_with_layers(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        x1 = self.features[0](images)
        x2 = self.features[1](F.max_pool2d(x1, 2))
        x3 = self.features[2](F.max_pool2d(x2, 2))
        layers = [_pooled(x1), _pooled(x2), _pooled(x3)]
        return self.classifier(layers[-1]), layers

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_layers(images)[0]


class ResidualBlock(nn.Module):
    def __init__(self, input_width: int, output_width: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_width, output_width, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_width)
        self.conv2 = nn.Conv2d(output_width, output_width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_width)
        self.skip = (
            nn.Identity() if input_width == output_width and stride == 1
            else nn.Sequential(
                nn.Conv2d(input_width, output_width, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_width),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = F.silu(self.bn1(self.conv1(x)))
        return F.silu(self.bn2(self.conv2(x)) + residual)


class TinyResNet(nn.Module):
    def __init__(self, width: int = 8, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.SiLU()
        )
        self.block1 = ResidualBlock(width, width)
        self.block2 = ResidualBlock(width, 2 * width, stride=2)
        self.block3 = ResidualBlock(2 * width, 4 * width, stride=2)
        self.classifier = nn.Linear(4 * width, classes)

    def forward_with_layers(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        x1 = self.block1(self.stem(images))
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        layers = [_pooled(x1), _pooled(x2), _pooled(x3)]
        return self.classifier(layers[-1]), layers

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_layers(images)[0]


class DepthwiseBlock(nn.Module):
    def __init__(self, input_width: int, output_width: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_width, input_width, 3, stride=stride, padding=1, groups=input_width, bias=False),
            nn.BatchNorm2d(input_width),
            nn.SiLU(),
            nn.Conv2d(input_width, output_width, 1, bias=False),
            nn.BatchNorm2d(output_width),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class TinyDepthwiseNet(nn.Module):
    def __init__(self, width: int = 16, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.SiLU()
        )
        self.block1 = DepthwiseBlock(width, width)
        self.block2 = DepthwiseBlock(width, 2 * width, stride=2)
        self.block3 = DepthwiseBlock(2 * width, 4 * width, stride=2)
        self.classifier = nn.Linear(4 * width, classes)

    def forward_with_layers(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        x1 = self.block1(self.stem(images))
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        layers = [_pooled(x1), _pooled(x2), _pooled(x3)]
        return self.classifier(layers[-1]), layers

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_layers(images)[0]


class PatchConvNet(nn.Module):
    def __init__(self, width: int = 24, classes: int = 10) -> None:
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(3, width, 4, stride=4, bias=False), nn.BatchNorm2d(width), nn.GELU()
        )
        self.mix1 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.Conv2d(width, 2 * width, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.GELU(),
        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, groups=2 * width, bias=False),
            nn.Conv2d(2 * width, 3 * width, 1, bias=False),
            nn.BatchNorm2d(3 * width),
            nn.GELU(),
        )
        self.classifier = nn.Linear(3 * width, classes)

    def forward_with_layers(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        x1 = self.patch(images)
        x2 = self.mix1(x1)
        x3 = self.mix2(x2)
        layers = [_pooled(x1), _pooled(x2), _pooled(x3)]
        return self.classifier(layers[-1]), layers

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_layers(images)[0]


class TinyMLPConv(nn.Module):
    """Pointwise-heavy family with a deliberately different spatial bias."""

    def __init__(self, width: int = 20, classes: int = 10) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(width, width, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.point1 = nn.Sequential(nn.Conv2d(width, 3 * width, 1), nn.ReLU())
        self.point2 = nn.Sequential(nn.Conv2d(3 * width, 3 * width, 1), nn.ReLU())
        self.classifier = nn.Linear(3 * width, classes)

    def forward_with_layers(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        x1 = self.down(images)
        x2 = self.point1(x1)
        x3 = self.point2(x2)
        layers = [_pooled(x1), _pooled(x2), _pooled(x3)]
        return self.classifier(layers[-1]), layers

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_layers(images)[0]


FAMILY_WIDTHS = {
    "shallow": 16,
    "resnet": 8,
    "depthwise": 16,
    "patchconv": 24,
    "pointwise": 20,
}


def make_model(family: str) -> nn.Module:
    if family == "shallow":
        return ShallowCNN(FAMILY_WIDTHS[family])
    if family == "resnet":
        return TinyResNet(FAMILY_WIDTHS[family])
    if family == "depthwise":
        return TinyDepthwiseNet(FAMILY_WIDTHS[family])
    if family == "patchconv":
        return PatchConvNet(FAMILY_WIDTHS[family])
    if family == "pointwise":
        return TinyMLPConv(FAMILY_WIDTHS[family])
    raise KeyError(family)
