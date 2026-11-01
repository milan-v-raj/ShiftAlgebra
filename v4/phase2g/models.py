"""Six lightweight architecture families absent from all Phase-2F fitting."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        )

    def forward(self, x):
        z = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(z)) + self.skip(x))


class TinyResidual(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.body = nn.Sequential(
            ResidualBlock(16, 16), ResidualBlock(16, 24, 2), ResidualBlock(24, 32, 2)
        )
        self.head = nn.Linear(32, classes)

    def forward_features(self, x):
        return F.adaptive_avg_pool2d(self.body(self.stem(x)), 1).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class TinyDepthwise(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            DepthwiseBlock(16, 24, 2), DepthwiseBlock(24, 40, 2),
            DepthwiseBlock(40, 56), DepthwiseBlock(56, 72, 2),
        )
        self.head = nn.Linear(72, classes)

    def forward_features(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pointwise1 = nn.Linear(channels, 4 * channels)
        self.pointwise2 = nn.Linear(4 * channels, channels)

    def forward(self, x):
        z = self.depthwise(x).permute(0, 2, 3, 1)
        z = self.pointwise2(F.gelu(self.pointwise1(self.norm(z))))
        return x + z.permute(0, 3, 1, 2)


class TinyConvNeXt(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.stem = nn.Conv2d(3, 28, 4, 4)
        self.blocks = nn.Sequential(ConvNeXtBlock(28), ConvNeXtBlock(28), ConvNeXtBlock(28))
        self.norm = nn.LayerNorm(28)
        self.head = nn.Linear(28, classes)

    def forward_features(self, x):
        return self.norm(self.blocks(self.stem(x)).mean((2, 3)))

    def forward(self, x):
        return self.head(self.forward_features(x))


def channel_shuffle(x, groups):
    batch, channels, height, width = x.shape
    return x.view(batch, groups, channels // groups, height, width).transpose(1, 2).reshape(
        batch, channels, height, width
    )


class ShuffleBlock(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        self.first = nn.Conv2d(channels, channels, 1, groups=groups, bias=False)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.second = nn.Conv2d(channels, channels, 1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        z = F.relu(self.first(x))
        z = channel_shuffle(z, self.groups)
        z = self.norm(self.second(self.depthwise(z)))
        return F.relu(x + z)


class TinyShuffle(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.blocks = nn.Sequential(ShuffleBlock(32), ShuffleBlock(32), nn.MaxPool2d(2), ShuffleBlock(32))
        self.head = nn.Linear(32, classes)

    def forward_features(self, x):
        return F.adaptive_avg_pool2d(self.blocks(self.stem(x)), 1).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class DilatedPyramid(nn.Module):
    def __init__(self, in_channels, branch_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, branch_channels, 3, padding=dilation, dilation=dilation)
            for dilation in (1, 2, 3)
        ])

    def forward(self, x):
        return F.relu(torch.cat([branch(x) for branch in self.branches], dim=1))


class TinyDilated(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 15, 3, padding=1), nn.ReLU())
        self.pyramid1 = DilatedPyramid(15, 8)
        self.pyramid2 = DilatedPyramid(24, 12)
        self.head = nn.Linear(36, classes)

    def forward_features(self, x):
        z = F.max_pool2d(self.pyramid1(self.stem(x)), 2)
        return F.adaptive_avg_pool2d(self.pyramid2(z), 1).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, 3, stride, 1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        value, gate = self.conv(x).chunk(2, dim=1)
        return self.norm(value) * torch.sigmoid(gate)


class TinyGated(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.features = nn.Sequential(
            GatedConv(3, 20), nn.ReLU(), GatedConv(20, 28, 2), nn.ReLU(),
            GatedConv(28, 40, 2), nn.ReLU(), GatedConv(40, 48), nn.ReLU(),
        )
        self.head = nn.Linear(48, classes)

    def forward_features(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


FAMILIES = (
    "residual_holdout",
    "depthwise_holdout",
    "convnext_holdout",
    "shuffle_holdout",
    "dilated_holdout",
    "gated_holdout",
)


def make_model(family, classes=100):
    constructors = {
        "residual_holdout": TinyResidual,
        "depthwise_holdout": TinyDepthwise,
        "convnext_holdout": TinyConvNeXt,
        "shuffle_holdout": TinyShuffle,
        "dilated_holdout": TinyDilated,
        "gated_holdout": TinyGated,
    }
    return constructors[family](classes)


def logits_and_features(model, family, x):
    del family
    features = model.forward_features(x)
    return model.head(features), features
