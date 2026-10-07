"""ResNet-18 (CIFAR variant) with a width multiplier.

Amendment 1: width=0.25 for the 2-core CPU sandbox. The architecture is the
standard ResNet-18 layout ([2,2,2,2] BasicBlocks over 4 stages) with a
3x3 stride-1 stem and no maxpool (CIFAR convention).
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, width=0.25, num_classes=10):
        super().__init__()
        w = [max(8, int(round(c * width))) for c in (64, 128, 256, 512)]
        self.in_planes = w[0]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.layer1 = self._make_layer(w[0], 2, stride=1)
        self.layer2 = self._make_layer(w[1], 2, stride=2)
        self.layer3 = self._make_layer(w[2], 2, stride=2)
        self.layer4 = self._make_layer(w[3], 2, stride=2)
        self.fc = nn.Linear(w[3], num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = w[3]

    def _make_layer(self, cout, n_blocks, stride):
        layers = [BasicBlock(self.in_planes, cout, stride)]
        self.in_planes = cout
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(cout, cout, 1))
        return nn.Sequential(*layers)

    def features(self, x):
        """Penultimate representation (feature_dim)."""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).flatten(1)
        return out

    def forward(self, x):
        return self.fc(self.features(x))


def resnet18(width=0.25, num_classes=10):
    return ResNet(width=width, num_classes=num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
