"""
Owner: Aiman Aslam (Experiments Lead)
Module: ResNetCIFAR
Reference: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)

Purpose:
    Provides the primary architectural baseline (ResNet-56) to compare against DenseNet-40
    on CIFAR-10, directly addressing the central hypothesis of Huang et al. (CVPR 2017):
    DenseNet concatenates feature maps (x_l = H([x_0, ..., x_{l-1}])), while ResNet adds them
    (x_l = H(x_{l-1}) + x_{l-1}). DenseNet achieves equal or better accuracy with substantially
    fewer parameters by reusing features rather than relearning redundant representations.

Structure of ResNet-56 on CIFAR:
    - Initial 3x3 conv with 16 filters (stride 1).
    - 3 stages with feature map sizes [32x32, 16x16, 8x8], each having n=9 BasicBlocks.
    - Total depth = 1 + 2 * (9 + 9 + 9) + 1 = 56 weighted layers.
    - Parameter count: ~0.85M parameters.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    """Custom lambda layer for option A zero-padding identity shortcuts."""

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    Standard residual BasicBlock for CIFAR ResNets (2 3x3 convs with BN and ReLU).
    """
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, option: str = "A") -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                # Option A: Zero-padding shortcut (used in original paper for CIFAR to save parameters)
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                # Option B: 1x1 conv projection shortcut
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    CIFAR-style ResNet with depth = 6*n + 2.
    n=9 yields ResNet-56.
    """

    def __init__(self, block=BasicBlock, num_blocks=(9, 9, 9), num_classes=10, in_channels=3) -> None:
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes

        # Initial stem
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 3 Stages of residual blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Final classification head
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, option="B"))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 32x32 -> 32x32
        out = self.layer2(out)  # 32x32 -> 16x16
        out = self.layer3(out)  # 16x16 -> 8x8
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        assert out.shape == (b, self.num_classes)
        return out


def resnet56_cifar(num_classes: int = 10, in_channels: int = 3) -> ResNetCIFAR:
    """Returns standard ResNet-56 for CIFAR-10 (n=9 per stage, ~0.85M parameters)."""
    return ResNetCIFAR(block=BasicBlock, num_blocks=(9, 9, 9), num_classes=num_classes, in_channels=in_channels)
