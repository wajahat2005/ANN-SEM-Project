"""
Owner: Aiman Aslam (Experiments Lead)
Module: SimpleCNN
Reference: Adapted from team's medmnist-download-data-and-model.ipynb notebook

Purpose:
    Serves as the lightweight convolutional baseline for PneumoniaMNIST, representing
    the standard 5-stage Conv-BN-ReLU architecture from the official MedMNIST baseline.
    Adapted specifically to handle 1-channel 64x64 input (as well as 28x28 and 128x128)
    via an adaptive spatial pooling stage before the multi-layer fully-connected head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple 5-layer CNN baseline for PneumoniaMNIST.

    Architecture:
        - Layer 1: Conv(in_channels -> 16, k=3) + BN + ReLU
        - Layer 2: Conv(16 -> 16, k=3) + BN + ReLU + MaxPool(2x2)
        - Layer 3: Conv(16 -> 64, k=3) + BN + ReLU
        - Layer 4: Conv(64 -> 64, k=3) + BN + ReLU
        - Layer 5: Conv(64 -> 64, k=3, pad=1) + BN + ReLU + MaxPool(2x2)
        - AdaptiveAvgPool2d((4, 4)) -> guarantees fixed 64*4*4 feature vector
        - FC Head: Linear(1024 -> 128) + ReLU + Linear(128 -> 128) + ReLU + Linear(128 -> num_classes)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pool ensures exact (B, 64, 4, 4) regardless of whether input is 28x28, 64x64, or 128x128
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected 4D input, got shape {x.shape}"
        assert x.size(1) == self.in_channels, (
            f"Expected in_channels={self.in_channels}, got {x.size(1)}"
        )
        b = x.size(0)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        assert out.shape == (b, self.num_classes)
        return out
