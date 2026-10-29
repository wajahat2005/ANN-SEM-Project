"""
Owner: Mohammad Abdul Rehman (Team & Paper Lead)
Module: DenseNet
Reference: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)

This module implements the complete Densely Connected Convolutional Network from scratch,
without relying on torchvision's DenseNet implementation.

Architectural Highlights from Huang et al.:
1. CIFAR Setup:
   - 3 dense blocks with equal number of layers per block.
   - Feature map sizes: 32x32 -> 16x16 -> 8x8.
   - Initial convolution:
       * 16 channels for standard DenseNet.
       * 2*k channels for DenseNet-BC.
   - Global average pooling followed by a linear classification head.
2. ImageNet Setup:
   - 7x7 initial convolution with stride 2 and 64 channels, followed by 3x3 MaxPool (stride 2).
   - 4 dense blocks with transition layers between each consecutive block.
   - Supports DenseNet-121, DenseNet-169, DenseNet-201, DenseNet-264 configs.
"""

import math
from typing import List, Tuple, Union, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dense_block import _DenseBlock
from src.models.transition import _Transition


class DenseNet(nn.Module):
    """
    Densely Connected Convolutional Network (DenseNet).

    Args:
        growth_rate (int): Growth rate k (channels added per layer). Default: 12.
        block_config (tuple of int): Number of layers in each dense block.
            Default (12, 12, 12) for DenseNet-40.
        num_init_features (int): Number of filters in the initial convolution layer.
            Default: 16 (or 2*k for BC).
        bottleneck (bool): If True, use DenseNet-B bottleneck structure. Default: False.
        theta (float): Compression factor in transition layers (0 < theta <= 1.0).
            Default: 1.0 (no compression); 0.5 in DenseNet-C / BC.
        drop_rate (float): Dropout probability p. Default: 0.0.
        in_channels (int): Input image channels (3 for CIFAR/RGB, 1 for PneumoniaMNIST).
        num_classes (int): Number of output classes (10 for CIFAR-10, 2 for PneumoniaMNIST).
        small_input (bool): If True, uses the CIFAR stem (3x3 conv, stride 1, no initial maxpool).
            If False, uses the ImageNet stem (7x7 conv, stride 2, maxpool stride 2).
        memory_efficient (bool): If True, uses gradient checkpointing inside dense layers.
    """

    def __init__(
        self,
        growth_rate: int = 12,
        block_config: Tuple[int, ...] = (12, 12, 12),
        num_init_features: Optional[int] = None,
        bottleneck: bool = False,
        theta: float = 1.0,
        drop_rate: float = 0.0,
        in_channels: int = 3,
        num_classes: int = 10,
        small_input: bool = True,
        memory_efficient: bool = False,
    ) -> None:
        super(DenseNet, self).__init__()
        self.growth_rate = int(growth_rate)
        self.block_config = tuple(block_config)
        self.bottleneck = bool(bottleneck)
        self.theta = float(theta)
        self.drop_rate = float(drop_rate)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.small_input = bool(small_input)
        self.memory_efficient = bool(memory_efficient)

        # Determine initial channels
        if num_init_features is None:
            if small_input:
                # Paper Section 3: "initial convolution with 16 channels (or 2k for DenseNet-BC)"
                num_init_features = 2 * self.growth_rate if self.bottleneck and self.theta < 1.0 else 16
            else:
                num_init_features = 64  # Standard ImageNet stem
        self.num_init_features = int(num_init_features)

        # -------------------------------------------------------------
        # 1. Initial Convolution Stem
        # -------------------------------------------------------------
        if self.small_input:
            # CIFAR stem: 3x3 Conv with stride 1, preserving 32x32 resolution
            # In paper: "The first layer of DenseNet is a convolution with 16 (or 2k) channels."
            self.features = nn.Sequential()
            self.features.add_module(
                "conv0",
                nn.Conv2d(
                    self.in_channels,
                    self.num_init_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
        else:
            # ImageNet stem: 7x7 Conv with stride 2 + BN + ReLU + 3x3 MaxPool (stride 2)
            self.features = nn.Sequential()
            self.features.add_module(
                "conv0",
                nn.Conv2d(
                    self.in_channels,
                    self.num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
            )
            self.features.add_module("norm0", nn.BatchNorm2d(self.num_init_features))
            self.features.add_module("relu0", nn.ReLU(inplace=True))
            self.features.add_module("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # -------------------------------------------------------------
        # 2. Dense Blocks and Transition Layers
        # -------------------------------------------------------------
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=self.growth_rate,
                bottleneck=self.bottleneck,
                drop_rate=self.drop_rate,
                memory_efficient=self.memory_efficient,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = block.out_channels

            # Add a transition layer between dense blocks (except after the final block)
            if i != len(self.block_config) - 1:
                trans = _Transition(
                    in_channels=num_features,
                    theta=self.theta,
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = trans.out_channels

        # -------------------------------------------------------------
        # 3. Final Pre-Activation and Classifier Head
        # -------------------------------------------------------------
        # Final BatchNorm + ReLU before global average pooling
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.features.add_module("relu_final", nn.ReLU(inplace=True))

        self.num_final_features = num_features
        self.classifier = nn.Linear(self.num_final_features, self.num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming He normal initialization for Conv layers, ones for BN weight, zeros for BN bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with comprehensive shape checks.

        Input:
            x: Tensor of shape (B, in_channels, H, W)
        Output:
            Logits of shape (B, num_classes)
        """
        assert x.dim() == 4, f"Expected 4D input tensor, got shape {x.shape}"
        assert x.size(1) == self.in_channels, (
            f"Expected in_channels={self.in_channels}, but got input with {x.size(1)} channels."
        )

        b = x.size(0)
        features = self.features(x)

        # Global Average Pooling: (B, num_final_features, H', W') -> (B, num_final_features, 1, 1)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)

        assert out.shape == (b, self.num_final_features), (
            f"Flattened feature shape mismatch: expected {(b, self.num_final_features)}, got {out.shape}"
        )

        logits = self.classifier(out)
        assert logits.shape == (b, self.num_classes), (
            f"Classifier logits shape mismatch: expected {(b, self.num_classes)}, got {logits.shape}"
        )
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the spatial feature maps before global pooling (used for Grad-CAM)."""
        return self.features(x)

    def count_parameters(self) -> int:
        """Returns total trainable parameters via sum(p.numel())."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_hand(self) -> int:
        """
        Calculates total parameters by analytically summing each component:
        - Initial conv: in_channels * num_init_features * (K_h * K_w) [+ BN if ImageNet stem]
        - Each DenseBlock: hand sum of each _DenseLayer
        - Each Transition: hand sum of BN + 1x1 Conv
        - Final BN: 2 * num_final_features
        - Linear head: num_final_features * num_classes + num_classes
        """
        total = 0

        # Initial stem
        if self.small_input:
            total += self.in_channels * self.num_init_features * 9  # 3x3 Conv
        else:
            total += self.in_channels * self.num_init_features * 49  # 7x7 Conv
            total += 2 * self.num_init_features  # norm0 BN

        # Blocks and transitions
        for name, module in self.features.named_children():
            if isinstance(module, _DenseBlock):
                total += module.count_parameters_hand()
            elif isinstance(module, _Transition):
                total += module.count_parameters_hand()

        # Final BN
        total += 2 * self.num_final_features

        # Classifier head
        total += self.num_final_features * self.num_classes + self.num_classes
        return total


# -------------------------------------------------------------------------
# Factory Constructors for Exact Paper Configurations
# -------------------------------------------------------------------------

def densenet40_cifar(
    growth_rate: int = 12,
    num_classes: int = 10,
    in_channels: int = 3,
    drop_rate: float = 0.0,
) -> DenseNet:
    """
    DenseNet-40 (k=12) for CIFAR-10.
    Paper: 40 layers total = 1 (initial conv) + 3 blocks * 12 layers (36 convs) + 2 transition convs + 1 linear head.
    Bottleneck: False, Compression theta: 1.0.
    Paper parameter count: ~1.0M parameters.
    Paper error on CIFAR-10+: 5.24%.
    """
    return DenseNet(
        growth_rate=growth_rate,
        block_config=(12, 12, 12),
        num_init_features=16,
        bottleneck=False,
        theta=1.0,
        drop_rate=drop_rate,
        in_channels=in_channels,
        num_classes=num_classes,
        small_input=True,
    )


def densenet_bc_100_cifar(
    growth_rate: int = 12,
    num_classes: int = 10,
    in_channels: int = 3,
    drop_rate: float = 0.0,
) -> DenseNet:
    """
    DenseNet-BC-100 (k=12) for CIFAR-10.
    Paper: (100 - 4)/6 = 16 bottleneck layers per block (each has 1x1 and 3x3 convs = 32 convs per block).
    Bottleneck: True, Compression theta: 0.5.
    Initial features: 2*k = 24.
    Paper parameter count: ~0.8M parameters.
    Paper error on CIFAR-10+: 4.51%.
    """
    return DenseNet(
        growth_rate=growth_rate,
        block_config=(16, 16, 16),
        num_init_features=2 * growth_rate,
        bottleneck=True,
        theta=0.5,
        drop_rate=drop_rate,
        in_channels=in_channels,
        num_classes=num_classes,
        small_input=True,
    )


def densenet121_custom(
    num_classes: int = 1000,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    small_input: bool = False,
) -> DenseNet:
    """
    DenseNet-121 (k=32) ImageNet-style configuration.
    Dense blocks: (6, 12, 24, 16), Bottleneck: True, Compression theta: 0.5.
    Can be parameter-compared directly with torchvision.models.densenet121.
    """
    return DenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bottleneck=True,
        theta=0.5,
        drop_rate=drop_rate,
        in_channels=in_channels,
        num_classes=num_classes,
        small_input=small_input,
    )
