"""
Owner: Muhammad Wajahat (Data & Pipeline Lead)
Module: _Transition
Reference: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)

Key Formulation:
    Transition layers connect two adjacent dense blocks.
    To reduce feature map sizes:
        BN -> 1x1 Conv -> 2x2 AvgPool (stride 2)
    In DenseNet-C, compression factor theta in (0, 1] compresses the channel count:
        out_channels = floor(theta * in_channels)
    In DenseNet-BC, theta = 0.5 (halving the feature channels).
    If theta = 1.0, channel count is unchanged.
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class _Transition(nn.Sequential):
    """
    Transition Layer between two dense blocks for spatial downsampling and channel compression.

    Attributes:
        in_channels (int): Input feature map count m from the preceding dense block.
        out_channels (int): Compressed output channel count floor(theta * m).
        theta (float): Compression factor theta in (0, 1].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        theta: float = 0.5,
    ) -> None:
        super(_Transition, self).__init__()
        self.in_channels = in_channels
        self.theta = float(theta)

        assert 0.0 < self.theta <= 1.0, f"Compression factor theta must be in (0, 1], got {theta}"

        if out_channels is None:
            # DenseNet-C rule: floor(theta * m)
            self.out_channels = int(math.floor(self.in_channels * self.theta))
        else:
            self.out_channels = int(out_channels)

        assert self.out_channels > 0, (
            f"out_channels must be positive, but got {self.out_channels} (in_channels={in_channels}, theta={theta})"
        )

        # -------------------------------------------------------------
        # Transition Operations: BN -> ReLU -> 1x1 Conv -> 2x2 AvgPool
        # -------------------------------------------------------------
        self.add_module("norm", nn.BatchNorm2d(self.in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming normal initialization for conv and unit initialization for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with explicit shape assertions.

        Input:
            x: Tensor of shape (B, in_channels, H, W)
        Output:
            Tensor of shape (B, out_channels, H // 2, W // 2)
        """
        assert x.dim() == 4, f"Expected 4D tensor (B, C, H, W), got {x.shape}"
        assert x.size(1) == self.in_channels, (
            f"Expected in_channels={self.in_channels}, got {x.size(1)}"
        )

        b, c_in, h, w = x.shape
        out = super(_Transition, self).forward(x)

        expected_h = h // 2
        expected_w = w // 2
        assert out.shape == (b, self.out_channels, expected_h, expected_w), (
            f"Transition output shape mismatch: expected {(b, self.out_channels, expected_h, expected_w)}, got {out.shape}"
        )
        return out

    def count_parameters_hand(self) -> int:
        """
        Closed-form hand parameter calculation:
            BN: 2 * in_channels
            1x1 Conv: in_channels * out_channels
            Total = 2 * in_channels + in_channels * out_channels
        """
        bn_params = 2 * self.in_channels
        conv1x1_params = self.in_channels * self.out_channels
        return bn_params + conv1x1_params
