"""
Owner: Mohammad Abdul Rehman (Team & Paper Lead)
Module: _DenseLayer
Reference: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)

Key Formulation:
    x_l = H_l([x_0, x_1, ..., x_{l-1}])
    Where H_l is a composite function of:
        BatchNorm (BN) -> ReLU -> 3x3 Conv (producing k feature maps)
    In DenseNet-B (bottleneck version), a 1x1 convolution is introduced before
    the 3x3 convolution to reduce the number of input feature maps:
        BN -> ReLU -> 1x1 Conv (producing 4*k feature maps) -> BN -> ReLU -> 3x3 Conv
"""

from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    """
    A single dense layer inside a DenseBlock.

    Attributes:
        in_channels (int): Total number of input channels from all preceding layers (k0 + (l-1)*k).
        growth_rate (int): Growth rate k (number of output feature maps added by this layer).
        bottleneck (bool): If True, uses DenseNet-B bottleneck structure (1x1 conv to 4*k channels).
        drop_rate (float): Dropout probability p. In paper, p=0.2 used only without data augmentation.
        memory_efficient (bool): If True, uses torch.utils.checkpoint to save activation memory.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bottleneck: bool = False,
        drop_rate: float = 0.0,
        memory_efficient: bool = False,
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.bottleneck = bottleneck
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

        # -------------------------------------------------------------
        # Architecture Definition according to Huang et al. (CVPR 2017)
        # -------------------------------------------------------------
        if self.bottleneck:
            # DenseNet-B: 1x1 bottleneck producing 4*k feature maps
            # BN -> ReLU -> 1x1 Conv
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(
                in_channels,
                4 * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            # BN -> ReLU -> 3x3 Conv
            self.bn2 = nn.BatchNorm2d(4 * growth_rate)
            self.conv2 = nn.Conv2d(
                4 * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            # Standard DenseNet: BN -> ReLU -> 3x3 Conv
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(
                in_channels,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.drop_rate) if self.drop_rate > 0.0 else None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """He (Kaiming normal) initialization as specified in paper Section 3."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _call_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the composite function H_l on the concatenated feature tensor.

        Expected Tensor Shapes:
            Input x:  (B, in_channels, H, W)
            Output y: (B, growth_rate, H, W)
        """
        assert x.dim() == 4, f"Expected 4D input tensor (B, C, H, W), got shape {x.shape}"
        assert x.size(1) == self.in_channels, (
            f"Expected in_channels={self.in_channels}, but received tensor with {x.size(1)} channels."
        )

        b, c, h, w = x.shape

        if self.bottleneck:
            # Stage 1: Bottleneck 1x1
            # Input:  (B, in_channels, H, W)
            # Output: (B, 4*growth_rate, H, W)
            out = self.conv1(self.relu(self.bn1(x)))
            assert out.shape == (b, 4 * self.growth_rate, h, w), (
                f"Bottleneck output shape mismatch: expected {(b, 4*self.growth_rate, h, w)}, got {out.shape}"
            )

            # Stage 2: 3x3 Convolution
            # Input:  (B, 4*growth_rate, H, W)
            # Output: (B, growth_rate, H, W)
            out = self.conv2(self.relu(self.bn2(out)))
        else:
            # Standard single 3x3 Conv
            # Input:  (B, in_channels, H, W)
            # Output: (B, growth_rate, H, W)
            out = self.conv1(self.relu(self.bn1(x)))

        if self.dropout is not None:
            out = self.dropout(out)

        assert out.shape == (b, self.growth_rate, h, w), (
            f"_DenseLayer output shape mismatch: expected {(b, self.growth_rate, h, w)}, got {out.shape}"
        )
        return out

    def forward(self, prev_features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass accepting either a concatenated tensor or a list of preceding tensors.

        Args:
            prev_features: A single Tensor (B, in_channels, H, W) or list of tensors [x_0, ..., x_{l-1}].

        Returns:
            torch.Tensor: The new feature maps generated by this layer: (B, growth_rate, H, W).
        """
        if isinstance(prev_features, list):
            concatenated = torch.cat(prev_features, dim=1)
        else:
            concatenated = prev_features

        if self.memory_efficient and concatenated.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._call_layer, concatenated, use_reentrant=False)
        else:
            return self._call_layer(concatenated)

    def count_parameters_hand(self) -> int:
        """
        Computes parameter count by closed-form analytical formula.
        Non-bottleneck:
            BN: 2 * in_channels (weight gamma + bias beta)
            Conv3x3: 9 * in_channels * growth_rate (no bias)
            Total = in_channels * (9 * growth_rate + 2)
        Bottleneck:
            BN1: 2 * in_channels
            Conv1x1: in_channels * (4 * growth_rate)
            BN2: 2 * (4 * growth_rate) = 8 * growth_rate
            Conv3x3: (4 * growth_rate) * growth_rate * 9 = 36 * (growth_rate^2)
            Total = 2*in_channels + 4*in_channels*k + 8*k + 36*(k^2)
        """
        c_in = self.in_channels
        k = self.growth_rate
        if not self.bottleneck:
            bn_params = 2 * c_in
            conv3x3_params = 9 * c_in * k
            return bn_params + conv3x3_params
        else:
            bn1_params = 2 * c_in
            conv1x1_params = c_in * (4 * k)
            bn2_params = 2 * (4 * k)
            conv3x3_params = (4 * k) * k * 9
            return bn1_params + conv1x1_params + bn2_params + conv3x3_params
