"""
Owner: Mohammad Abdul Rehman (Team & Paper Lead)
Module: _DenseBlock
Reference: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)

Key Formulation:
    A DenseBlock comprises L dense layers.
    If the block receives k0 channels, then layer l (for l=1..L) receives:
        C_in(l) = k0 + (l - 1) * k
    Each layer outputs k channels.
    After L layers, the DenseBlock produces:
        C_out = k0 + L * k channels.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from src.models.dense_layer import _DenseLayer


class _DenseBlock(nn.ModuleDict):
    """
    Dense Block containing multiple densely connected convolutional layers.

    Attributes:
        num_layers (int): Number of dense layers inside this block (L).
        in_channels (int): Input channel count entering the dense block (k0).
        growth_rate (int): Growth rate k.
        bottleneck (bool): Whether to use DenseNet-B bottleneck layers.
        drop_rate (float): Dropout probability.
        memory_efficient (bool): Memory efficient checkpointing flag.
    """

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bottleneck: bool = False,
        drop_rate: float = 0.0,
        memory_efficient: bool = False,
    ) -> None:
        super(_DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.bottleneck = bottleneck
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

        # Construct each dense layer with linearly growing input channels:
        # Layer 1: k0
        # Layer 2: k0 + k
        # Layer l: k0 + (l - 1) * k
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            layer = _DenseLayer(
                in_channels=layer_in_channels,
                growth_rate=growth_rate,
                bottleneck=bottleneck,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    @property
    def out_channels(self) -> int:
        """Total output channels leaving this dense block: k0 + L * k."""
        return self.in_channels + self.num_layers * self.growth_rate

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dense block.

        Input:
            init_features: Tensor of shape (B, in_channels, H, W)
        Output:
            Tensor of shape (B, in_channels + num_layers * growth_rate, H, W)
        """
        assert init_features.dim() == 4, f"Expected 4D tensor, got {init_features.shape}"
        assert init_features.size(1) == self.in_channels, (
            f"Expected {self.in_channels} channels at block input, got {init_features.size(1)}"
        )

        b, c_in, h, w = init_features.shape
        features: List[torch.Tensor] = [init_features]

        for name, layer in self.items():
            new_features = layer(features)
            assert new_features.shape == (b, self.growth_rate, h, w), (
                f"Layer {name} output shape mismatch: expected {(b, self.growth_rate, h, w)}, got {new_features.shape}"
            )
            features.append(new_features)

        out = torch.cat(features, dim=1)
        expected_channels = self.out_channels
        assert out.shape == (b, expected_channels, h, w), (
            f"DenseBlock output shape mismatch: expected {(b, expected_channels, h, w)}, got {out.shape}"
        )
        return out

    def get_layer_shapes(self, batch_size: int = 1, h: int = 32, w: int = 32) -> List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
        """
        Returns a list of (layer_name, input_shape, output_shape) for each layer in this block.
        Useful for verification tables and viva explanations.
        """
        shapes = []
        for i in range(self.num_layers):
            c_in = self.in_channels + i * self.growth_rate
            in_shape = (batch_size, c_in, h, w)
            out_shape = (batch_size, self.growth_rate, h, w)
            shapes.append((f"denselayer{i + 1}", in_shape, out_shape))
        return shapes

    def count_parameters_hand(self) -> int:
        """Computes sum of closed-form parameters across all layers in this block."""
        total = 0
        for layer in self.values():
            if isinstance(layer, _DenseLayer):
                total += layer.count_parameters_hand()
        return total
