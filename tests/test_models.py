"""
Unit Tests for DenseNet Models
Owner: Mohammad Abdul Rehman (Team & Paper Lead)

Tests:
1. DenseNet forward output shape on small_input (CIFAR-10) and ImageNet stem.
2. Channel growth formula: C_in(l) = k0 + (l - 1) * k.
3. Transition compression: out_channels = floor(theta * m).
4. Bottleneck layer output shape and channels.
5. SimpleCNN and ResNet-56 baselines output shapes.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.models.dense_layer import _DenseLayer
from src.models.dense_block import _DenseBlock
from src.models.transition import _Transition
from src.models.densenet import (
    DenseNet,
    densenet40_cifar,
    densenet_bc_100_cifar,
    densenet121_custom,
)
from src.models.resnet_cifar import resnet56_cifar
from src.models.simple_cnn import SimpleCNN


def test_dense_layer_shapes_standard():
    """Verify standard _DenseLayer without bottleneck produces exactly growth_rate channels."""
    in_channels = 16
    growth_rate = 12
    layer = _DenseLayer(in_channels=in_channels, growth_rate=growth_rate, bottleneck=False)

    x = torch.randn(2, in_channels, 32, 32)
    out = layer(x)

    assert out.shape == (2, growth_rate, 32, 32), (
        f"Expected shape (2, {growth_rate}, 32, 32), got {out.shape}"
    )


def test_dense_layer_shapes_bottleneck():
    """Verify DenseNet-B bottleneck layer produces exactly growth_rate channels."""
    in_channels = 24
    growth_rate = 12
    layer = _DenseLayer(in_channels=in_channels, growth_rate=growth_rate, bottleneck=True)

    x = torch.randn(2, in_channels, 32, 32)
    out = layer(x)

    assert out.shape == (2, growth_rate, 32, 32), (
        f"Expected bottleneck shape (2, {growth_rate}, 32, 32), got {out.shape}"
    )


def test_dense_block_channel_growth():
    """Verify channel growth formula: C_out = k0 + L * k."""
    k0 = 16
    k = 12
    L = 6
    block = _DenseBlock(num_layers=L, in_channels=k0, growth_rate=k, bottleneck=False)

    assert block.out_channels == k0 + L * k, (
        f"Expected out_channels={k0 + L * k}, got {block.out_channels}"
    )

    x = torch.randn(2, k0, 32, 32)
    out = block(x)

    expected_shape = (2, k0 + L * k, 32, 32)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"


def test_transition_compression_theta():
    """Verify transition layer channel compression: floor(theta * m) and spatial downsampling."""
    in_channels = 100
    thetas = [0.5, 0.75, 1.0]

    for theta in thetas:
        expected_out = int(math.floor(in_channels * theta))
        trans = _Transition(in_channels=in_channels, theta=theta)

        assert trans.out_channels == expected_out, (
            f"For theta={theta}, expected {expected_out}, got {trans.out_channels}"
        )

        x = torch.randn(2, in_channels, 32, 32)
        out = trans(x)

        expected_shape = (2, expected_out, 16, 16)
        assert out.shape == expected_shape, (
            f"Expected transition output {expected_shape}, got {out.shape}"
        )


def test_densenet40_forward():
    """Verify end-to-end forward pass of DenseNet-40 on CIFAR-10 input."""
    model = densenet40_cifar(growth_rate=12, num_classes=10, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)

    assert logits.shape == (2, 10), f"Expected logits shape (2, 10), got {logits.shape}"


def test_densenet_bc_100_forward():
    """Verify end-to-end forward pass of DenseNet-BC-100 on CIFAR-10 input."""
    model = densenet_bc_100_cifar(growth_rate=12, num_classes=10, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)

    assert logits.shape == (2, 10), f"Expected logits shape (2, 10), got {logits.shape}"


def test_simple_cnn_resolutions():
    """Verify SimpleCNN functions on 64x64, 28x28, and 128x128 1-channel images."""
    model = SimpleCNN(in_channels=1, num_classes=2)

    for res in [28, 64, 128]:
        x = torch.randn(2, 1, res, res)
        out = model(x)
        assert out.shape == (2, 2), f"Failed for resolution {res}x{res}, got shape {out.shape}"


def test_resnet56_forward():
    """Verify ResNet-56 baseline forward pass on CIFAR-10."""
    model = resnet56_cifar(num_classes=10, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)

    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
