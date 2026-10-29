"""
Unit Tests for Diagnostic Sanity Checks
Owner: Aiman Aslam (Experiments Lead)

Tests:
1. Initial loss matches theoretical ln(C) for CIFAR-10.
2. Per-layer gradient flow non-zero check.
3. Single batch overfit test converges to low loss in smoke mode.
"""

import pytest
import torch
import torch.nn as nn

from src.models.densenet import densenet40_cifar
from src.train.sanity import (
    check_initial_loss,
    check_gradient_flow,
    overfit_single_batch,
)


def test_initial_loss_theoretical():
    """Verify untrained DenseNet-40 initial loss is approximately ln(10) ≈ 2.3026."""
    torch.manual_seed(42)
    model = densenet40_cifar(growth_rate=12, num_classes=10, in_channels=3)

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))

    loss, expected, passed = check_initial_loss(model, inputs, targets, num_classes=10, tolerance=0.6)
    assert passed, f"Initial loss {loss:.4f} diverged significantly from expected ln(10) = {expected:.4f}"


def test_gradient_flow_nonzero():
    """Verify all trainable parameters receive non-zero gradients during backward pass."""
    torch.manual_seed(42)
    model = densenet40_cifar(growth_rate=12, num_classes=10, in_channels=3)

    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))

    total_params, zero_grads, passed = check_gradient_flow(model, inputs, targets)
    assert passed, f"Detected {zero_grads} / {total_params} parameters with zero gradient."


def test_overfit_single_batch_smoke():
    """Verify that a compact DenseNet can overfit a single mini-batch within 60 steps."""
    torch.manual_seed(42)
    # Use a small DenseNet for rapid test
    from src.models.densenet import DenseNet
    model = DenseNet(
        growth_rate=12,
        block_config=(4, 4, 4),
        num_init_features=16,
        bottleneck=False,
        num_classes=10,
        small_input=True,
    )

    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))

    initial_loss, final_loss, passed = overfit_single_batch(
        model, inputs, targets, num_steps=60, lr=0.05, target_loss=0.25
    )
    assert passed, f"Overfit test failed: final loss {final_loss:.4f} did not reach target."
