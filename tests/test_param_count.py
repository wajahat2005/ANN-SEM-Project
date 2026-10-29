"""
Unit Tests for Analytical Hand Parameter Counts vs PyTorch numel()
Owner: Mohammad Abdul Rehman (Team & Paper Lead)

Tests:
1. Block 1, Layer 1 of DenseNet-40 (C_in=16, k=12) == 1,760 params.
2. Full DenseNet-40 hand parameter formula == sum(p.numel()).
3. Full DenseNet-BC-100 hand parameter formula == sum(p.numel()).
4. Comparison of custom DenseNet-121 with torchvision.models.densenet121.
"""

import pytest
import torch
import torchvision.models as tv_models

from src.models.dense_layer import _DenseLayer
from src.models.densenet import (
    DenseNet,
    densenet40_cifar,
    densenet_bc_100_cifar,
    densenet121_custom,
)
from src.utils.param_count import (
    verify_block1_layer1_params,
    verify_densenet40_params,
    verify_densenet_bc_100_params,
    compare_with_torchvision_densenet121,
)


def test_block1_layer1_exact_1760():
    """Verify that Block 1, Layer 1 of DenseNet-40 equals exactly 1,760 parameters."""
    hand, runtime, match = verify_block1_layer1_params()
    assert hand == 1760
    assert runtime == 1760
    assert match is True


def test_densenet40_hand_vs_numel():
    """Verify that analytical hand count equals sum(p.numel()) for DenseNet-40."""
    hand, runtime, match = verify_densenet40_params()
    assert match is True
    # DenseNet-40 has approx 1.02M params
    assert 1_000_000 < runtime < 1_050_000, f"Expected ~1.0M params, got {runtime}"


def test_densenet_bc_100_hand_vs_numel():
    """Verify that analytical hand count equals sum(p.numel()) for DenseNet-BC-100."""
    hand, runtime, match = verify_densenet_bc_100_params()
    assert match is True
    # DenseNet-BC-100 has approx 0.77M params
    assert 700_000 < runtime < 850_000, f"Expected ~0.8M params, got {runtime}"


def test_densenet121_torchvision_match():
    """
    Compare custom DenseNet-121 with torchvision.models.densenet121.
    Both should have the exact same number of parameters for the 1000-class ImageNet configuration.
    """
    res = compare_with_torchvision_densenet121()
    assert res["match"] is True, (
        f"Custom DenseNet-121 ({res['custom_runtime']}) did not match torchvision ({res['torchvision_runtime']})"
    )
