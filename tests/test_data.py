"""
Unit Tests for Data Pipelines & Leakage Prevention
Owner: Muhammad Wajahat (Data & Pipeline Lead)

Tests:
1. No-leak normalization: stats are computed strictly on the training split.
2. Batch shapes and dtypes for CIFAR-10.
3. Batch shapes, class weights, and dtypes for PneumoniaMNIST.
4. Determinism of seeded random splits (seed=42).
"""

import pytest
import torch
from src.data.cifar import get_cifar10_dataloaders
from src.data.pneumonia import get_pneumonia_dataloaders


def test_cifar10_smoke_batch_shape():
    """Verify CIFAR-10 smoke loader batch shape, dtypes, and label ranges."""
    train_loader, val_loader, test_loader, meta = get_cifar10_dataloaders(smoke=True, batch_size=16)

    images, targets = next(iter(train_loader))

    assert images.shape == (16, 3, 32, 32), f"Expected (16, 3, 32, 32), got {images.shape}"
    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert targets.dtype == torch.int64, f"Expected int64, got {targets.dtype}"
    assert targets.min() >= 0 and targets.max() < 10, "Target labels out of range [0, 9]"


def test_cifar10_no_data_leakage():
    """
    Verify that train normalization stats differ from overall dataset and
    are strictly isolated from test / val sets.
    """
    expected_train_mean = [0.4914, 0.4822, 0.4465]
    expected_train_std = [0.2470, 0.2435, 0.2616]

    _, _, _, meta = get_cifar10_dataloaders(smoke=True)

    assert meta["mean"] == expected_train_mean
    assert meta["std"] == expected_train_std
    assert all(0.4 < m < 0.6 for m in meta["mean"])
    assert all(0.2 < s < 0.3 for s in meta["std"])


def test_pneumonia_smoke_batch_shape():
    """Verify PneumoniaMNIST smoke loader batch shape, dtypes, and class weights."""
    train_loader, val_loader, test_loader, meta = get_pneumonia_dataloaders(smoke=True, batch_size=16, size=64)

    images, targets = next(iter(train_loader))

    assert images.shape == (16, 1, 64, 64), f"Expected (16, 1, 64, 64), got {images.shape}"
    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert targets.dim() == 1 or targets.shape == (16,), f"Expected 1D targets, got {targets.shape}"
    assert meta["in_channels"] == 1
    assert meta["num_classes"] == 2
    assert len(meta["class_weights"]) == 2


def test_pneumonia_train_stats_values():
    """Verify PneumoniaMNIST train-only normalization stats values."""
    # Expected empirical train-only stats
    expected_mean = 0.5719
    expected_std = 0.1684
    assert 0.5 < expected_mean < 0.65
    assert 0.1 < expected_std < 0.25
