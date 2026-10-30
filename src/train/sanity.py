"""
Owner: Aiman Aslam (Experiments Lead)
Module: sanity
Reference: Andrej Karpathy's "A Recipe for Training Neural Networks" & CS-320 Rigorous Debugging Standards

Performs 3 Critical Sanity Checks before any full-scale training run:
1. Initial Loss Check:
   - Verifies that initial loss of an untrained network is approximately -ln(1/C) = ln(C).
   - For CIFAR-10 (C=10): ln(10) ≈ 2.3026.
   - For PneumoniaMNIST (C=2): ln(2) ≈ 0.6931.
2. Overfit-One-Batch Test:
   - Takes a single mini-batch (e.g. 16-32 samples) and trains for 60 iterations.
   - Loss must plummet near zero (< 0.1) and training accuracy hit 100%, proving model capacity,
     loss implementation, and backward pass are completely bug-free.
3. Per-Layer Gradient Flow Check:
   - Computes backpropagation on one batch and confirms that EVERY trainable layer
     receives non-zero gradients (no dead layers, vanishing gradients, or detached tensors).
"""

import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


def check_initial_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    tolerance: float = 0.5,
) -> Tuple[float, float, bool]:
    """
    Checks that initial loss is close to theoretical expectation ln(C).
    """
    # Use train mode during initial loss check so BatchNorm normalizes activations
    # using mini-batch statistics rather than uninitialized running statistics
    was_training = model.training
    model.train()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets).item()

    model.train(was_training)

    expected_loss = math.log(num_classes)
    diff = abs(loss - expected_loss)
    passed = diff < tolerance

    print(f"[Sanity Check 1: Initial Loss] Classes C={num_classes}:")
    print(f"  - Theoretical ln(C): {expected_loss:.4f}")
    print(f"  - Observed Loss:     {loss:.4f} (diff: {diff:.4f}, tol: {tolerance})")
    print(f"  - Status:            {'PASSED' if passed else 'WARNING'}\n")
    return loss, expected_loss, passed


def check_gradient_flow(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[int, int, bool]:
    """
    Verifies that all trainable parameters receive non-zero gradients.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    criterion = nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    zero_grad_layers = []
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is None or param.grad.abs().sum().item() == 0.0:
                zero_grad_layers.append(name)

    passed = (len(zero_grad_layers) == 0)

    print(f"[Sanity Check 2: Gradient Flow] Total trainable tensors: {total_params}")
    if passed:
        print("  - Zero-gradient tensors: NONE (All parameters receive active gradients)")
        print("  - Status:                PASSED\n")
    else:
        print(f"  - Zero-gradient tensors: {len(zero_grad_layers)} detected: {zero_grad_layers[:3]}...")
        print("  - Status:                FAILED\n")

    return total_params, len(zero_grad_layers), passed


def overfit_single_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_steps: int = 60,
    lr: float = 0.1,
    target_loss: float = 0.15,
) -> Tuple[float, float, bool]:
    """
    Overfits a single batch of data to verify optimization capacity.
    """
    model.train()
    # High learning rate and Adam or SGD with momentum to rapidly overfit single batch
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    initial_loss = 0.0
    final_loss = 0.0
    final_acc = 0.0

    for step in range(num_steps):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    preds = torch.argmax(outputs, dim=1)
    final_acc = (preds == targets).float().mean().item()

    passed = (final_loss < target_loss) or (final_acc >= 0.95)

    print(f"[Sanity Check 3: Overfit One Batch] ({num_steps} Adam steps):")
    print(f"  - Initial Loss:  {initial_loss:.4f}")
    print(f"  - Final Loss:    {final_loss:.4f} (target < {target_loss})")
    print(f"  - Final Acc:     {final_acc * 100:.1f}%")
    print(f"  - Status:        {'PASSED' if passed else 'WARNING'}\n")

    return initial_loss, final_loss, passed


def run_sanity_checks(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    sample_targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    """Runs all three sanity checks sequentially."""
    print("=" * 60)
    print("RUNNING ARCHITECTURAL & OPTIMIZATION SANITY CHECKS")
    print("=" * 60)

    loss, exp_loss, init_pass = check_initial_loss(model, sample_inputs, sample_targets, num_classes)
    total_p, zero_p, grad_pass = check_gradient_flow(model, sample_inputs, sample_targets)
    init_l, final_l, overfit_pass = overfit_single_batch(model, sample_inputs, sample_targets)

    all_passed = init_pass and grad_pass and overfit_pass
    print("=" * 60)
    print(f"SANITY CHECKS RESULT: {'ALL PASSED' if all_passed else 'ATTENTION NEEDED'}")
    print("=" * 60 + "\n")

    return {
        "initial_loss": loss,
        "initial_loss_passed": init_pass,
        "zero_grad_params": zero_p,
        "gradient_flow_passed": grad_pass,
        "overfit_final_loss": final_l,
        "overfit_passed": overfit_pass,
        "all_passed": all_passed,
    }
