"""
Owner: Aiman Aslam (Experiments Lead)
Module: lr_range_test
Reference: Leslie N. Smith, "Cyclical Learning Rates for Training Neural Networks" (WACV 2017)

Purpose:
    Performs a short learning rate range test by exponentially increasing the learning rate
    from min_lr (1e-6) to max_lr (10.0) over a fixed number of iterations.
    Plots the loss vs learning rate curve to identify the steepest negative gradient,
    empirically validating the paper's initial learning rate choice (lr=0.1).
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def run_lr_range_test(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    min_lr: float = 1e-6,
    max_lr: float = 2.0,
    num_iter: int = 100,
    smooth_f: float = 0.05,
    diverge_threshold: float = 5.0,
    save_path: str = "results/lr_range_test.png",
) -> Tuple[List[float], List[float], float]:
    """
    Executes LR range sweep and returns (lrs, smoothed_losses, suggested_lr).
    """
    model.train()
    # Save original model weights to restore after test
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    optimizer = optim.SGD(model.parameters(), lr=min_lr, momentum=0.9, weight_decay=1e-4)

    # Multiplicative factor per step
    mult = (max_lr / min_lr) ** (1.0 / num_iter)

    lrs = []
    losses = []
    best_loss = float("inf")
    current_lr = min_lr
    avg_loss = 0.0
    batch_num = 0

    loader_iter = iter(dataloader)

    print(f"\n[LR Range Test] Sweeping LR from {min_lr:.1e} to {max_lr:.1e} over {num_iter} steps...")

    for i in range(num_iter):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            inputs, targets = next(loader_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)
        if targets.dim() > 1:
            targets = targets.squeeze(-1).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Check for divergence
        loss_val = loss.item()
        if torch.isnan(loss) or loss_val > diverge_threshold * (best_loss if best_loss != float("inf") else 10.0):
            print(f"[LR Range Test] Loss diverged at step {i} (LR = {current_lr:.4f}, Loss = {loss_val:.4f}). Stopping sweep.")
            break

        loss.backward()
        optimizer.step()

        # Exponential moving average smoothing
        if i == 0:
            avg_loss = loss_val
            best_loss = loss_val
        else:
            avg_loss = smooth_f * loss_val + (1 - smooth_f) * avg_loss
            if avg_loss < best_loss:
                best_loss = avg_loss

        lrs.append(current_lr)
        losses.append(avg_loss)

        # Update learning rate for next step
        current_lr *= mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    # Restore original weights
    model.load_state_dict(original_state)
    model.to(device)

    # Suggest learning rate: point of steepest negative gradient
    gradients = np.gradient(losses)
    steepest_idx = np.argmin(gradients)
    suggested_lr = lrs[steepest_idx]

    print(f"[LR Range Test] Minimum Loss: {best_loss:.4f}")
    print(f"[LR Range Test] Suggested LR (steepest descent): {suggested_lr:.4f}")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses, label="Smoothed Loss", color="royalblue", linewidth=2)
    plt.xscale("log")
    plt.scatter([suggested_lr], [losses[steepest_idx]], color="crimson", zorder=5, label=f"Steepest Descent ({suggested_lr:.4f})")
    plt.xlabel("Learning Rate (log scale)", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Learning Rate Range Test", fontsize=13)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[LR Range Test] Plot saved to {save_path}")

    return lrs, losses, suggested_lr
