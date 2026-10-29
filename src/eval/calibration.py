"""
Owner: Aiman Aslam (Experiments Lead)
Module: calibration
Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)

Purpose:
    Provides clinical calibration tools for PneumoniaMNIST:
    1. Calibrate decision threshold on the VALIDATION split to guarantee >= 95% sensitivity
       (minimizing false negatives in clinical triage), then evaluate generalization on the TEST split.
    2. Reliability diagrams and Expected Calibration Error (ECE).
    3. Post-hoc Temperature Scaling to calibrate predicted probabilities.
    4. Sensitivity vs Specificity trade-off visualization curve.
"""

import os
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def calibrate_threshold_for_sensitivity(
    val_targets: np.ndarray,
    val_probs: np.ndarray,
    target_sensitivity: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Finds the optimal decision threshold on the VALIDATION split that achieves >= target_sensitivity
    while maximizing specificity.

    Returns:
        (optimal_threshold, val_sensitivity, val_specificity)
    """
    val_targets = np.asarray(val_targets).squeeze()
    if val_probs.ndim == 2:
        val_probs = val_probs[:, 1]
    val_probs = np.asarray(val_probs).squeeze()

    fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
    # tpr is sensitivity; (1 - fpr) is specificity

    # Filter all thresholds that satisfy Sensitivity >= target_sensitivity
    valid_indices = np.where(tpr >= target_sensitivity)[0]

    if len(valid_indices) == 0:
        # Fallback to minimum threshold (highest sensitivity possible)
        best_idx = np.argmax(tpr)
    else:
        # Among all thresholds with TPR >= target_sensitivity, pick the one with highest specificity (lowest FPR)
        best_idx = valid_indices[np.argmin(fpr[valid_indices])]

    chosen_thresh = float(thresholds[best_idx])
    # Bound threshold to reasonable numerical range
    chosen_thresh = max(1e-4, min(1.0 - 1e-4, chosen_thresh))

    achieved_sens = float(tpr[best_idx])
    achieved_spec = float(1.0 - fpr[best_idx])

    print(f"[Calibration] Chosen threshold on Validation set: {chosen_thresh:.4f}")
    print(f"  - Validation Sensitivity: {achieved_sens * 100:.2f}% (Target: >= {target_sensitivity * 100:.1f}%)")
    print(f"  - Validation Specificity: {achieved_spec * 100:.2f}%")

    return chosen_thresh, achieved_sens, achieved_spec


def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) and binning statistics.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    targets = np.asarray(targets).squeeze()
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
        predictions = (probs >= 0.5).astype(int)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    n_samples = len(targets)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == targets[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (np.sum(in_bin) / n_samples)

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for probability calibration (Guo et al., ICML 2017).
    Optimizes a single scalar parameter T > 0 on validation NLL without modifying model weights.
    """

    def __init__(self) -> None:
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature: logits / T."""
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """
        Optimizes temperature T on validation logits using L-BFGS to minimize negative log-likelihood.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        val_logits = val_logits.detach()
        val_labels = val_labels.detach()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        optimal_t = float(self.temperature.item())
        print(f"[Temperature Scaling] Optimal Temperature T: {optimal_t:.4f}")
        return optimal_t


def plot_reliability_diagram(
    probs_uncal: np.ndarray,
    probs_cal: Optional[np.ndarray],
    targets: np.ndarray,
    save_path: str = "results/reliability_diagram.png",
    n_bins: int = 10,
) -> None:
    """
    Plots reliability diagram and ECE comparison before and after temperature scaling.
    """
    ece_uncal, accs_uncal, confs_uncal, _ = compute_ece(probs_uncal, targets, n_bins=n_bins)

    fig, ax = plt.subplots(1, 2 if probs_cal is not None else 1, figsize=(11, 5))
    if probs_cal is None:
        ax = [ax]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Uncalibrated
    ax[0].bar(bin_centers, accs_uncal, width=1.0 / n_bins, alpha=0.6, color="royalblue", edgecolor="black", label="Outputs")
    ax[0].plot([0, 1], [0, 1], "--", color="red", label="Perfect Calibration")
    ax[0].set_title(f"Uncalibrated (ECE = {ece_uncal * 100:.2f}%)", fontsize=12)
    ax[0].set_xlabel("Confidence", fontsize=11)
    ax[0].set_ylabel("Accuracy", fontsize=11)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.5)

    # Calibrated
    if probs_cal is not None:
        ece_cal, accs_cal, confs_cal, _ = compute_ece(probs_cal, targets, n_bins=n_bins)
        ax[1].bar(bin_centers, accs_cal, width=1.0 / n_bins, alpha=0.6, color="forestgreen", edgecolor="black", label="Outputs")
        ax[1].plot([0, 1], [0, 1], "--", color="red", label="Perfect Calibration")
        ax[1].set_title(f"Temperature Scaled (ECE = {ece_cal * 100:.2f}%)", fontsize=12)
        ax[1].set_xlabel("Confidence", fontsize=11)
        ax[1].set_ylabel("Accuracy", fontsize=11)
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].legend()
        ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Calibration] Reliability diagram saved to {save_path}")


def plot_sensitivity_specificity_tradeoff(
    targets: np.ndarray,
    probs: np.ndarray,
    chosen_threshold: float,
    save_path: str = "results/sens_spec_tradeoff.png",
) -> None:
    """Plots sensitivity and specificity curves across decision thresholds [0, 1]."""
    targets = np.asarray(targets).squeeze()
    if probs.ndim == 2:
        probs = probs[:, 1]

    thresholds = np.linspace(0.01, 0.99, 100)
    sensitivities = []
    specificities = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (targets == 1))
        fn = np.sum((preds == 0) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, sensitivities, label="Sensitivity (Recall)", color="firebrick", linewidth=2)
    plt.plot(thresholds, specificities, label="Specificity (TNR)", color="teal", linewidth=2)
    plt.axvline(chosen_threshold, color="black", linestyle="--", label=f"Calibrated Threshold ({chosen_threshold:.3f})")
    plt.axhline(0.95, color="gray", linestyle=":", alpha=0.8, label="95% Sensitivity Target")

    plt.xlabel("Classification Decision Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Clinical Triage: Sensitivity vs Specificity Trade-Off", fontsize=13)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Calibration] Sensitivity-specificity curve saved to {save_path}")
