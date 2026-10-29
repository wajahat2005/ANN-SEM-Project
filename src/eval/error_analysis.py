"""
Owner: Aiman Aslam (Experiments Lead)
Module: error_analysis
Reference: CS-320 Deep Learning Project Evaluation

Purpose:
    Identifies failure modes and provides visual inspection grids for misclassified samples.
    Computes per-class error breakdowns and generates informative diagnostic figures
    for both CIFAR-10 and PneumoniaMNIST.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from src.eval.metrics import plot_confusion_matrix


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Computes individual accuracy for each category to detect biased errors or class vulnerabilities.
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()

    per_class = {}
    for idx, cname in enumerate(class_names):
        mask = (y_true == idx)
        if np.sum(mask) > 0:
            acc = float(np.mean(y_pred[mask] == y_true[mask]))
            per_class[cname] = acc
        else:
            per_class[cname] = 0.0

    return per_class


def analyze_misclassifications(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    max_samples: int = 16,
) -> Dict[str, Any]:
    """
    Gathers all misclassified samples from a dataloader for visual inspection and analysis.
    """
    model.eval()
    misclassified_imgs = []
    misclassified_preds = []
    misclassified_targets = []
    misclassified_confs = []

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            incorrect = (preds != targets)
            if incorrect.any() and len(misclassified_imgs) < max_samples:
                for img, pred, target, conf in zip(
                    inputs[incorrect], preds[incorrect], targets[incorrect], confs[incorrect]
                ):
                    if len(misclassified_imgs) < max_samples:
                        misclassified_imgs.append(img.cpu())
                        misclassified_preds.append(int(pred.item()))
                        misclassified_targets.append(int(target.item()))
                        misclassified_confs.append(float(conf.item()))

    per_class_acc = compute_per_class_accuracy(
        np.array(all_targets), np.array(all_preds), class_names
    )

    return {
        "images": misclassified_imgs,
        "preds": misclassified_preds,
        "targets": misclassified_targets,
        "confidences": misclassified_confs,
        "all_targets": np.array(all_targets),
        "all_preds": np.array(all_preds),
        "per_class_acc": per_class_acc,
    }


def plot_misclassified_grid(
    misclassified_data: Dict[str, Any],
    class_names: List[str],
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    save_path: str = "results/misclassified_grid.png",
    num_samples: int = 16,
) -> None:
    """
    Renders and saves a grid of misclassified images with True vs Predicted tags and confidence scores.
    """
    imgs = misclassified_data["images"][:num_samples]
    preds = misclassified_data["preds"][:num_samples]
    targets = misclassified_data["targets"][:num_samples]
    confs = misclassified_data["confidences"][:num_samples]

    if len(imgs) == 0:
        print("[Error Analysis] Zero misclassifications found (100% accuracy on this batch).")
        return

    n_cols = 4
    n_rows = int(np.ceil(len(imgs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < len(imgs):
            img = imgs[i]
            # Un-normalize if stats provided
            if mean is not None and std is not None:
                m = torch.tensor(mean).view(-1, 1, 1)
                s = torch.tensor(std).view(-1, 1, 1)
                img = img * s + m
                img = torch.clamp(img, 0.0, 1.0)

            if img.shape[0] == 1:
                # Grayscale
                axes[i].imshow(img[0].numpy(), cmap="gray")
            else:
                # RGB
                axes[i].imshow(img.permute(1, 2, 0).numpy())

            true_label = class_names[targets[i]]
            pred_label = class_names[preds[i]]
            conf = confs[i] * 100.0

            axes[i].set_title(
                f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)",
                fontsize=9,
                color="red",
            )
        axes[i].axis("off")

    plt.suptitle("Misclassified Test Samples Inspection", fontsize=14, y=1.02)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Error Analysis] Misclassified grid saved to {save_path}")
