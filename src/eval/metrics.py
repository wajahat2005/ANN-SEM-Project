"""
Owner: Muhammad Wajahat (Data & Pipeline Lead)
Module: metrics
Reference: CS-320 Evaluation Protocol & Medical Image Benchmarking

Computes classification metrics for:
1. Multi-class classification (CIFAR-10):
   - Top-1 Accuracy, Macro-averaged F1, Per-class accuracy, Confusion Matrix.
2. Binary clinical classification (PneumoniaMNIST):
   - Accuracy, ROC-AUC, Sensitivity (TPR / Recall), Specificity (TNR), F1-score, Confusion Matrix.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    is_binary: bool = True,
) -> Dict[str, float]:
    """
    Computes all essential evaluation metrics from true labels and predicted probabilities.

    Args:
        y_true: 1D array of ground truth integer labels.
        y_pred_probs: 2D array of predicted class probabilities (N, C) or 1D array for positive class.
        threshold: Decision threshold for positive class in binary tasks (default 0.5).
        is_binary: If True, computes medical metrics (ROC-AUC, sensitivity, specificity).

    Returns:
        dict containing metric scores.
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred_probs = np.asarray(y_pred_probs)

    metrics = {}

    if is_binary:
        # If probabilities are 2D (N, 2), extract positive class probability
        if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 2:
            p_pos = y_pred_probs[:, 1]
        elif y_pred_probs.ndim == 1:
            p_pos = y_pred_probs
        else:
            raise ValueError(f"Unexpected shape for binary probabilities: {y_pred_probs.shape}")

        y_pred = (p_pos >= threshold).astype(int)

        acc = float(accuracy_score(y_true, y_pred))

        # ROC-AUC (threshold-independent ranking metric)
        try:
            auc = float(roc_auc_score(y_true, p_pos))
        except ValueError:
            auc = 0.5  # Fallback if only 1 class present in batch

        # Confusion Matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Sensitivity (Recall / True Positive Rate): TP / (TP + FN)
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Specificity (True Negative Rate): TN / (TN + FP)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Precision (Positive Predictive Value): TP / (TP + FP)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        # F1 Score
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        metrics = {
            "accuracy": acc,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "threshold": threshold,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    else:
        # Multi-class (CIFAR-10)
        if y_pred_probs.ndim == 2:
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            y_pred = y_pred_probs

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, y_pred_probs, multi_class="ovr"))
        except (ValueError, IndexError):
            auc = 0.0

        metrics = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "auc": auc,
        }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> None:
    """
    Plots a visually styled confusion matrix with optional percentage normalization.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    else:
        cm_norm = cm

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = format(cm_norm[i, j], fmt)
            if normalize:
                val_str = f"{val_str}\n({cm[i, j]})"
            plt.text(
                j, i, val_str,
                horizontalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Metrics] Confusion matrix saved to {save_path}")
    else:
        plt.show()
