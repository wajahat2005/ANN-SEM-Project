"""
Owner: Aiman Aslam (Experiments Lead)
Module: run_pneumonia
Reference: MedMNIST v2/v3 Benchmarking & Clinical Transfer Evaluation

Compares three architectures on PneumoniaMNIST (size=64):
1. Simple CNN (5-layer baseline adapted from MedMNIST notebook).
2. Custom DenseNet (e.g. DenseNet-BC-40 adapted for 1-channel 64x64 input).
3. Fine-tuned DenseNet-121 (torchvision ImageNet pretrained).

Evaluation Protocol:
- Primary metric: ROC-AUC.
- Secondary metrics: Sensitivity (TPR), Specificity (TNR), Accuracy, F1.
- Class Imbalance: Handled via class-weighted CrossEntropyLoss.
- Calibration: Decision threshold chosen on the VALIDATION split to achieve >= 95% sensitivity,
  then evaluated on the unseen TEST split.
- Diagnostics: Reliability diagram, ECE, and confusion matrix generated.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.simple_cnn import SimpleCNN
from src.models.densenet import DenseNet
from src.models.pretrained import get_pretrained_densenet121
from src.data.pneumonia import get_pneumonia_dataloaders
from src.train.engine import train_model, set_seed, get_device, evaluate
from src.eval.metrics import compute_classification_metrics, plot_confusion_matrix
from src.eval.calibration import (
    calibrate_threshold_for_sensitivity,
    compute_ece,
    plot_reliability_diagram,
    plot_sensitivity_specificity_tradeoff,
    TemperatureScaler,
)
from src.eval.error_analysis import (
    analyze_misclassifications,
    plot_misclassified_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PneumoniaMNIST Medical Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="custom_densenet",
        choices=["simple_cnn", "custom_densenet", "pretrained_densenet121", "all"],
        help="Model architecture to evaluate",
    )
    parser.add_argument("--size", type=int, default=64, choices=[28, 64, 128], help="Input resolution (default: 64)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode (tiny subset, 2 epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
    return parser.parse_args()


def get_model(model_name: str, in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "custom_densenet":
        # Compact DenseNet-BC for 64x64 medical imaging
        return DenseNet(
            growth_rate=12,
            block_config=(6, 6, 6),
            num_init_features=24,
            bottleneck=True,
            theta=0.5,
            in_channels=in_channels,
            num_classes=num_classes,
            small_input=True,
        )
    elif model_name == "pretrained_densenet121":
        return get_pretrained_densenet121(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=True,
            resize_to_224=False,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_single_model(model_name: str, args, train_loader, val_loader, test_loader, metadata, device):
    print(f"\n{'='*70}")
    print(f"RUNNING PNEUMONIAMNIST EXPERIMENT: {model_name.upper()} (size={args.size})")
    print(f"{'='*70}")

    epochs = 2 if args.smoke else args.epochs

    model = get_model(model_name, in_channels=1, num_classes=2).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Parameters: {params:,}")

    # Use class-weighted CrossEntropyLoss to address the ~1:2.88 imbalance
    class_weights = metadata["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_name = f"pneumonia_{model_name}_{args.size}_{'smoke' if args.smoke else f'{epochs}ep'}"

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        experiment_name=exp_name,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        resume_path=args.resume,
        is_binary=True,
        primary_metric="auc",
    )

    # -------------------------------------------------------------
    # Clinical Calibration: Pick Threshold on Validation, Test on Test
    # -------------------------------------------------------------
    print("\n--- Clinical Sensitivity Calibration ---")
    # Gather validation probabilities
    trained_model.eval()
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            p = torch.softmax(trained_model(x), dim=1)[:, 1]
            val_probs.extend(p.cpu().numpy().tolist())
            val_targets.extend(y.squeeze().numpy().tolist())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Calibrate threshold for >= 95% sensitivity
    chosen_thresh, val_sens, val_spec = calibrate_threshold_for_sensitivity(
        val_targets=val_targets,
        val_probs=val_probs,
        target_sensitivity=0.95,
    )

    # Evaluate on Unseen Test Split using calibrated threshold
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = torch.softmax(trained_model(x), dim=1)[:, 1]
            test_probs.extend(p.cpu().numpy().tolist())
            test_targets.extend(y.squeeze().numpy().tolist())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    test_metrics_cal = compute_classification_metrics(
        y_true=test_targets,
        y_pred_probs=test_probs,
        threshold=chosen_thresh,
        is_binary=True,
    )

    test_metrics_default = compute_classification_metrics(
        y_true=test_targets,
        y_pred_probs=test_probs,
        threshold=0.5,
        is_binary=True,
    )

    print(f"\n[{model_name}] Final Test Metrics:")
    print(f"  - Primary Metric (ROC-AUC): {test_metrics_cal['auc']:.4f}")
    print(f"  - At Default Threshold (0.50): Sensitivity={test_metrics_default['sensitivity']*100:.1f}%, Specificity={test_metrics_default['specificity']*100:.1f}%")
    print(f"  - At Calibrated Threshold ({chosen_thresh:.2f}): Sensitivity={test_metrics_cal['sensitivity']*100:.1f}%, Specificity={test_metrics_cal['specificity']*100:.1f}%")

    # Generate Calibration Figures
    cal_fig_path = os.path.join(args.results_dir, f"{exp_name}_reliability.png")
    plot_reliability_diagram(test_probs, None, test_targets, save_path=cal_fig_path)

    tradeoff_path = os.path.join(args.results_dir, f"{exp_name}_tradeoff.png")
    plot_sensitivity_specificity_tradeoff(test_targets, test_probs, chosen_thresh, save_path=tradeoff_path)

    # Confusion matrix
    cm_path = os.path.join(args.results_dir, f"{exp_name}_cm.png")
    preds = (test_probs >= chosen_thresh).astype(int)
    plot_confusion_matrix(test_targets, preds, class_names=["Normal", "Pneumonia"], save_path=cm_path)

    return {
        "model": model_name,
        "params": params,
        "auc": test_metrics_cal["auc"],
        "calibrated_threshold": chosen_thresh,
        "sensitivity": test_metrics_cal["sensitivity"],
        "specificity": test_metrics_cal["specificity"],
        "accuracy": test_metrics_cal["accuracy"],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader, meta = get_pneumonia_dataloaders(
        batch_size=args.batch_size,
        size=args.size,
        smoke=args.smoke,
    )

    models_to_run = ["simple_cnn", "custom_densenet", "pretrained_densenet121"] if args.model == "all" else [args.model]

    results = []
    for m in models_to_run:
        res = run_single_model(m, args, train_loader, val_loader, test_loader, meta, device)
        results.append(res)

    print("\n" + "=" * 80)
    print("PNEUMONIAMNIST BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<24} | {'Params':<10} | {'AUC':<8} | {'Sens (>=95%)':<14} | {'Spec':<8} | {'Thresh':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<24} | {r['params']:<10,d} | {r['auc']:<8.4f} | {r['sensitivity']*100:<14.1f}% | {r['specificity']*100:<8.1f}% | {r['calibrated_threshold']:<8.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
