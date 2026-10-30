"""
Owner: Mohammad Abdul Rehman (Team & Paper Lead)
Module: reproduce_cifar
Reference: Huang et al. (CVPR 2017) Section 4: "Results on CIFAR"

Reproduces the core comparative benchmark of the DenseNet paper:
1. DenseNet-40 (k=12): ~1.0M parameters, paper error: 5.24% on CIFAR-10+
2. DenseNet-BC-100 (k=12): ~0.8M parameters, paper error: 4.51% on CIFAR-10+
3. ResNet-56: ~0.85M parameters, paper error: ~6.97% on CIFAR-10+

Training Hyperparameters:
- Optimizer: SGD with Nesterov momentum 0.9, weight decay 1e-4.
- Batch size: 64.
- Total epochs: 300 (standard full reproduction schedule; configurable via --epochs).
- Learning rate: Initial 0.1, divided by 10 at 50% (epoch 150) and 75% (epoch 225).
- Weight initialization: He normal.
- Checkpointing: Saves last and best-val models to checkpoints/ (Google Drive resumable).
- Logging: Per-epoch CSV log and loss/accuracy plots saved to results/.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.densenet import densenet40_cifar, densenet_bc_100_cifar
from src.models.resnet_cifar import resnet56_cifar
from src.data.cifar import get_cifar10_dataloaders
from src.train.engine import train_model, set_seed, get_device, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="DenseNet CIFAR-10 Reproduction")
    parser.add_argument(
        "--model",
        type=str,
        default="densenet40",
        choices=["densenet40", "densenet_bc_100", "resnet56", "all"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs (default 300)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64)")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--smoke", action="store_true", help="Run 1-2 epochs on tiny subset for smoke verification")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save logs and plots")
    return parser.parse_args()


def run_experiment(model_name: str, args, train_loader, val_loader, test_loader, device):
    print(f"\n{'='*70}")
    print(f"STARTING CIFAR-10 REPRODUCTION: {model_name.upper()}")
    print(f"{'='*70}")

    epochs = 2 if args.smoke else args.epochs

    if model_name == "densenet40":
        model = densenet40_cifar(growth_rate=12, num_classes=10, in_channels=3)
    elif model_name == "densenet_bc_100":
        model = densenet_bc_100_cifar(growth_rate=12, num_classes=10, in_channels=3)
    elif model_name == "resnet56":
        model = resnet56_cifar(num_classes=10, in_channels=3)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Total Trainable Parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # MultiStepLR: divide lr by 10 at 50% and 75% of total epochs
    milestones = [int(0.50 * epochs), int(0.75 * epochs)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    exp_name = f"cifar10_{model_name}_{'smoke' if args.smoke else f'{epochs}ep'}"

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
        is_binary=False,
        primary_metric="accuracy",
    )

    # Final Evaluation on the 10,000 Test Set
    test_metrics = evaluate(trained_model, test_loader, criterion, device, is_binary=False)
    test_acc = test_metrics["accuracy"] * 100.0
    test_err = 100.0 - test_acc

    print(f"\n[{model_name}] Final Test Evaluation:")
    print(f"  - Test Accuracy: {test_acc:.2f}%")
    print(f"  - Test Top-1 Error: {test_err:.2f}%\n")

    return {
        "model": model_name,
        "params": param_count,
        "test_acc": test_acc,
        "test_err": test_err,
        "history": history,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader, meta = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
        smoke=args.smoke,
    )

    models_to_run = ["densenet40", "densenet_bc_100", "resnet56"] if args.model == "all" else [args.model]

    results = []
    for m in models_to_run:
        res = run_experiment(m, args, train_loader, val_loader, test_loader, device)
        results.append(res)

    print("\n" + "=" * 60)
    print("REPRODUCTION SUMMARY TABLE (CIFAR-10+)")
    print("=" * 60)
    print(f"{'Model':<20} | {'Parameters':<12} | {'Test Acc (%)':<14} | {'Test Error (%)':<14}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} | {r['params']:<12,d} | {r['test_acc']:<14.2f} | {r['test_err']:<14.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
