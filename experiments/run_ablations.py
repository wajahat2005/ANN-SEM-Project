"""
Orchestrator: Aiman Aslam (Experiments Lead)
Contributors: Mohammad Abdul Rehman (Lead & Paper Lead), Muhammad Wajahat (Data & Pipeline Lead)
Module: run_ablations
Reference: Huang et al. (CVPR 2017) Section 4 & CS-320 Systematic Ablation Protocol

Scientific Methodology:
    Strict One-Factor-At-A-Time (OFAT) exploration starting from ONE fixed, verified
    baseline config (configs/baseline.yaml):
    - Fixed Baseline: DenseNet-BC (L=40, k=12, bottleneck=True, theta=0.5, drop_rate=0.0, seed=42).
    - Reduced epoch budget (e.g. 30 epochs for full, 2 for smoke) to permit exhaustive comparison
      within student Colab limits (<3 hours total).

Ablation Matrix:
    1. Factor 1: Growth Rate k in {8, 12, 24} (Lead: Mohammad Abdul Rehman)
    2. Factor 2: Bottleneck On vs Off (DenseNet-BC vs DenseNet-C) (Lead: Mohammad Abdul Rehman)
    3. Factor 3: Transition Compression theta in {0.5, 1.0} (Lead: Muhammad Wajahat)
    4. Factor 4: Input Resolution (28, 64, 128 on PneumoniaMNIST) (Lead: Muhammad Wajahat)
    5. Factor 5: Dropout Rate p in {0.0, 0.2} (Lead: Aiman Aslam)
    6. Factor 6: Clinical Threshold (0.50 vs calibrated >=95% sensitivity) (Lead: Aiman Aslam)

Output:
    - Master CSV table: results/ablations.csv
      Columns: factor, value, dataset, params, val_metric, test_metric, epochs, seed, train_time
    - Comparative plot per factor saved to results/ablation_{factor}.png
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Tuple, Optional, List
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.models.densenet import DenseNet
from src.data.cifar import get_cifar10_dataloaders
from src.data.pneumonia import get_pneumonia_dataloaders
from src.train.engine import train_model, set_seed, get_device, evaluate
from src.eval.calibration import calibrate_threshold_for_sensitivity


def parse_args():
    parser = argparse.ArgumentParser(description="DenseNet Systematic Ablation Suite")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to baseline yaml")
    parser.add_argument(
        "--factor",
        type=str,
        default="all",
        choices=["growth_rate", "bottleneck", "compression", "resolution", "dropout", "threshold", "all"],
        help="Ablation factor to evaluate",
    )
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode (tiny subset, 1-2 epochs)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSV and plots")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for model weights")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cifar_model(growth_rate=12, bottleneck=True, theta=0.5, drop_rate=0.0) -> DenseNet:
    # DenseNet-BC L=40: (40 - 4)/6 = 6 layers per block if bottleneck=True
    # If bottleneck=False: (40 - 4)/3 = 12 layers per block
    num_layers = 6 if bottleneck else 12
    init_features = 2 * growth_rate if bottleneck and theta < 1.0 else 16
    return DenseNet(
        growth_rate=growth_rate,
        block_config=(num_layers, num_layers, num_layers),
        num_init_features=init_features,
        bottleneck=bottleneck,
        theta=theta,
        drop_rate=drop_rate,
        in_channels=3,
        num_classes=10,
        small_input=True,
    )


def run_cifar_ablation_trial(
    factor_name: str,
    val_name: Any,
    growth_rate: int = 12,
    bottleneck: bool = True,
    theta: float = 0.5,
    drop_rate: float = 0.0,
    epochs: int = 30,
    seed: int = 42,
    smoke: bool = False,
    device: torch.device = torch.device("cpu"),
    results_dir: str = "results",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Any]:
    set_seed(seed)
    train_loader, val_loader, test_loader, _ = get_cifar10_dataloaders(smoke=smoke, seed=seed)

    model = build_cifar_model(
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        theta=theta,
        drop_rate=drop_rate,
    ).to(device)

    params = model.count_parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)

    exp_id = f"ablation_{factor_name}_{val_name}"
    start_t = time.time()

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        experiment_name=exp_id,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        primary_metric="accuracy",
    )
    elapsed = time.time() - start_t

    val_res = evaluate(trained_model, val_loader, criterion, device)
    test_res = evaluate(trained_model, test_loader, criterion, device)

    return {
        "factor": factor_name,
        "value": str(val_name),
        "dataset": "CIFAR-10",
        "params": params,
        "val_metric": round(val_res["accuracy"] * 100.0, 2),
        "test_metric": round(test_res["accuracy"] * 100.0, 2),
        "epochs": epochs,
        "seed": seed,
        "train_time": round(elapsed, 1),
    }


def run_resolution_ablation_trial(
    size: int,
    epochs: int,
    seed: int,
    smoke: bool,
    device: torch.device,
    results_dir: str,
    checkpoint_dir: str,
) -> Dict[str, Any]:
    set_seed(seed)
    train_loader, val_loader, test_loader, meta = get_pneumonia_dataloaders(
        size=size,
        smoke=smoke,
    )

    model = DenseNet(
        growth_rate=12,
        block_config=(6, 6, 6),
        num_init_features=24,
        bottleneck=True,
        theta=0.5,
        in_channels=1,
        num_classes=2,
        small_input=True,
    ).to(device)

    params = model.count_parameters()
    class_weights = meta["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_id = f"ablation_resolution_{size}x{size}"
    start_t = time.time()

    trained_model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        experiment_name=exp_id,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        is_binary=True,
        primary_metric="auc",
    )
    elapsed = time.time() - start_t

    val_res = evaluate(trained_model, val_loader, criterion, device, is_binary=True)
    test_res = evaluate(trained_model, test_loader, criterion, device, is_binary=True)

    return {
        "factor": "resolution",
        "value": f"{size}x{size}",
        "dataset": "PneumoniaMNIST",
        "params": params,
        "val_metric": round(val_res["auc"], 4),
        "test_metric": round(test_res["auc"], 4),
        "epochs": epochs,
        "seed": seed,
        "train_time": round(elapsed, 1),
    }


def plot_ablation_bar(df_factor: pd.DataFrame, factor_name: str, results_dir: str) -> None:
    """Generates a clean bar plot for the ablation factor."""
    plt.figure(figsize=(7, 5))
    values = df_factor["value"].astype(str)
    metrics = df_factor["test_metric"].astype(float)

    bars = plt.bar(values, metrics, color="royalblue", edgecolor="black", width=0.5)
    plt.xlabel(f"Ablation Value ({factor_name})", fontsize=11)
    plt.ylabel(f"Test Metric ({df_factor['dataset'].iloc[0]})", fontsize=11)
    plt.title(f"Ablation Study: Impact of {factor_name.replace('_', ' ').title()}", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h}", ha="center", va="bottom", fontsize=10)

    save_path = os.path.join(results_dir, f"ablation_{factor_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Ablations] Plot saved to {save_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "ablations.csv")

    epochs = 2 if args.smoke else config["ablation"].get("reduced_epochs", 30)
    seed = config.get("seed", 42)

    rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Growth Rate k in {8, 12, 24} (Mohammad Abdul Rehman)
    # -------------------------------------------------------------
    if args.factor in ("all", "growth_rate"):
        print("\n--- Running Ablation 1: Growth Rate k in {8, 12, 24} ---")
        for k in [8, 12, 24]:
            r = run_cifar_ablation_trial(
                "growth_rate", k, growth_rate=k, bottleneck=True, theta=0.5, drop_rate=0.0,
                epochs=epochs, seed=seed, smoke=args.smoke, device=device,
                results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir
            )
            rows.append(r)

    # -------------------------------------------------------------
    # 2. Bottleneck On vs Off (Mohammad Abdul Rehman)
    # -------------------------------------------------------------
    if args.factor in ("all", "bottleneck"):
        print("\n--- Running Ablation 2: Bottleneck (DenseNet-BC vs DenseNet-C) ---")
        for b in [True, False]:
            val_str = "bottleneck_on" if b else "bottleneck_off"
            r = run_cifar_ablation_trial(
                "bottleneck", val_str, growth_rate=12, bottleneck=b, theta=0.5, drop_rate=0.0,
                epochs=epochs, seed=seed, smoke=args.smoke, device=device,
                results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir
            )
            rows.append(r)

    # -------------------------------------------------------------
    # 3. Compression theta in {0.5, 1.0} (Muhammad Wajahat)
    # -------------------------------------------------------------
    if args.factor in ("all", "compression"):
        print("\n--- Running Ablation 3: Compression Factor theta in {0.5, 1.0} ---")
        for th in [0.5, 1.0]:
            r = run_cifar_ablation_trial(
                "compression", th, growth_rate=12, bottleneck=True, theta=th, drop_rate=0.0,
                epochs=epochs, seed=seed, smoke=args.smoke, device=device,
                results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir
            )
            rows.append(r)

    # -------------------------------------------------------------
    # 4. Input Resolution on PneumoniaMNIST: 28, 64, 128 (Muhammad Wajahat)
    # -------------------------------------------------------------
    if args.factor in ("all", "resolution"):
        print("\n--- Running Ablation 4: Resolution (28, 64, 128) on PneumoniaMNIST ---")
        for s in [28, 64, 128]:
            r = run_resolution_ablation_trial(
                s, epochs=epochs, seed=seed, smoke=args.smoke, device=device,
                results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir
            )
            rows.append(r)

    # -------------------------------------------------------------
    # 5. Dropout p in {0.0, 0.2} (Aiman Aslam)
    # -------------------------------------------------------------
    if args.factor in ("all", "dropout"):
        print("\n--- Running Ablation 5: Dropout Rate p in {0.0, 0.2} ---")
        for p in [0.0, 0.2]:
            r = run_cifar_ablation_trial(
                "dropout", p, growth_rate=12, bottleneck=True, theta=0.5, drop_rate=p,
                epochs=epochs, seed=seed, smoke=args.smoke, device=device,
                results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir
            )
            rows.append(r)

    # Compile master DataFrame
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(
                subset=["factor", "value", "dataset"], keep="last"
            )
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"\n[Ablations] Master ablation table written to {csv_path}")

    # Generate plot per factor
    for factor in df_combined["factor"].unique():
        df_f = df_combined[df_combined["factor"] == factor]
        plot_ablation_bar(df_f, factor, args.results_dir)

    print("\n" + "=" * 90)
    print(f"{'Factor':<15} | {'Value':<15} | {'Dataset':<15} | {'Params':<10} | {'Val Score':<10} | {'Test Score':<10}")
    print("-" * 90)
    for _, row in df_combined.iterrows():
        print(f"{row['factor']:<15} | {str(row['value']):<15} | {row['dataset']:<15} | {int(row['params']):<10,d} | {row['val_metric']:<10} | {row['test_metric']:<10}")
    print("=" * 90)


if __name__ == "__main__":
    main()
