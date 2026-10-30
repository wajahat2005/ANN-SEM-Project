"""
Owner: Aiman Aslam (Experiments Lead)
Module: engine
Reference: CS-320 Deep Learning Project Engineering Standards & Huang et al. (CVPR 2017)

Key Features:
    1. Training & Evaluation Loops:
       - Proper model.train() and model.eval() state management.
       - Automatic Mixed Precision (AMP) via torch.autocast and torch.cuda.amp.GradScaler
         (automatically falls back cleanly on CPU without errors).
       - Explicit optimizer.zero_grad(set_to_none=True) for optimal memory throughput.
    2. Checkpointing & Resumability:
       - Saves 'best_val.pt' and 'last.pt' at every epoch.
       - Supports configurable Google Drive destination path to safeguard long Colab sessions.
       - Full state serialization: model weights, optimizer, lr_scheduler, scaler, epoch, and metric history.
    3. Reproducibility & Seeding:
       - Sets torch, numpy, random, and cuda seeds with deterministic flags.
    4. Diagnostics & Logging:
       - Per-epoch tabular metrics logged directly to CSV.
       - Automated curve generation (loss & accuracy plots) for every run.
"""

import os
import time
import random
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.eval.metrics import compute_classification_metrics


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Seed] Deterministic seed set to {seed}")


def get_device() -> torch.device:
    """Selects CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)} with AMP enabled")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available. Running on CPU (AMP disabled)")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Executes one complete training epoch with AMP support.

    Returns:
        (avg_train_loss, train_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == "cuda" and scaler is not None)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # Ensure targets are 1D LongTensor for CrossEntropyLoss
        if targets.dim() > 1:
            targets = targets.squeeze(-1).long()

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total) * 100:.2f}%"
        })

    avg_loss = running_loss / total
    train_acc = correct / total
    return avg_loss, train_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool = False,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates the model over a validation or test DataLoader.

    Returns:
        dict containing loss, accuracy, and task-specific metrics.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.dim() > 1:
                targets = targets.squeeze(-1).long()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            probs = torch.softmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = running_loss / total
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)

    metrics = compute_classification_metrics(
        y_true=targets_np,
        y_pred_probs=probs_np,
        threshold=threshold,
        is_binary=is_binary,
    )
    metrics["loss"] = avg_loss
    return metrics


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    experiment_name: str = "densenet",
) -> str:
    """Saves checkpoint to disk and maintains 'best_val.pt' and 'last.pt'."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_path = os.path.join(checkpoint_dir, f"{experiment_name}_last.pt")
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{experiment_name}_best_val.pt")
        torch.save(state, best_path)
        return best_path
    return last_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """
    Resumes training from a saved checkpoint file.

    Returns:
        start_epoch (int): Next epoch to execute.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"[Checkpoint] Loading state from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", 0.0)
    print(f"[Checkpoint] Successfully resumed from epoch {start_epoch - 1} (Best metric: {best_metric:.4f})")
    return start_epoch


def plot_training_curves(
    history: pd.DataFrame,
    save_path: str = "results/training_curves.png",
    metric_name: str = "accuracy",
) -> None:
    """Plots and saves train/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(history["epoch"], history["val_loss"], label="Val Loss", color="crimson", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=13)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Metric curve
    train_m = f"train_{metric_name}" if f"train_{metric_name}" in history.columns else "train_acc"
    val_m = f"val_{metric_name}" if f"val_{metric_name}" in history.columns else "val_acc"

    ax2.plot(history["epoch"], history[train_m], label=f"Train {metric_name.title()}", color="royalblue", linewidth=2)
    ax2.plot(history["epoch"], history[val_m], label=f"Val {metric_name.title()}", color="crimson", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel(metric_name.title(), fontsize=11)
    ax2.set_title(f"Training & Validation {metric_name.title()}", fontsize=13)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Curves] Training plots written to {save_path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epochs: int,
    device: torch.device,
    experiment_name: str = "densenet",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    resume_path: Optional[str] = None,
    is_binary: bool = False,
    primary_metric: str = "accuracy",
) -> Tuple[nn.Module, pd.DataFrame]:
    """
    Complete end-to-end training pipeline.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    start_epoch = 1
    best_metric_val = 0.0

    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)

    history_records: List[Dict[str, Any]] = []
    csv_log_path = os.path.join(results_dir, f"{experiment_name}_history.csv")

    total_start_time = time.time()

    print(f"\n[Training] Starting experiment '{experiment_name}' for {epochs} epochs on {device.type.upper()}...")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            is_binary=is_binary,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_metric = val_metrics.get(primary_metric, val_metrics.get("accuracy", 0.0))
        is_best = current_metric > best_metric_val
        if is_best:
            best_metric_val = current_metric

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": epoch_time,
        }
        if is_binary and "auc" in val_metrics:
            record["val_auc"] = val_metrics["auc"]
            record["val_sensitivity"] = val_metrics["sensitivity"]
            record["val_specificity"] = val_metrics["specificity"]

        history_records.append(record)

        # Print summary
        metric_str = f"Val {primary_metric}: {current_metric:.4f} {'(*BEST*)' if is_best else ''}"
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | {metric_str} | Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric_val,
            "history": history_records,
        }
        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name,
        )

    # Export history to CSV and plot curves
    df_history = pd.DataFrame(history_records)
    df_history.to_csv(csv_log_path, index=False)
    print(f"[Training] Epoch log written to {csv_log_path}")

    plot_training_curves(
        history=df_history,
        save_path=os.path.join(results_dir, f"{experiment_name}_curves.png"),
        metric_name=primary_metric,
    )

    total_time = time.time() - total_start_time
    print(f"[Training] Completed in {total_time / 60:.2f} minutes. Best {primary_metric}: {best_metric_val:.4f}")
    return model, df_history
