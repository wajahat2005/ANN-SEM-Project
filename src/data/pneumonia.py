"""
Owner: Muhammad Wajahat (Data & Pipeline Lead)
Module: pneumonia
Reference: MedMNIST v2/v3 (Yang et al., Scientific Data 2023)

Dataset Setup:
    - Dataset: PneumoniaMNIST (Pediatric Chest X-rays from Kermany et al., Cell 2018).
    - Task: Binary classification (0: Normal, 1: Pneumonia).
    - Resolution: size=64 (64x64 grayscale images; ablations at 28x28 and 128x128).
    - Official Splits:
        * Train: 4,708 samples (1,214 Normal, 3,494 Pneumonia) -> Severe class imbalance (~1 : 2.88).
        * Validation: 524 samples (135 Normal, 389 Pneumonia).
        * Test: 624 samples (234 Normal, 390 Pneumonia).
    - Normalization:
        * Computed STRICTLY on the 4,708 training split.
        * Expected approx: mean = 0.5719, std = 0.1684.
    - Class Imbalance Handling:
        * We compute positive class weight w_pos = N_neg / N_pos = 1214 / 3494 ≈ 0.347
          or inverse frequency weights for CrossEntropyLoss: w_class = [Total / (2 * N_c)].
          Class-weighted loss is chosen over oversampling because oversampling artificially distorts
          posterior calibration in clinical risk triage, while class weighting properly penalizes
          the under-represented class while preserving natural prevalence during probabilistic calibration.
"""

import os
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

try:
    import medmnist
    from medmnist import PneumoniaMNIST
except ImportError:
    PneumoniaMNIST = None


class SyntheticPneumoniaDataset(Dataset):
    """
    Fallback dataset used in offline smoke tests if medmnist data cannot be downloaded.
    Generates deterministic synthetic tensors matching PneumoniaMNIST shapes and class ratio.
    """

    def __init__(self, size: int = 64, num_samples: int = 128, split: str = "train", transform=None) -> None:
        self.size = size
        self.num_samples = num_samples
        self.split = split
        self.transform = transform
        # Replicate ~74% pneumonia prevalence
        rng = np.random.RandomState(42 if split == "train" else 100)
        self.data = rng.uniform(0, 255, size=(num_samples, size, size)).astype(np.uint8)
        self.labels = (rng.rand(num_samples, 1) < 0.74).astype(np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        img = self.data[index]
        target = self.labels[index]
        from PIL import Image
        img = Image.fromarray(img, mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, target.squeeze()


def compute_pneumonia_train_stats(size: int = 64, data_dir: str = "./data_cache") -> Tuple[float, float]:
    """
    Computes grayscale mean and standard deviation strictly on the 4,708 training images.
    Expected: mean ≈ 0.5719, std ≈ 0.1684.
    """
    if PneumoniaMNIST is None:
        print("[Pneumonia Data] medmnist package not available, using analytical default stats.")
        return 0.5719, 0.1684

    try:
        raw_train = PneumoniaMNIST(split="train", download=True, size=size, root=data_dir)
        # raw_train.imgs shape: (4708, size, size), uint8 in [0, 255]
        imgs = raw_train.imgs.astype(np.float32) / 255.0
        mean = float(np.mean(imgs))
        std = float(np.std(imgs))
        print(f"[Pneumonia Data] Train Stats (size={size}) -> Mean: {mean:.4f}, Std: {std:.4f}")
        return round(mean, 4), round(std, 4)
    except Exception as e:
        print(f"[Pneumonia Data] Download failed ({e}), falling back to standard expected stats.")
        return 0.5719, 0.1684


def get_pneumonia_class_weights(train_dataset) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for cross-entropy loss:
        weight[c] = total_samples / (num_classes * count[c])
    """
    if hasattr(train_dataset, "labels"):
        labels = train_dataset.labels.squeeze()
    else:
        # Extract from subset or loop
        labels = [target for _, target in train_dataset]
        labels = np.array(labels)

    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = [total / (len(classes) * c) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"[Pneumonia Data] Class counts: {dict(zip(classes, counts))}, Computed Loss Weights: {weights_tensor.tolist()}")
    return weights_tensor


def get_pneumonia_dataloaders(
    data_dir: str = "./data_cache",
    batch_size: int = 64,
    size: int = 64,
    smoke: bool = False,
    smoke_samples: int = 128,
    num_workers: int = 0,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    use_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Constructs train, validation, and test DataLoaders for PneumoniaMNIST.

    Args:
        data_dir: Cache root for dataset.
        batch_size: Mini-batch size.
        size: Image spatial resolution (28, 64, or 128). Default: 64.
        smoke: If True, uses tiny subset for fast execution on CPU.
        smoke_samples: Number of samples in smoke mode.
        num_workers: DataLoader worker count.
        mean: Training set mean (default: 0.5719).
        std: Training set std (default: 0.1684).
        use_sampler: If True, uses WeightedRandomSampler instead of class-weighted loss.

    Returns:
        (train_loader, val_loader, test_loader, metadata_dict)
    """
    if mean is None or std is None:
        mean, std = 0.5719, 0.1684

    # Data Augmentation for training: slight random horizontal flip and rotation for chest X-rays
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    # Attempt to load official MedMNIST dataset
    try:
        if PneumoniaMNIST is None:
            raise RuntimeError("MedMNIST package not installed.")

        raw_train = PneumoniaMNIST(split="train", download=True, size=size, root=data_dir, transform=train_transform)
        raw_val = PneumoniaMNIST(split="val", download=True, size=size, root=data_dir, transform=eval_transform)
        raw_test = PneumoniaMNIST(split="test", download=True, size=size, root=data_dir, transform=eval_transform)

        if smoke:
            # Slice real dataset for smoke testing
            train_set = Subset(raw_train, range(min(len(raw_train), smoke_samples)))
            val_set = Subset(raw_val, range(min(len(raw_val), min(64, smoke_samples))))
            test_set = Subset(raw_test, range(min(len(raw_test), min(64, smoke_samples))))
        else:
            train_set = raw_train
            val_set = raw_val
            test_set = raw_test

    except Exception as e:
        print(f"[Pneumonia Data] Note: Initializing synthetic data ({e}).")
        train_set = SyntheticPneumoniaDataset(size=size, num_samples=smoke_samples if smoke else 500, split="train", transform=train_transform)
        val_set = SyntheticPneumoniaDataset(size=size, num_samples=min(64, smoke_samples), split="val", transform=eval_transform)
        test_set = SyntheticPneumoniaDataset(size=size, num_samples=min(64, smoke_samples), split="test", transform=eval_transform)

    # Class balance and weights
    class_weights = get_pneumonia_class_weights(train_set)

    sampler = None
    shuffle = True
    if use_sampler:
        # Weighted random sampler alternative
        labels = train_set.labels.squeeze() if hasattr(train_set, "labels") else [y for _, y in train_set]
        sample_weights = [class_weights[int(label)].item() for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_batch, sample_targets = next(iter(train_loader))
    print(f"[Pneumonia Data] Batch Shape: {sample_batch.shape}, Dtype: {sample_batch.dtype}")
    print(f"[Pneumonia Data] Targets Shape: {sample_targets.shape}, Dtype: {sample_targets.dtype}")
    print(f"[Pneumonia Data] Splits -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    metadata = {
        "dataset": f"PneumoniaMNIST-{size}",
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "num_classes": 2,
        "classes": ["Normal", "Pneumonia"],
        "in_channels": 1,
        "mean": mean,
        "std": std,
        "class_weights": class_weights,
        "batch_shape": tuple(sample_batch.shape),
    }

    return train_loader, val_loader, test_loader, metadata


def plot_pneumonia_sample_grid(
    dataloader: DataLoader,
    save_path: str = "results/pneumonia_sample_grid.png",
    num_images: int = 16,
) -> None:
    """Plots and saves a grid of pediatric chest X-rays with class labels."""
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Un-normalize
    mean = 0.5719
    std = 0.1684
    unnorm = images * std + mean
    unnorm = torch.clamp(unnorm, 0.0, 1.0)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    classes = ["Normal", "Pneumonia"]

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img_np = unnorm[i, 0].cpu().numpy()
            ax.imshow(img_np, cmap="gray")
            label_idx = int(labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i])
            ax.set_title(classes[label_idx], fontsize=10)
        ax.axis("off")

    plt.suptitle("PneumoniaMNIST Sample Grid (64x64 Grayscale)", fontsize=13)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Pneumonia Data] Sample grid saved to {save_path}")


if __name__ == "__main__":
    t_l, v_l, te_l, meta = get_pneumonia_dataloaders(smoke=True)
    plot_pneumonia_sample_grid(t_l)
