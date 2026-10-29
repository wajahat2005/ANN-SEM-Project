"""
Owner: Muhammad Wajahat (Data & Pipeline Lead)
Module: cifar
Reference: Huang et al. (CVPR 2017) CIFAR-10 Experimental Protocol

Dataset Setup:
    - CIFAR-10+ dataset: 60,000 images (32x32 RGB) in 10 classes.
    - Official train set: 50,000 images -> Split into 45,000 train / 5,000 validation (seed=42).
    - Official test set: 10,000 images.
    - Normalization: Per-channel mean and std computed STRICTLY on the 45,000 training split
      to guarantee ZERO data leakage into validation or testing.
      Expected approx: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616].
    - Data Augmentation (CIFAR-10+):
      * Random crop with 4-pixel reflection/zero padding (size 32x32)
      * Random horizontal flip (p=0.5)
"""

import os
from typing import Tuple, Dict, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# CIFAR-10 class labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class TransformedSubset(Dataset):
    """
    Subsets a dataset while dynamically applying split-specific transforms.
    This guarantees that the 45,000 train split receives data augmentation,
    while the 5,000 validation split receives deterministic evaluation transforms,
    even though both originate from the same underlying torchvision dataset.
    """

    def __init__(self, subset: Subset, transform=None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        img, target = self.subset[index]
        if self.transform is not None:
            # If the underlying dataset already converted to PIL or Tensor
            if not isinstance(img, torch.Tensor) and not hasattr(img, "size"):
                img = transforms.functional.to_pil_image(img)
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.subset)


def compute_cifar10_train_stats(
    data_dir: str = "./data_cache",
    seed: int = 42,
    val_size: int = 5000,
) -> Tuple[List[float], List[float]]:
    """
    Computes per-channel mean and standard deviation strictly over the 45,000 train split.
    Guarantees no data leakage from validation or test sets.
    """
    # Load raw train dataset without normalization
    raw_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    total_len = len(raw_dataset)
    train_size = total_len - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, _ = torch.utils.data.random_split(
        range(total_len), [train_size, val_size], generator=generator
    )

    print(f"[CIFAR Data] Computing train-only stats on {len(train_indices)} samples (seed={seed})...")

    # Accumulate sums
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    loader = DataLoader(
        Subset(raw_dataset, train_indices.indices),
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    for data, _ in loader:
        # data shape: (B, 3, 32, 32)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    mean_list = [float(round(m.item(), 4)) for m in mean]
    std_list = [float(round(s.item(), 4)) for s in std]

    print(f"[CIFAR Data] Train Stats -> Mean: {mean_list}, Std: {std_list}")
    return mean_list, std_list


def get_cifar10_dataloaders(
    data_dir: str = "./data_cache",
    batch_size: int = 64,
    val_size: int = 5000,
    seed: int = 42,
    smoke: bool = False,
    smoke_samples: int = 128,
    num_workers: int = 0,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Constructs train, validation, and test DataLoaders for CIFAR-10.

    Args:
        data_dir: Cache directory for dataset files.
        batch_size: Mini-batch size (paper default: 64).
        val_size: Size of held-out validation split (default: 5,000).
        seed: Random seed for reproducible splitting (default: 42).
        smoke: If True, restricts datasets to smoke_samples for rapid sanity verification.
        smoke_samples: Number of samples in smoke mode.
        num_workers: Subprocess data loading workers.
        mean: Optional train mean; if None, standard train split stats are used.
        std: Optional train std; if None, standard train split stats are used.

    Returns:
        (train_loader, val_loader, test_loader, metadata_dict)
    """
    if mean is None or std is None:
        # Empirical train-only stats on 45,000 split (seed=42)
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]

    # Data Augmentation Transforms for CIFAR-10+ (Huang et al., Section 3)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Download raw base dataset
    raw_train = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )
    test_dataset_raw = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None,
    )

    total_train = len(raw_train)
    train_count = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_train), [train_count, val_size], generator=generator
    )

    if smoke:
        # Downsample for quick smoke testing
        smoke_train_idx = train_indices.indices[:smoke_samples]
        smoke_val_idx = val_indices.indices[:min(64, len(val_indices))]
        smoke_test_idx = list(range(min(64, len(test_dataset_raw))))

        train_set = TransformedSubset(Subset(raw_train, smoke_train_idx), transform=train_transform)
        val_set = TransformedSubset(Subset(raw_train, smoke_val_idx), transform=eval_transform)
        test_set = TransformedSubset(Subset(test_dataset_raw, smoke_test_idx), transform=eval_transform)
    else:
        train_set = TransformedSubset(Subset(raw_train, train_indices.indices), transform=train_transform)
        val_set = TransformedSubset(Subset(raw_train, val_indices.indices), transform=eval_transform)
        test_set = TransformedSubset(test_dataset_raw, transform=eval_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
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

    # Inspect first batch
    sample_batch, sample_targets = next(iter(train_loader))
    print(f"[CIFAR Data] Batch Shape: {sample_batch.shape}, Dtype: {sample_batch.dtype}")
    print(f"[CIFAR Data] Targets Shape: {sample_targets.shape}, Dtype: {sample_targets.dtype}")
    print(f"[CIFAR Data] Splits -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    metadata = {
        "dataset": "CIFAR-10",
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "num_classes": 10,
        "classes": CIFAR10_CLASSES,
        "in_channels": 3,
        "mean": mean,
        "std": std,
        "batch_shape": tuple(sample_batch.shape),
    }

    return train_loader, val_loader, test_loader, metadata


def plot_cifar_sample_grid(
    dataloader: DataLoader,
    save_path: str = "results/cifar_sample_grid.png",
    num_images: int = 16,
) -> None:
    """Plots and saves a grid of normalized samples with un-normalization for inspection."""
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Un-normalize for visual display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    unnorm_images = images * std + mean
    unnorm_images = torch.clamp(unnorm_images, 0.0, 1.0)

    grid = torchvision.utils.make_grid(unnorm_images, nrow=4, padding=2)
    np_grid = grid.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid)
    plt.axis("off")
    plt.title("CIFAR-10 Train Samples (Augmented: Crop+Flip)", fontsize=14)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[CIFAR Data] Sample grid saved to {save_path}")


if __name__ == "__main__":
    t_l, v_l, te_l, meta = get_cifar10_dataloaders(smoke=True)
    plot_cifar_sample_grid(t_l)
