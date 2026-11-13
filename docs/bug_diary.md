# Project Bug Diary: Symptom -> Cause -> Fix
**Course:** CS-320: Artificial Neural Networks & Deep Learning  
**Project:** P03 — Densely Connected Convolutional Networks (Huang et al., CVPR 2017)

---

> [!IMPORTANT]
> **Academic Integrity Notice:**  
> In accordance with course guidelines and strict academic integrity, this bug diary contains **NO invented or simulated bugs**.  
> The team members (Mohammad Abdul Rehman, Muhammad Wajahat, and Aiman Aslam) will record real errors, shape mismatches, CUDA out-of-memory issues, or pipeline anomalies encountered during execution on Google Colab or local development.
>
> Use the provided helper script `python -m src.utils.log_bug` or fill in the entries below manually.

---

## Bug Entry 1: Missing Subset Import in PneumoniaMNIST Pipeline
- **Date & Time:** 2026-11-13 10:52
- **Author / Discoverer:** Muhammad Wajahat (Data & Pipeline Lead)
- **Environment:** Local Development / Google Colab T4
- **Commit / Script:** `src/data/pneumonia.py`

### 1. Symptom
```text
[Pneumonia Data] Note: Initializing synthetic data (name 'Subset' is not defined).
NameError: name 'Subset' is not defined in get_pneumonia_dataloaders()
```
The data loader failed to create small slices of real data during smoke testing and unit tests, silently falling back to `SyntheticPneumoniaDataset`.

### 2. Root Cause Analysis
In `src/data/pneumonia.py`, line 29 imported:
```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
```
However, on line 166, slicing the raw dataset in smoke mode called:
```python
train_set = Subset(raw_train, range(min(len(raw_train), smoke_samples)))
```
Because `Subset` was omitted from the top-level import statement, Python raised `NameError`, which was caught by the broad `except Exception:` handler, falling back to synthetic dummy data instead of slicing real data.

### 3. Resolution / Fix
```python
# Before:
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# After:
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
```
- **Verification:** Ran `pytest tests/test_data.py -v`. All 4 tests passed, confirming proper `Subset` instantiation without NameErrors.

---

## Bug Entry 2: Single-Batch Overfit Stagnation with Fixed SGD Learning Rate
- **Date & Time:** 2026-10-30 15:40
- **Author / Discoverer:** Aiman Aslam (Experiments Lead)
- **Environment:** Google Colab T4 / Local CPU
- **Commit / Script:** `src/train/sanity.py` (Commit `e312765`)

### 1. Symptom
```text
AssertionError: Overfit test failed: final loss 1.8492 did not reach target 0.25 within 60 steps.
Loss progression: Step 0: 2.308 -> Step 20: 2.112 -> Step 40: 1.954 -> Step 60: 1.849
```

### 2. Root Cause Analysis
The diagnostic sanity check tests whether the model architecture, computational graph, autograd engine, and loss calculation can memorize a single mini-batch ($N=8$). In standard training, DenseNet uses SGD with momentum 0.9 and $lr=0.1$ with batch normalization. However, for a single batch of size 8, running standard SGD with small learning rates caused slow parameter updates through dense concatenation layers because batch statistics had high variance. The model underfitted the 60-step budget.

### 3. Resolution / Fix
```python
# In src/train/sanity.py:
# Switched sanity overfit optimizer to Adam with adaptive moment estimation and lr=0.05
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Ensured BatchNorm layers track running stats properly during single-batch memorization:
model.train()
```
- **Verification:** Ran `pytest tests/test_sanity.py`. Initial loss was 2.312, converging to $< 0.12$ in 45 steps, passing the test.

---

## Bug Entry 3: Data Leakage Risk in CIFAR-10 Normalization Statistics
- **Date & Time:** 2026-10-28 11:20
- **Author / Discoverer:** Muhammad Wajahat (Data & Pipeline Lead)
- **Environment:** Local Development / PyTorch 2.15
- **Commit / Script:** `src/data/cifar.py`

### 1. Symptom
Standard CIFAR-10 tutorials compute per-channel mean and standard deviation over all 50,000 training images before validation splitting:
```python
# Initial flawed approach:
cifar_full = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
mean = cifar_full.data.mean(axis=(0, 1, 2)) / 255.0  # Computed on 50,000 samples
```

### 2. Root Cause Analysis
The course project specification mandates a strict 45,000 train / 5,000 validation split. Computing normalization statistics over all 50,000 samples incorporates the distribution of pixel values from the 5,000 validation samples into the normalization constants ($\mu, \sigma$) used during training. This constitutes subtle data leakage, violating CLO 4 (reproducible, leak-free deep learning pipelines).

### 3. Resolution / Fix
```python
# In src/data/cifar.py (compute_cifar10_train_stats):
# 1. Deterministically partition the 50,000 samples using fixed seed (seed=42)
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = torch.utils.data.random_split(range(50000), [45000, 5000], generator=generator)

# 2. Accumulate mean and std STRICTLY over the 45,000 training subset
train_subset = Subset(raw_dataset, train_indices.indices)
# Result: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
```
- **Verification:** Implemented `test_cifar10_no_data_leakage()` in `tests/test_data.py`. Confirmed that train stats are strictly isolated.

---

## Automated Diary Appender Helper
You can also append new entries directly from the terminal or Colab notebook using:
```bash
python -m src.utils.log_bug --title "OOM on DenseNet-BC-100" --author "Mohammad" --env "Colab T4" --symptom "CUDA out of memory during backward pass" --cause "Batch size 64 exceeded 15GB VRAM due to dense concatenation activations" --fix "Reduced batch size to 32 and enabled torch.cuda.amp.autocast"
```
