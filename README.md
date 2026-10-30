# CS-320 Project P03: Densely Connected Convolutional Networks (DenseNet)
**Course:** CS-320: Artificial Neural Networks & Deep Learning (Fall 2026)  
**Paper Reference:** Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. *"Densely Connected Convolutional Networks"*, CVPR 2017 (Best Paper Award).  
**Target Environments:** Google Colab (Free T4 GPU with AMP) & Local Development (CPU `--smoke` mode).

---

## 1. Project Overview & Architecture

DenseNet fundamentally rethinks representation propagation in deep neural networks. Unlike ResNet which adds feature maps ($\mathbf{x}_l = H_l(\mathbf{x}_{l-1}) + \mathbf{x}_{l-1}$), DenseNet connects **every layer to every subsequent layer** within a dense block via **channel-wise concatenation**:
$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{l-1}])$$

### Key Architectural Pillars:
1. **Feature Reuse:** Early low-level filters remain directly accessible to deeper layers without linear superimposition.
2. **Growth Rate ($k$):** Each composite function $H_l = \text{BN} \rightarrow \text{ReLU} \rightarrow 3\times 3\text{ Conv}$ outputs $k$ channels (typically $k=12$ or $k=32$), keeping layers narrow.
3. **Bottleneck Layers (DenseNet-B):** Injects a $\text{BN} \rightarrow \text{ReLU} \rightarrow 1\times 1\text{ Conv}$ compressing incoming wide features to $4k$ before the expensive $3\times 3$ convolution.
4. **Transition Compression (DenseNet-C):** Between dense blocks, transition layers apply $\text{BN} \rightarrow 1\times 1\text{ Conv} \rightarrow 2\times 2\text{ AvgPool}$, reducing channels from $m$ to $\lfloor \theta \cdot m \rfloor$ (typically $\theta=0.5$).
5. **DenseNet-BC:** Combines both bottleneck ($4k$) and transition compression ($\theta=0.5$).

---

## 2. Team Division of Ownership

| Member | Role | Core Components Owned | Ablations Assigned |
|---|---|---|---|
| **Mohammad Abdul Rehman** | Team & Paper Lead | `src/models/dense_layer.py`<br/>`src/models/dense_block.py`<br/>`src/models/densenet.py`<br/>`src/utils/param_count.py`<br/>`src/utils/shape_checks.py`<br/>`experiments/reproduce_cifar.py` | 1. Growth rate $k \in \{8, 12, 24\}$<br/>2. Bottleneck on/off (DenseNet vs DenseNet-B) |
| **Muhammad Wajahat** | Data & Pipeline Lead | `src/data/cifar.py`<br/>`src/data/pneumonia.py`<br/>`src/models/transition.py`<br/>`src/eval/metrics.py` | 1. Compression $\theta \in \{0.5, 1.0\}$<br/>2. Input resolution ($28 \times 28, 64 \times 64, 128 \times 128$) |
| **Aiman Aslam** | Experiments Lead | `src/train/engine.py`<br/>`src/train/sanity.py`<br/>`src/train/lr_range_test.py`<br/>`src/models/resnet_cifar.py`<br/>`src/models/simple_cnn.py`<br/>`src/models/pretrained.py`<br/>`src/eval/calibration.py`<br/>`src/eval/error_analysis.py`<br/>`app/gradio_demo.py` | 1. Dropout $p \in \{0.0, 0.2\}$<br/>2. High-sensitivity clinical threshold calibration |

---

## 3. Analytical Parameter Verification

All parameter counts are calculated from theoretical first principles and rigorously verified against PyTorch's `sum(p.numel())`:
- **Formula for Conv3x3 Layer:** $\text{Params} = 2 \cdot C_{\text{in}} + 9 \cdot C_{\text{in}} \cdot k$ (BN affine parameters + Conv weights).
- **Block 1, Layer 1 ($C_{\text{in}}=16, k=12$):**  
  $\text{BN}(16) = 32$; $\text{Conv}_{3\times 3}(16 \rightarrow 12) = 16 \times 12 \times 9 = 1,728$. Total = **1,760 parameters (Exact Match)**.
- **DenseNet-40 ($k=12$):** Hand = **1,019,722**; PyTorch = **1,019,722** (~1.02M).
- **DenseNet-BC-100 ($k=12$):** Hand = **769,162**; PyTorch = **769,162** (~0.77M).
- **Custom DenseNet-121 vs Torchvision:** **7,978,856** vs **7,978,856** (**0 difference, Exact Match**).

---

## 4. Repository Structure

```text
ANN-SEM-Project/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Locked production dependencies
├── AI_USE.md                      # Academic disclosure of AI scaffolding
├── COMMIT_PLAN.md                 # Per-member git commit commands & ownership
├── LICENSE_AND_DATA.md            # MIT License and CIFAR-10 / MedMNIST attribution
├── configs/
│   └── baseline.yaml              # Single fixed baseline configuration for all ablations
├── src/
│   ├── models/
│   │   ├── dense_layer.py         # _DenseLayer with pre-activation, bottleneck, and dropout
│   │   ├── dense_block.py         # _DenseBlock iterative channel concatenation
│   │   ├── transition.py          # _Transition downsampling and theta compression
│   │   ├── densenet.py            # DenseNet full model (CIFAR and ImageNet stems)
│   │   ├── resnet_cifar.py        # ResNet-56 baseline for CIFAR-10+
│   │   ├── simple_cnn.py          # 5-layer baseline CNN for PneumoniaMNIST
│   │   └── pretrained.py          # Torchvision DenseNet-121 fine-tune wrapper
│   ├── data/
│   │   ├── cifar.py               # CIFAR-10 45k/5k/10k splits & train-only normalization
│   │   └── pneumonia.py           # MedMNIST PneumoniaMNIST loader & class imbalance weights
│   ├── train/
│   │   ├── engine.py              # AMP training engine, Drive checkpointing, CSV logging
│   │   ├── sanity.py              # Theoretical ln(C) loss check, gradient flow, 1-batch overfit
│   │   └── lr_range_test.py       # Exponential LR finder (1e-6 to 2.0)
│   ├── eval/
│   │   ├── metrics.py             # Accuracy, ROC-AUC, Sensitivity, Specificity, F1, Confusion Matrix
│   │   ├── calibration.py         # Threshold calibration (>=95% sens), ECE, Temperature Scaling
│   │   └── error_analysis.py      # Diagnostic misclassification grids & per-class error
│   └── utils/
│       ├── param_count.py         # Analytical hand proofs vs sum(p.numel())
│       ├── shape_checks.py        # Tensor shape assertion engine
│       └── log_bug.py             # Automated bug diary appender helper
├── experiments/
│   ├── reproduce_cifar.py         # Reproduction runner for DenseNet-40, BC-100, ResNet-56
│   ├── run_pneumonia.py           # Medical benchmark (Simple CNN, Custom DenseNet, DenseNet-121)
│   └── run_ablations.py           # OFAT ablation suite writing to results/ablations.csv
├── app/
│   └── gradio_demo.py             # Web UI with Figure 5 feature reuse heatmap & Grad-CAM
├── notebooks/
│   └── CS320_P03_DenseNet.ipynb   # Top-to-bottom Colab notebook (Sections 0-10)
├── docs/
│   ├── architecture_onepager.md   # 1-page reference diagram, formulas, 3 self-check questions
│   ├── report_draft.md            # 6-page project report template
│   ├── bug_diary.md               # Real debugging log (academic integrity compliant)
│   └── viva_prep.md               # Viva preparation walkthrough & 15 examination questions
├── tests/
│   ├── test_models.py             # Unit tests for all model forward passes and shapes
│   ├── test_param_count.py        # Parameter count verification tests
│   ├── test_data.py               # Data split isolation and normalization leak tests
│   └── test_sanity.py             # Sanity engine verification tests
└── results/
    └── shape_table.md             # Complete 47-layer tensor shape trace
```

---

## 5. Quickstart & Installation

### Option A: Local Setup (CPU Smoke Testing)
```bash
# Clone the repository
git clone https://github.com/YourTeam/ANN-SEM-Project.git
cd ANN-SEM-Project

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the complete test suite
pytest tests/ -v
```

### Option B: Google Colab Setup (GPU Full Runs)
Open [notebooks/CS320_P03_DenseNet.ipynb](file:///notebooks/CS320_P03_DenseNet.ipynb) directly in Google Colab:
1. Navigate to **Runtime > Change runtime type** and select **T4 GPU**.
2. Run Section 0 to mount Google Drive and install requirements.
3. Checkpoints are automatically saved to `/content/drive/MyDrive/cs320_checkpoints/`.

---

## 6. How to Run Experiments

### 1. CIFAR-10+ Reproduction
```bash
# Smoke test (local CPU, 2 epochs, 128 samples)
python -m experiments.reproduce_cifar --model densenet40 --smoke

# Full reproduction on Google Colab (300 epochs, AMP enabled)
python -m experiments.reproduce_cifar --model densenet40 --epochs 300 --device cuda
python -m experiments.reproduce_cifar --model densenet_bc_100 --epochs 300 --device cuda
python -m experiments.reproduce_cifar --model resnet56 --epochs 300 --device cuda
```

### 2. MedMNIST PneumoniaMNIST Medical Transfer
```bash
# Smoke test (local CPU)
python -m experiments.run_pneumonia --model custom_densenet --smoke

# Full benchmark on Colab (40 epochs)
python -m experiments.run_pneumonia --model simple_cnn --epochs 40 --device cuda
python -m experiments.run_pneumonia --model custom_densenet --epochs 40 --device cuda
python -m experiments.run_pneumonia --model pretrained_densenet121 --epochs 40 --device cuda
```

### 3. Systematic Ablation Suite (OFAT)
```bash
# Smoke test a single ablation factor
python -m experiments.run_ablations --factor growth_rate --smoke

# Run full ablation suite on Colab (writes results/ablations.csv & factor plots)
python -m experiments.run_ablations --all --epochs 20 --device cuda
```

### 4. Interactive Gradio Demo
```bash
# Launch interactive web interface locally or in Colab
python -m app.gradio_demo --share
```

---

## 7. Diagnostic Sanity Suite

Run Karpathy-style pre-flight sanity checks before training:
```bash
# 1. Analytical Parameter Count Check
python -m src.utils.param_count

# 2. Layer Shape Verification
python -m src.utils.shape_checks

# 3. Learning Rate Range Finder
python -m src.train.lr_range_test --dataset cifar10 --model densenet40
```

---

## 8. Academic Integrity & Transparency

- **AI Usage:** Scaffolding generation by AI assistant is disclosed in [AI_USE.md](file:///AI_USE.md).
- **Commit History:** Per-member authorship mapping is defined in [COMMIT_PLAN.md](file:///COMMIT_PLAN.md).
- **Data Attribution:** CIFAR-10 and MedMNIST PneumoniaMNIST licensing and citations are in [LICENSE_AND_DATA.md](file:///LICENSE_AND_DATA.md).
- **Bug Diary:** Real errors and resolutions encountered during runs are documented in [docs/bug_diary.md](file:///docs/bug_diary.md).
