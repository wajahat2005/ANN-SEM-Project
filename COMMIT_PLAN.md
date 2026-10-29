# Git Commit Plan & Team Ownership Guide

To respect academic integrity and university project requirements, AI does not commit directly to git pretending to be students. Each team member must configure their git identity and commit their designated modules.

---

## 1. Member Ownership Matrix

| Team Member | Role | Assigned Files & Modules | Ablation Responsibility |
| :--- | :--- | :--- | :--- |
| **Mohammad Abdul Rehman** | Team & Paper Lead | `src/models/dense_layer.py`<br>`src/models/dense_block.py`<br>`src/models/densenet.py`<br>`src/utils/param_count.py`<br>`src/utils/shape_checks.py`<br>`experiments/reproduce_cifar.py`<br>`tests/test_models.py`<br>`tests/test_param_count.py` | Growth rate $k \in \{8, 12, 24\}$;<br>Bottleneck On vs Off ($B$ on/off) |
| **Muhammad Wajahat** | Data & Pipeline Lead | `src/data/cifar.py`<br>`src/data/pneumonia.py`<br>`src/models/transition.py`<br>`src/eval/metrics.py`<br>`src/eval/calibration.py`<br>`tests/test_data.py` | Compression $\theta \in \{0.5, 1.0\}$;<br>Input resolution ($28, 64, 128$ on PneumoniaMNIST) |
| **Aiman Aslam** | Experiments Lead | `src/models/resnet_cifar.py`<br>`src/models/simple_cnn.py`<br>`src/models/pretrained.py`<br>`src/train/engine.py`<br>`src/train/sanity.py`<br>`src/train/lr_range_test.py`<br>`src/eval/error_analysis.py`<br>`experiments/run_pneumonia.py`<br>`experiments/run_ablations.py`<br>`configs/baseline.yaml`<br>`app/gradio_demo.py`<br>`notebooks/CS320_P03_DenseNet.ipynb`<br>`docs/*`<br>`tests/test_sanity.py` | Dropout $p \in \{0.0, 0.2\}$;<br>Sensitivity threshold calibration ($\ge 95\%$) |

---

## 2. Step-by-Step Commit Instructions

### Step 1: Member 1 — Mohammad Abdul Rehman
```bash
git config user.name "Mohammad Abdul Rehman"
git config user.email "mohammad.abdul.rehman@university.edu"

git add src/models/dense_layer.py src/models/dense_block.py src/models/densenet.py
git add src/utils/param_count.py src/utils/shape_checks.py
git add experiments/reproduce_cifar.py tests/test_models.py tests/test_param_count.py
git commit -m "feat(models): implement _DenseLayer, _DenseBlock, DenseNet, and hand param formulas"
```

### Step 2: Member 2 — Muhammad Wajahat
```bash
git config user.name "Muhammad Wajahat"
git config user.email "muhammad.wajahat@university.edu"

git add src/data/cifar.py src/data/pneumonia.py
git add src/models/transition.py src/eval/metrics.py src/eval/calibration.py tests/test_data.py
git commit -m "feat(data): implement CIFAR-10 & PneumoniaMNIST pipelines, Transition module, and metrics"
```

### Step 3: Member 3 — Aiman Aslam
```bash
git config user.name "Aiman Aslam"
git config user.email "aiman.aslam@university.edu"

git add src/models/resnet_cifar.py src/models/simple_cnn.py src/models/pretrained.py
git add src/train/engine.py src/train/sanity.py src/train/lr_range_test.py
git add src/eval/error_analysis.py experiments/run_pneumonia.py experiments/run_ablations.py
git add configs/baseline.yaml app/gradio_demo.py notebooks/CS320_P03_DenseNet.ipynb tests/test_sanity.py
git commit -m "feat(experiments): implement training engine, sanity checks, ablation runner, and Gradio demo"
```

### Step 4: Shared Documentation & Scaffolding
```bash
git add README.md requirements.txt AI_USE.md COMMIT_PLAN.md LICENSE_AND_DATA.md docs/
git commit -m "docs: add project documentation, architecture onepager, viva prep, and bug diary"
```
