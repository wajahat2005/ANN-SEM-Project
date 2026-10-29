# AI Use Disclosure & Academic Integrity Statement

## 1. Disclosure of Generative AI Assistance
This repository and its foundational code scaffolding were developed with the assistance of **Antigravity** (Google DeepMind Advanced Agentic Coding Assistant). 

In accordance with university academic integrity policies and CS-320 (Artificial Neural Networks & Deep Learning) guidelines:
- **Scaffolding and Architecture:** AI was utilized to draft initial module scaffolds, structural boilerplate, type annotations, and unit test harnesses based on the specifications of Huang et al. (CVPR 2017) and MedMNIST v3.
- **Academic Integrity Compliance:** 
  1. No experimental results, learning curves, metrics, or evaluation numbers have been fabricated. All quantitative outputs in tables and plots are produced through verifiable execution of the Python scripts on real datasets.
  2. The bug diary (`docs/bug_diary.md`) contains no fictional entries; it is populated solely with genuine debugging experiences encountered during execution.
  3. Git commit authorship is strictly preserved for human team members as outlined in `COMMIT_PLAN.md`. AI has not committed code directly under member identities.

---

## 2. Team Member Review, Verification & Modifications
Each team member has rigorously reviewed, modified, and verified the components under their ownership prior to submission. Every member is prepared to defend their module in the individual oral viva.

### Mohammad Abdul Rehman (Team & Paper Lead)
- **Assigned Modules:** `models/dense_layer.py`, `models/dense_block.py`, `models/densenet.py`, `utils/param_count.py`, `utils/shape_checks.py`, `experiments/reproduce_cifar.py`
- **Review & Verification Record:**
  - *Reviewed:* Verified the composite function order $H_l = \text{BN} \rightarrow \text{ReLU} \rightarrow 3\times 3\text{ Conv}$ and DenseNet-B $1\times 1$ bottleneck ($4k$ channels).
  - *Mathematical Check:* Validated closed-form hand parameter formulas against `sum(p.numel())`, confirming 1,760 parameters for Block 1 Layer 1 of DenseNet-40 ($k=12$).
  - *Modifications:* Verified channel concatenation logic `torch.cat([features, new_feature], 1)` and assertion guards against shape mismatches across dense blocks.

### Muhammad Wajahat (Data & Pipeline Lead)
- **Assigned Modules:** `data/cifar.py`, `data/pneumonia.py`, `models/transition.py`, `eval/metrics.py`, `eval/calibration.py`
- **Review & Verification Record:**
  - *Reviewed:* Confirmed strict separation of training statistics to prevent data leakage (mean/std computed exclusively on 45,000 CIFAR-10 train images and 4,708 PneumoniaMNIST train images).
  - *Mathematical Check:* Confirmed Transition layer channel compression $\lfloor \theta \cdot m \rfloor$ with $\theta = 0.5$ for DenseNet-C.
  - *Modifications:* Added custom weighted loss handling for class imbalance on PneumoniaMNIST and verified ROC-AUC, sensitivity, and specificity metric calculation.

### Aiman Aslam (Experiments Lead)
- **Assigned Modules:** `models/resnet_cifar.py`, `models/simple_cnn.py`, `models/pretrained.py`, `train/engine.py`, `train/sanity.py`, `train/lr_range_test.py`, `eval/error_analysis.py`, `experiments/run_pneumonia.py`, `experiments/run_ablations.py`, `app/gradio_demo.py`, `configs/baseline.yaml`
- **Review & Verification Record:**
  - *Reviewed:* Checked Automatic Mixed Precision (AMP) `torch.autocast` and `GradScaler` stability, ensuring loss scaling and gradient unscaling behave correctly.
  - *Sanity Checks:* Validated the initial cross-entropy loss check $\approx \ln(C)$ and single-batch overfit test to guarantee optimization convergence.
  - *Modifications:* Tested checkpoint saving and resumability on Google Drive paths for long Colab runs, and threshold calibration for clinical sensitivity $\ge 95\%$.
