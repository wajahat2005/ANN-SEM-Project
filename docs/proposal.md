# CS-320 Semester Project Proposal (M0)
**Course:** CS-320 Artificial Neural Networks & Deep Learning (Fall 2026)  
**Institution:** NUST Balochistan Campus (NBC), Department of Computer Science / AI  
**Instructor:** Dr. Zirak Khan  
**Project Track:** P03 — Densely Connected Convolutional Networks (DenseNet)  
**Date Submitted:** October 16, 2026 (Milestone M0) / Reviewed: November 13, 2026 (Week 10)

---

## 1. Landmark Paper Overview (In Our Own Words)

**Paper:** Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. *"Densely Connected Convolutional Networks"*, IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017 (Best Paper Award).

### Motivation & Central Innovation
As convolutional networks grow in depth, they face two fundamental bottlenecks: (1) **vanishing/exploding gradients**, where error signals dissipate during backpropagation across many layers, and (2) **feature map redundancy**, where deeper layers re-learn low-level spatial features because early representations are obscured. ResNet partially mitigated vanishing gradients via identity skip connections ($\mathbf{x}_l = H_l(\mathbf{x}_{l-1}) + \mathbf{x}_{l-1}$), but element-wise summation linearly combines features, potentially degrading early, fine-grained representations.

DenseNet introduces a radically different structural paradigm: **direct feature reuse through iterative channel-wise concatenation**:
$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{l-1}])$$
where $[\mathbf{x}_0, \dots, \mathbf{x}_{l-1}]$ represents the channel-wise concatenation of all prior feature representations within a dense block. Each composite layer $H_l = \text{BN} \rightarrow \text{ReLU} \rightarrow 3\times 3\text{ Conv}$ outputs a narrow set of $k$ channels (the **growth rate**). This design provides two decisive advantages:
1. **Implicit Deep Supervision & Gradient Highways:** Every layer has direct gradient paths connecting it to the loss function and to all preceding layers, completely eliminating vanishing gradients.
2. **Extreme Parameter Efficiency:** Because early features are permanently accessible to later layers, subsequent layers only need to produce new, complementary features rather than copying redundant representations.

To maintain manageable parameter scaling in deeper architectures, **DenseNet-B** introduces $1 \times 1$ bottleneck convolutions compressing inputs to $4k$ before the $3 \times 3$ convolution, while **DenseNet-C** applies $\theta=0.5$ channel compression in transition pooling layers between blocks (**DenseNet-BC**).

---

## 2. Target Datasets & Colab Loading Verification

Our project evaluates DenseNet across two distinct computer vision domains:

### A. Reproduction Benchmark: CIFAR-10+ (Natural Images)
- **Modality:** 60,000 $32 \times 32$ RGB images across 10 object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Split Protocol:** Deterministic 45,000 training, 5,000 validation, 10,000 testing (seed=42).
- **Zero-Leakage Safeguard:** Normalization statistics ($\mu = [0.4914, 0.4822, 0.4465], \sigma = [0.2470, 0.2435, 0.2616]$) are computed **strictly on the 45k training partition**, ensuring zero test/val distribution contamination.
- **Loading Status:** Verified loading in Google Colab T4 environment via `src/data/cifar.py`.

### B. Extension Benchmark: MedMNIST PneumoniaMNIST (Pediatric Radiography)
- **Modality:** Pediatric chest X-ray radiographs ($64 \times 64$ grayscale, single-channel) classified as **Normal** (Class 0) vs. **Pneumonia** (Class 1).
- **Split Protocol:** Official MedMNIST benchmark partition: 4,708 train / 524 val / 624 test.
- **Clinical Challenge (Class Imbalance):** The training set exhibits a ~1:2.88 class imbalance (1,214 Normal vs. 3,494 Pneumonia). We address this via cost-sensitive class-weighted cross-entropy loss ($w_{\text{Normal}} \approx 1.939, w_{\text{Pneumonia}} \approx 0.674$).
- **Loading Status:** Verified loading in Google Colab T4 environment via `src/data/pneumonia.py`.

---

## 3. Team Member Role Distribution & Ownership Matrix

In compliance with CS-320 project guidelines, the workload is distributed across three designated functional roles. Each member owns specific modules, runs distinct ablations, and is individually accountable:

| Team Member | Official Role | Assigned Source Modules & Tests | Assigned Ablations | Viva Ownership Focus |
| :--- | :--- | :--- | :--- | :--- |
| **Mohammad Abdul Rehman** | **Team & Paper Lead** (1st Member) | • `src/models/dense_layer.py`<br/>• `src/models/dense_block.py`<br/>• `src/models/densenet.py`<br/>• `src/utils/param_count.py`<br/>• `src/utils/shape_checks.py`<br/>• `experiments/reproduce_cifar.py`<br/>• `tests/test_models.py`<br/>• `tests/test_param_count.py` | 1. Growth rate $k \in \{8, 12, 24\}$<br/>2. Bottleneck on/off ($B$ on/off) | Architecture derivation, hand parameter proofs (1,760 params for Block 1 Layer 1), CIFAR-10 reproduction dynamics. |
| **Muhammad Wajahat** | **Data & Pipeline Lead** | • `src/data/cifar.py`<br/>• `src/data/pneumonia.py`<br/>• `src/models/transition.py`<br/>• `src/eval/metrics.py`<br/>• `src/eval/calibration.py`<br/>• `tests/test_data.py` | 1. Compression $\theta \in \{0.5, 1.0\}$<br/>2. Input resolution ($28, 64, 128$) | Data loading pipelines, leak-free normalization, transition layer pooling, evaluation metrics. |
| **Aiman Aslam** | **Experiments Lead** | • `src/models/resnet_cifar.py`<br/>• `src/models/simple_cnn.py`<br/>• `src/models/pretrained.py`<br/>• `src/train/engine.py`<br/>• `src/train/sanity.py`<br/>• `src/train/lr_range_test.py`<br/>• `src/eval/error_analysis.py`<br/>• `experiments/run_pneumonia.py`<br/>• `experiments/run_ablations.py`<br/>• `configs/baseline.yaml`<br/>• `app/gradio_demo.py`<br/>• `notebooks/CS320_P03_DenseNet.ipynb`<br/>• `tests/test_sanity.py` | 1. Dropout $p \in \{0.0, 0.2\}$<br/>2. Sensitivity calibration ($\ge 95\%$) | AMP training engine, Drive checkpointing, sanity checks ($\ln(C)$, overfit, gradient flow), Gradio web demo. |

---

## 4. Semester Milestones & Schedule Alignment

- **Week 6 (16 Oct) [M0]:** Proposal submitted; datasets loading in Colab; GitHub repository established. *(Completed)*
- **Week 8 (30 Oct) [M1-2]:** Architecture walkthrough, hand parameter count proofs, DataLoader tensor verification, overfit-one-batch sanity test, baseline training curves. *(Completed)*
- **Week 9:** Mid-semester examinations. *(Completed)*
- **Week 10 (13 Nov):** Full pre-flight code freeze, PyTorch checklist validation, bug diary curation, and reproduction pipeline readiness. *(Completed)*
- **Week 11 (20 Nov) [M3]:** CIFAR-10+ key result reproduction (DenseNet-40 vs. ResNet-56) with paper comparison. *(Scheduled)*
- **Week 13 (4 Dec) [M4]:** MedMNIST PneumoniaMNIST medical transfer benchmark, systematic OFAT ablations, and error analysis. *(Scheduled)*
- **Week 15 (18 Dec) [M5]:** Final technical report, printed architecture one-pager, and demo rehearsal. *(Scheduled)*
- **Weeks 16–17 (21 Dec – 1 Jan):** Interactive live demo, project presentation, and individual viva examinations. *(Scheduled)*
