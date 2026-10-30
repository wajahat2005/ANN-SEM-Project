# CS-320 Project P03: Comprehensive Viva Examination Guide
**Course:** CS-320: Artificial Neural Networks & Deep Learning  
**Project:** P03 — Densely Connected Convolutional Networks (Huang et al., CVPR 2017)  
**Target Audience:** Mohammad Abdul Rehman, Muhammad Wajahat, Aiman Aslam

---

## Part 1: Per-Member Walkthrough & Architecture Ownership

### 1. Mohammad Abdul Rehman — Team & Paper Lead
* **Primary Code Ownership:**
  - `src/models/dense_layer.py`: `_DenseLayer` (BN-ReLU-Conv pre-activation, bottleneck $4k$, dropout, hand parameter formula).
  - `src/models/dense_block.py`: `_DenseBlock` (iterative concatenation along `dim=1`, shape verification).
  - `src/models/densenet.py`: Full `DenseNet` (CIFAR stem vs ImageNet stem, dense blocks, transitions, classification head).
  - `src/utils/param_count.py`: Analytical parameter proofs vs `sum(p.numel())`.
  - `src/utils/shape_checks.py`: Shape assertions across all 47 layers for a $1\times 3\times 32\times 32$ tensor.
  - `experiments/reproduce_cifar.py`: CIFAR-10+ reproduction pipeline.
* **Ablations Assigned:**
  - Growth rate $k \in \{8, 12, 24\}$.
  - Bottleneck on/off (DenseNet vs DenseNet-B).
* **Viva Explanatory Points:**
  - *Mathematical Concatenation:* Explain why $\mathbf{x}_l = H_l([\mathbf{x}_0, \dots, \mathbf{x}_{l-1}])$ preserves all feature representations rather than superimposing them as in ResNet $\mathbf{x}_l = H_l(\mathbf{x}_{l-1}) + \mathbf{x}_{l-1}$.
  - *Parameter Efficiency Proof:* Walk through the calculation for Block 1 Layer 1: $\text{BN}(16) = 32$ params; $3\times 3\text{ Conv}(16 \rightarrow 12) = 16 \times 12 \times 9 = 1,728$ params; Total $= 1,760$ params.
  - *Bottleneck Necessity:* Explain why $1\times 1$ conv compressing $C_{\text{in}}$ to $4k$ prevents parameter explosion when $C_{\text{in}}$ grows to hundreds of channels in later layers.

---

### 2. Muhammad Wajahat — Data & Pipeline Lead
* **Primary Code Ownership:**
  - `src/data/cifar.py`: 45k train / 5k val / 10k test split (seed=42), train-only normalization stats ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]), RandomCrop + HorizontalFlip.
  - `src/data/pneumonia.py`: MedMNIST PneumoniaMNIST loader ($64\times 64$, 28, 128), train-only stats (0.5719, 0.1684), class imbalance weights.
  - `src/models/transition.py`: `_Transition` layer (BN $\rightarrow 1\times 1$ Conv $\rightarrow 2\times 2$ AvgPool, $\lfloor \theta \cdot m \rfloor$ compression).
  - `src/eval/metrics.py`: Classification accuracy, ROC-AUC, Sensitivity, Specificity, F1-score, Confusion Matrix generation.
* **Ablations Assigned:**
  - Compression rate $\theta \in \{0.5, 1.0\}$ (DenseNet vs DenseNet-C).
  - Input resolution on PneumoniaMNIST ($28 \times 28$, $64 \times 64$, $128 \times 128$).
* **Viva Explanatory Points:**
  - *Data Leakage Prevention:* Emphasize that normalization mean and std were calculated **strictly on the 45,000 training images**, never touching validation or test data.
  - *Transition Pooling:* Explain why transitions are necessary: feature map spatial dimensions must remain identical inside a dense block for concatenation; transition layers perform required downsampling ($2\times 2$ average pooling with stride 2).
  - *Class Imbalance:* Explain why class-weighted loss ($w_c = \frac{N}{2 N_c}$) was chosen over oversampling (oversampling distorts posterior calibration in clinical triage).

---

### 3. Aiman Aslam — Experiments Lead
* **Primary Code Ownership:**
  - `src/train/engine.py`: AMP training loop (`torch.autocast`, `GradScaler`), resumable Drive checkpointing (`best_val.pt`, `last.pt`), epoch CSV logging, automated curve plots.
  - `src/train/sanity.py`: Initial loss theoretical check ($\approx \ln(C)$), non-zero gradient check, single mini-batch overfit.
  - `src/train/lr_range_test.py`: Exponential learning rate sweep from $10^{-6}$ to $2.0$ with steepest descent locator.
  - `src/models/resnet_cifar.py` & `src/models/simple_cnn.py`: Baseline models for comparison.
  - `src/models/pretrained.py`: Pretrained DenseNet-121 fine-tuning wrapper (single-channel adaptation, freeze/unfreeze).
  - `src/eval/calibration.py`: Clinical threshold calibration for $\ge 95\%$ sensitivity, ECE, reliability diagrams, temperature scaling.
  - `src/eval/error_analysis.py`: Misclassified sample extraction and confusion analysis.
  - `app/gradio_demo.py`: Interactive web UI with Figure 5 feature-reuse heatmap and Grad-CAM saliency.
* **Ablations Assigned:**
  - Dropout rate $p \in \{0.0, 0.2\}$.
  - Clinical sensitivity threshold calibration ($\tau = 0.5$ vs calibrated $\tau$).
* **Viva Explanatory Points:**
  - *AMP & Numerical Stability:* Explain how `torch.autocast('cuda')` runs convolutions in FP16 while `GradScaler` prevents underflow of small gradients by dynamically scaling loss.
  - *Clinical Sensitivity Calibration:* Why standard 0.5 threshold is dangerous in medicine: false negatives (missing pneumonia) can be fatal; lowering threshold on validation set to achieve $\ge 95\%$ sensitivity prioritizes patient recall.
  - *Temperature Scaling:* Why post-processing logits via $z / T$ (where $T > 0$ is learned via L-BFGS to minimize NLL) improves ECE without changing classification order.

---

## Part 2: 15 Core Viva Questions & Expert Answers

### Q1: How does DenseNet alleviate the vanishing gradient problem?
**Answer:**  
In conventional deep feedforward networks, supervision gradients must backpropagate through dozens or hundreds of parameterized layers sequentially, decaying exponentially (vanishing) due to repeated matrix multiplications.  
In DenseNet, every layer has a direct skip connection from its output to all subsequent layers in the block, and receives direct gradient signals from the final loss during backpropagation:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \sum_{k=l+1}^L \frac{\partial \mathcal{L}}{\partial \mathbf{x}_k} \frac{\partial \mathbf{x}_k}{\partial \mathbf{x}_l}$$
This creates an "implicit deep supervision" effect, allowing gradients to flow effortlessly to early layers regardless of total network depth.

---

### Q2: How does DenseNet's channel concatenation differ from ResNet's additive identity connection?
**Answer:**  
- **ResNet:** Adds feature representations element-wise ($\mathbf{x}_l = H_l(\mathbf{x}_{l-1}) + \mathbf{x}_{l-1}$). Both $H_l(\mathbf{x}_{l-1})$ and $\mathbf{x}_{l-1}$ must share identical channel dimensions. While addition allows gradient highway flow, it linearly superimposes information, which can impede information flow if early low-level features are altered.
- **DenseNet:** Concatenates feature representations along the channel axis ($\mathbf{x}_l = H_l([\mathbf{x}_0, \dots, \mathbf{x}_{l-1}])$). Each layer produces only a small number of channels ($k \ll C_{\text{in}}$), preserving earlier features unmodified in their original form and allowing subsequent layers to freely access any previously learned feature.

---

### Q3: Why does DenseNet utilize pre-activation (BN $\rightarrow$ ReLU $\rightarrow$ Conv) instead of standard post-activation (Conv $\rightarrow$ BN $\rightarrow$ ReLU)?
**Answer:**  
He et al. (2016, "Identity Mappings in Deep Residual Networks") demonstrated that placing Batch Normalization and ReLU *before* the convolution produces cleaner information propagation across skip connections.  
In DenseNet, pre-activation ensures that the feature maps concatenated together are already normalized and non-linearly activated. Furthermore, convolutions in pre-activation layers omit bias parameters because the subsequent BN affine transformation ($\gamma \hat{x} + \beta$) mathematically absorbs any constant additive bias, saving parameters.

---

### Q4: What is the growth rate $k$? What happens if we increase $k$ from 12 to 24?
**Answer:**  
The growth rate $k$ dictates the number of new feature maps produced by each composite function $H_l$. It regulates the rate at which "new knowledge" is added to the collective representation at each step.  
**Live Hyperparameter Impact:**  
- **Parameter Count:** The input channels to layer $l$ grow as $k_0 + (l-1)k$. In standard DenseNet (no bottleneck), layer $l$ has $9 \cdot [k_0 + (l-1)k] \cdot k$ conv parameters, which scales quadratically with $k$ ($O(k^2)$). For DenseNet-40, increasing $k$ from 12 to 24 roughly quadruples block parameters from ~1.0M to ~3.8M.
- **Memory & Throughput:** GPU activation memory scales directly with $k$, and training time increases.
- **Accuracy:** The network gains significantly higher representational capacity, improving top-1 accuracy on complex datasets until limited by overfitting.

---

### Q5: What is the bottleneck layer (DenseNet-B), why is the factor $4k$, and what happens if we turn it off?
**Answer:**  
In deep DenseNets, later layers receive inputs with hundreds or thousands of channels ($C_{\text{in}} \gg k$). A standard $3\times 3$ convolution directly on $C_{\text{in}}$ would require $9 \cdot C_{\text{in}} \cdot k$ parameters, becoming prohibitively expensive.  
DenseNet-B injects a $1\times 1$ convolution before the $3\times 3$ convolution:
$$\text{Input } (C_{\text{in}}) \xrightarrow{\text{BN-ReLU-1x1 Conv}} 4k \text{ feature maps} \xrightarrow{\text{BN-ReLU-3x3 Conv}} k \text{ feature maps}$$
The $4k$ factor was empirically identified by Huang et al. as an optimal trade-off: it compresses the wide input into an information-dense subspace sufficient for $3\times 3$ feature extraction while keeping FLOPs low.  
**Live Hyperparameter Impact:**  
Turning bottleneck OFF in deep networks (e.g., DenseNet-100 or DenseNet-121) causes the parameter count to explode, leading to higher compute costs and potential GPU out-of-memory errors during training.

---

### Q6: What does the transition layer do? Why does DenseNet-C use compression $\theta=0.5$?
**Answer:**  
Transition layers sit between consecutive Dense Blocks. They serve two vital functions:
1. **Spatial Downsampling:** Applies $2\times 2$ average pooling with stride 2 to halve $H$ and $W$ ($32 \rightarrow 16 \rightarrow 8$).
2. **Channel Compression ($\theta$):** A $1\times 1$ convolution reduces $m$ accumulated feature maps to $\lfloor \theta \cdot m \rfloor$.  
In DenseNet-BC, $\theta = 0.5$, discarding 50% of the redundant feature channels before entering the next block.  
**Live Hyperparameter Impact:**  
If $\theta=1.0$ (no compression), transition layers retain all $m$ channels. This increases parameter count and FLOPs in subsequent blocks by ~30–40% and expands the input to the final linear classifier.

---

### Q7: Why does DenseNet divide layers into multiple Dense Blocks instead of one single monolithic block?
**Answer:**  
Direct concatenation $[\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{l-1}]$ strictly requires all tensors to share the exact same spatial dimensions $(H \times W)$. Downsampling feature map resolution is essential in CNNs to enlarge the receptive field and make spatial representations translation-invariant.  
Because downsampled feature maps cannot be concatenated with high-resolution feature maps without spatial interpolation, the architecture is partitioned into Dense Blocks of constant spatial size, separated by Transition Layers where pooling occurs.

---

### Q8: Why does DenseNet consume more GPU memory (VRAM) during training than ResNet despite having fewer parameters?
**Answer:**  
DenseNet's high memory consumption stems from **activation storage**, not parameter storage.  
In ResNet, the summation $\mathbf{x} + F(\mathbf{x})$ can be computed in-place, overwriting intermediate activations. In DenseNet, concatenation creates new tensors, and PyTorch's autograd engine must cache the activations of every intermediate layer in memory to compute gradients during the backward pass:
$$\text{Stored Activations} \propto \sum_{l=1}^L C_{\text{in}}(l) \times H \times W$$
This scales quadratically ($O(L^2)$) with block depth. While parameters are compact, the activation tensor footprint is large, requiring Automatic Mixed Precision (AMP) or gradient checkpointing.

---

### Q9: In the classification head, why is Global Average Pooling (GAP) used before the linear classifier instead of flattening all feature maps?
**Answer:**  
If the final feature map of size $(C, H, W)$ were flattened into a vector of length $C \times H \times W$, a fully-connected layer to $K$ classes would require:
$$\text{Params} = C \times H \times W \times K$$
For $C=448, H=8, W=8, K=10$, flattening would require $448 \times 64 \times 10 = 286,720$ parameters in the head alone.  
Global Average Pooling collapses $(C, H, W) \rightarrow (C, 1, 1)$, reducing head parameters to $C \times K = 448 \times 10 = 4,480$ parameters (a $\sim 64\times$ reduction). Additionally, GAP makes the network invariant to spatial translations and acts as a structural regularizer against overfitting.

---

### Q10: How did the team prevent data leakage during preprocessing on CIFAR-10 and PneumoniaMNIST?
**Answer:**  
In both datasets:
1. **Split-Before-Statistics:** Data was partitioned into training, validation, and test splits *prior* to computing any summary statistics.
2. **Train-Only Computation:** The per-channel normalization mean and standard deviation were computed **strictly on the training split** (45,000 CIFAR-10 images; 4,708 PneumoniaMNIST images).
3. **Application:** The computed training statistics were applied to transform the validation and test sets. Calculating mean/std over the entire dataset prior to splitting leaks distributional properties of the test set into the model, artificially inflating validation metrics.

---

### Q11: In PneumoniaMNIST, why was class-weighted loss chosen over oversampling or undersampling?
**Answer:**  
PneumoniaMNIST has an inherent class imbalance of ~1 : 2.88 (1,214 Normal vs. 3,494 Pneumonia in training).  
- **Undersampling:** Discards over 2,200 clinical X-ray images, sacrificing valuable pathological diversity.
- **Oversampling:** Replicates minority class images, increasing risk of overfitting to specific patient artifacts and distorting empirical prior probabilities, which degrades probabilistic posterior calibration.
- **Class-Weighted Loss:** Computes inverse-frequency loss weights:
  $$w_c = \frac{N_{\text{total}}}{2 \cdot N_c} \implies w_{\text{Normal}} \approx 1.94, \quad w_{\text{Pneumonia}} \approx 0.67$$
  This penalizes errors on the minority class during optimization while preserving natural data prevalence and sample diversity.

---

### Q12: Why is classification accuracy an inadequate metric for PneumoniaMNIST? Why calibrate to $\ge 95\%$ sensitivity?
**Answer:**  
On imbalanced datasets, a trivial majority classifier predicting "Pneumonia" on every sample achieves $\sim 74\%$ accuracy while having zero ability to detect normal patients.  
In clinical diagnostic triage, **False Negatives (missing a sick patient with pneumonia)** carry catastrophic clinical risk (untreated progression, sepsis, mortality), whereas False Positives (flagging a healthy patient) merely prompt a secondary physician review.  
Therefore, the team prioritized **ROC-AUC** (which is threshold-independent) and calibrated the decision threshold $\tau$ on the validation set to enforce $\ge 95\%$ Sensitivity ($\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \ge 0.95$), and evaluated the resulting Specificity on the unseen test set.

---

### Q13: What is Expected Calibration Error (ECE), and how does Temperature Scaling improve it?
**Answer:**  
Modern deep networks are frequently overconfident: predicted softmax probabilities do not reflect real empirical accuracy (e.g., predictions with confidence 0.90 may only be correct 75% of the time).  
ECE partitions predictions into $M$ confidence bins $B_m$ and computes the weighted average difference between accuracy and confidence:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
**Temperature Scaling:** Post-processes logits $z$ by dividing by a single scalar parameter $T > 0$:
$$\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
$T$ is optimized on the validation set via L-BFGS to minimize Negative Log-Likelihood (NLL). Because $T$ is monotonic, class rankings and accuracy are completely unchanged, but softmax confidence is "softened" ($T > 1$), dramatically reducing ECE to produce reliable probabilities for clinical decision-making.

---

### Q14: Explain the Figure 5 "Feature-Reuse Heatmap" from the Huang et al. paper. What does it prove?
**Answer:**  
Figure 5 computes the average absolute magnitude of the convolutional weights connecting layer $l$ to each preceding layer $s < l$ within a dense block:
$$A_{l, s} = \frac{1}{C_{\text{out}} \times k_s \times 9} \sum_{i, j, u, v} |W_{i, j, u, v}|$$
The resulting heatmap shows:
1. **Active Early Feature Reuse:** Layers deep within the block place high weight not only on their immediate predecessor ($l-1$), but also directly on early layers (including layer 0, the block input).
2. **Transition Layer Filtering:** Transition layers distribute their attention across all layers in the preceding block, confirming that both low-level and high-level representations are propagated forward.
3. This directly disproves the traditional hypothesis that deep networks only use representations from the immediately preceding layer.

---

### Q15: What would happen if we trained DenseNet using standard vanilla SGD (no Nesterov momentum, no learning rate schedule)?
**Answer:**  
Huang et al. trained DenseNet using SGD with **Nesterov momentum (0.9)**, weight decay ($10^{-4}$), and a **multi-step learning rate schedule** (initial LR 0.1, divided by 10 at 50% and 75% of epochs):
- Without Nesterov momentum, gradient updates lack velocity compensation, causing slow traversal of saddle points and ravines in high-dimensional loss landscapes.
- Without the step-decay LR schedule, the optimizer remains stuck oscillating around shallow local minima at $\text{LR}=0.1$. The step decays allow the optimizer to settle into sharp, high-generalization minima, which accounts for the characteristic drop in test error at epochs 150 and 225 in the paper.
