# DenseNet (CVPR 2017) — 1-Page Architecture Reference
**Course:** CS-320: Artificial Neural Networks & Deep Learning · Fall 2026 · NUST NBC  
**Paper:** *Densely Connected Convolutional Networks* (Huang, Liu, van der Maaten, Weinberger)  
**Team P03:** Mohammad Abdul Rehman (Lead) · Muhammad Wajahat (Data) · Aiman Aslam (Experiments)

---

### 1. Key Architectural Idea & Core Equations
Standard ResNet adds transformations to identity skip connections: $\mathbf{x}_l = H_l(\mathbf{x}_{l-1}) + \mathbf{x}_{l-1}$, which superimposes representations and can dilute early features. **DenseNet replaces summation with channel-wise concatenation**:
$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{l-1}])$$
where each composite layer $H_l = \text{BN} \rightarrow \text{ReLU} \rightarrow 3\times 3\text{ Conv}$ outputs a narrow set of $k$ channels (**Growth Rate**).  
Input channels to layer $l$ within a dense block: $C_{\text{in}}(l) = k_0 + (l - 1) \times k$.

---

### 2. DenseNet-40 ($k=12$) Tensor Flow & Layer-by-Layer Dimensions
| Stage / Layer | Layer Specification ($H_l$) | Input Dim $(C \times H \times W)$ | Output Dim $(C \times H \times W)$ | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Stem Conv** | $3 \times 3$ Conv, stride 1, padding 1 | $3 \times 32 \times 32$ | $16 \times 32 \times 32$ | $3 \times 16 \times 9 = 432$ |
| **Dense Block 1** | 12 dense layers ($H_l: \text{BN}\to\text{ReLU}\to 3\times 3\text{ Conv}, k=12$) | $16 \times 32 \times 32$ | $160 \times 32 \times 32$ | $106,656$ |
| **Transition 1** | $\text{BN}\to\text{ReLU}\to 1\times 1\text{ Conv}\to 2\times 2\text{ AvgPool } (s=2)$ | $160 \times 32 \times 32$ | $160 \times 16 \times 16$ | $160\times 2 + 160\times 160 = 25,920$ |
| **Dense Block 2** | 12 dense layers ($H_l: \text{BN}\to\text{ReLU}\to 3\times 3\text{ Conv}, k=12$) | $160 \times 16 \times 16$ | $304 \times 16 \times 16$ | $292,416$ |
| **Transition 2** | $\text{BN}\to\text{ReLU}\to 1\times 1\text{ Conv}\to 2\times 2\text{ AvgPool } (s=2)$ | $304 \times 16 \times 16$ | $304 \times 8 \times 8$ | $304\times 2 + 304\times 304 = 92,720$ |
| **Dense Block 3** | 12 dense layers ($H_l: \text{BN}\to\text{ReLU}\to 3\times 3\text{ Conv}, k=12$) | $304 \times 8 \times 8$ | $448 \times 8 \times 8$ | $478,176$ |
| **Head** | Final $\text{BN}\to\text{ReLU}\to\text{GlobalAvgPool}\to\text{Linear}(448 \to 10)$ | $448 \times 8 \times 8$ | $10$ logits | $448\times 2 + (448\times 10 + 10) = 5,386$ |
| **Total** | **DenseNet-40 (CIFAR-10 configuration)** | **$3 \times 32 \times 32$** | **$10$ Classes** | **$1,019,722$ (~1.02M)** |

---

### 3. Analytical Hand Parameter Count Proof (Block 1, Layer 1)
For $H_1$ with $C_{\text{in}} = 16$ and $k = 12$ (no bottleneck, pre-activation):
* **$\text{BatchNorm}_1$:** $\gamma, \beta \implies 2 \times C_{\text{in}} = 2 \times 16 = \mathbf{32}$ parameters
* **$3 \times 3\text{ Conv}$ (no bias):** $C_{\text{in}} \times k \times 3 \times 3 = 16 \times 12 \times 9 = \mathbf{1,728}$ parameters
* **Layer 1 Total:** $32 + 1,728 = \mathbf{1,760}$ parameters $\equiv \sum p.\text{numel()}$ **(Exact Match)**

---

### 4. Key Architectural Hyperparameters
* **Growth Rate ($k=12$):** Channels output by each layer. Governs capacity and memory scaling.
* **Bottleneck (DenseNet-B):** Injects $\text{BN}\to 1\times 1\text{ Conv}$ compressing $C_{\text{in}} \to 4k$ before $3\times 3$ conv. Suppresses quadratic parameter explosion in deep models.
* **Transition Compression ($\theta=0.5$ in DenseNet-C):** Halves channels at transitions ($\lfloor \theta \cdot m \rfloor$). Discards redundant feature maps.
* **Dropout ($p=0.2$):** Injected after $3\times 3$ conv only when trained without data augmentation to curb co-adaptation.

---

### 5. Strengths & Practical Trade-offs
* **Strengths:** Direct gradient highways eliminate vanishing gradients; explicit feature reuse yields superior parameter efficiency over ResNets; implicit deep supervision accelerates convergence.
* **Trade-offs:** Concatenating tensors produces large intermediate memory allocations; higher GPU memory-access cost (MAC) can yield lower wall-clock throughput without memory-efficient concatenation.

---

### 6. Viva Self-Check Questions (Quick Revision)
1. **Q: Why are transition layers needed instead of one giant dense block?**  
   *A:* Concatenation strictly requires identical spatial resolution $(H \times W)$. Transition layers allow necessary spatial downsampling (pooling) and channel compression between blocks.
2. **Q: In DenseNet-40, how many input channels enter Layer 7 of Dense Block 2?**  
   *A:* $k_0 = 160$ (from Transition 1). $C_{\text{in}}(7) = 160 + (7 - 1) \times 12 = 160 + 72 = \mathbf{232}$ channels.
3. **Q: Why do convolutions in DenseNet omit bias parameters ($\text{bias}=\text{False}$)?**  
   *A:* Convolution is immediately preceded or followed by Batch Normalization, whose $\beta$ parameter absorbs any constant additive bias, rendering convolutional bias redundant.
