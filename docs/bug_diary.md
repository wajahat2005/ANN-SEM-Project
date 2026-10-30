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

## Bug Entry 1: [TODO: Title / Component]
- **Date & Time:** [TODO: e.g. 2026-10-30 14:00]
- **Author / Discoverer:** [TODO: Team Member Name]
- **Environment:** [TODO: e.g. Google Colab T4 GPU / Local Windows CPU]
- **Commit / Script:** [TODO: e.g. experiments/reproduce_cifar.py at commit xxxxx]

### 1. Symptom
```text
[TODO: Paste exact traceback or unexpected behavior here]
```

### 2. Root Cause Analysis
[TODO: Explain why the bug occurred, referencing tensor shapes, memory allocation, PyTorch autograd semantics, or mathematical formulas]

### 3. Resolution / Fix
```python
# [TODO: Paste the before / after code diff or configuration fix here]
```
- **Verification:** [TODO: How was the fix verified? E.g., pytest passed, smoke run succeeded]

---

## Bug Entry 2: [TODO: Title / Component]
- **Date & Time:** [TODO: e.g. 2026-10-30 16:30]
- **Author / Discoverer:** [TODO: Team Member Name]
- **Environment:** [TODO: e.g. Google Colab T4 GPU]
- **Commit / Script:** [TODO: e.g. experiments/run_pneumonia.py]

### 1. Symptom
```text
[TODO: Paste exact traceback or unexpected behavior here]
```

### 2. Root Cause Analysis
[TODO: Explain the mechanism behind the failure]

### 3. Resolution / Fix
```python
# [TODO: Paste the code diff or fix here]
```
- **Verification:** [TODO: Unit test or reproduction check]

---

## Bug Entry 3: [TODO: Title / Component]
- **Date & Time:** [TODO: e.g. 2026-10-31 11:15]
- **Author / Discoverer:** [TODO: Team Member Name]
- **Environment:** [TODO: e.g. Google Colab T4 GPU]
- **Commit / Script:** [TODO: e.g. experiments/run_ablations.py]

### 1. Symptom
```text
[TODO: Paste exact traceback or unexpected behavior here]
```

### 2. Root Cause Analysis
[TODO: Explain why the issue occurred]

### 3. Resolution / Fix
```python
# [TODO: Paste the code diff or fix here]
```
- **Verification:** [TODO: Unit test or reproduction check]

---

## Automated Diary Appender Helper
You can also append new entries directly from the terminal or Colab notebook using:
```bash
python -m src.utils.log_bug --title "OOM on DenseNet-BC-100" --author "Mohammad" --env "Colab T4" --symptom "CUDA out of memory during backward pass" --cause "Batch size 64 exceeded 15GB VRAM due to dense concatenation activations" --fix "Reduced batch size to 32 and enabled torch.cuda.amp.autocast"
```
