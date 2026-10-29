# License and Dataset Attributions

## 1. Code License
This project is open-source and released under the **MIT License**.

```
MIT License

Copyright (c) 2026 CS-320 Project P03 Team
(Mohammad Abdul Rehman, Muhammad Wajahat, Aiman Aslam)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Dataset Attributions & Terms of Use

### (a) CIFAR-10
- **Authors:** Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton (2009).
- **Source:** *Learning Multiple Layers of Features from Tiny Images*, Technical Report, University of Toronto.
- **Description:** 60,000 $32 \times 32$ color images in 10 classes (6,000 images per class). 
- **Reproduction Split:** 45,000 train / 5,000 validation / 10,000 test.
- **Terms:** Freely available for educational and non-commercial research purposes.

### (b) MedMNIST v3 (PneumoniaMNIST)
- **Authors:** Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni (2021, 2023).
- **Primary Citations:**
  - *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 2023.
  - *MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis*, IEEE ISBI 2021.
- **Underlying Clinical Source:** Kermany et al. (Cell 2018), "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" (Pediatric chest X-rays from Guangzhou Women and Children's Medical Center).
- **Dataset License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Usage in this Project:** Binary pediatric pneumonia classification (Normal vs Pneumonia) at resolution $64 \times 64$ (with ablations at $28 \times 28$ and $128 \times 128$).
