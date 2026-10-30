# DenseNet-40 Layer-by-Layer Tensor Shape Table

Input: `1 x 3 x 32 x 32` (CIFAR-10 image)
Configuration: `growth_rate k = 12`, `layers_per_block = (12, 12, 12)`, `bottleneck = False`, `theta = 1.0`

| Stage | Module Name | Input Shape | Output Shape | C_in | C_out | Spatial Dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Input | Raw Image | `(1, 3, 32, 32)` | `(1, 3, 32, 32)` | 3 | 3 | 32x32 |
| Stem | conv0 (3x3 Conv, s=1, p=1) | `(1, 3, 32, 32)` | `(1, 16, 32, 32)` | 3 | 16 | 32x32 |
| denseblock1 | denselayer1 (H_l: BN->ReLU->3x3) | `(1, 16, 32, 32)` | `(1, 12, 32, 32)` | 16 | 12 | 32x32 |
| denseblock1 | denselayer2 (H_l: BN->ReLU->3x3) | `(1, 28, 32, 32)` | `(1, 12, 32, 32)` | 28 | 12 | 32x32 |
| denseblock1 | denselayer3 (H_l: BN->ReLU->3x3) | `(1, 40, 32, 32)` | `(1, 12, 32, 32)` | 40 | 12 | 32x32 |
| denseblock1 | denselayer4 (H_l: BN->ReLU->3x3) | `(1, 52, 32, 32)` | `(1, 12, 32, 32)` | 52 | 12 | 32x32 |
| denseblock1 | denselayer5 (H_l: BN->ReLU->3x3) | `(1, 64, 32, 32)` | `(1, 12, 32, 32)` | 64 | 12 | 32x32 |
| denseblock1 | denselayer6 (H_l: BN->ReLU->3x3) | `(1, 76, 32, 32)` | `(1, 12, 32, 32)` | 76 | 12 | 32x32 |
| denseblock1 | denselayer7 (H_l: BN->ReLU->3x3) | `(1, 88, 32, 32)` | `(1, 12, 32, 32)` | 88 | 12 | 32x32 |
| denseblock1 | denselayer8 (H_l: BN->ReLU->3x3) | `(1, 100, 32, 32)` | `(1, 12, 32, 32)` | 100 | 12 | 32x32 |
| denseblock1 | denselayer9 (H_l: BN->ReLU->3x3) | `(1, 112, 32, 32)` | `(1, 12, 32, 32)` | 112 | 12 | 32x32 |
| denseblock1 | denselayer10 (H_l: BN->ReLU->3x3) | `(1, 124, 32, 32)` | `(1, 12, 32, 32)` | 124 | 12 | 32x32 |
| denseblock1 | denselayer11 (H_l: BN->ReLU->3x3) | `(1, 136, 32, 32)` | `(1, 12, 32, 32)` | 136 | 12 | 32x32 |
| denseblock1 | denselayer12 (H_l: BN->ReLU->3x3) | `(1, 148, 32, 32)` | `(1, 12, 32, 32)` | 148 | 12 | 32x32 |
| denseblock1 | Full denseblock1 Concatenation | `(1, 16, 32, 32)` | `(1, 160, 32, 32)` | 16 | 160 | 32x32 |
| transition1 | transition1 (BN->1x1 Conv->2x2 AvgPool) | `(1, 160, 32, 32)` | `(1, 160, 16, 16)` | 160 | 160 | 16x16 |
| denseblock2 | denselayer1 (H_l: BN->ReLU->3x3) | `(1, 160, 16, 16)` | `(1, 12, 16, 16)` | 160 | 12 | 16x16 |
| denseblock2 | denselayer2 (H_l: BN->ReLU->3x3) | `(1, 172, 16, 16)` | `(1, 12, 16, 16)` | 172 | 12 | 16x16 |
| denseblock2 | denselayer3 (H_l: BN->ReLU->3x3) | `(1, 184, 16, 16)` | `(1, 12, 16, 16)` | 184 | 12 | 16x16 |
| denseblock2 | denselayer4 (H_l: BN->ReLU->3x3) | `(1, 196, 16, 16)` | `(1, 12, 16, 16)` | 196 | 12 | 16x16 |
| denseblock2 | denselayer5 (H_l: BN->ReLU->3x3) | `(1, 208, 16, 16)` | `(1, 12, 16, 16)` | 208 | 12 | 16x16 |
| denseblock2 | denselayer6 (H_l: BN->ReLU->3x3) | `(1, 220, 16, 16)` | `(1, 12, 16, 16)` | 220 | 12 | 16x16 |
| denseblock2 | denselayer7 (H_l: BN->ReLU->3x3) | `(1, 232, 16, 16)` | `(1, 12, 16, 16)` | 232 | 12 | 16x16 |
| denseblock2 | denselayer8 (H_l: BN->ReLU->3x3) | `(1, 244, 16, 16)` | `(1, 12, 16, 16)` | 244 | 12 | 16x16 |
| denseblock2 | denselayer9 (H_l: BN->ReLU->3x3) | `(1, 256, 16, 16)` | `(1, 12, 16, 16)` | 256 | 12 | 16x16 |
| denseblock2 | denselayer10 (H_l: BN->ReLU->3x3) | `(1, 268, 16, 16)` | `(1, 12, 16, 16)` | 268 | 12 | 16x16 |
| denseblock2 | denselayer11 (H_l: BN->ReLU->3x3) | `(1, 280, 16, 16)` | `(1, 12, 16, 16)` | 280 | 12 | 16x16 |
| denseblock2 | denselayer12 (H_l: BN->ReLU->3x3) | `(1, 292, 16, 16)` | `(1, 12, 16, 16)` | 292 | 12 | 16x16 |
| denseblock2 | Full denseblock2 Concatenation | `(1, 160, 32, 32)` | `(1, 304, 16, 16)` | 160 | 304 | 16x16 |
| transition2 | transition2 (BN->1x1 Conv->2x2 AvgPool) | `(1, 304, 16, 16)` | `(1, 304, 8, 8)` | 304 | 304 | 8x8 |
| denseblock3 | denselayer1 (H_l: BN->ReLU->3x3) | `(1, 304, 8, 8)` | `(1, 12, 8, 8)` | 304 | 12 | 8x8 |
| denseblock3 | denselayer2 (H_l: BN->ReLU->3x3) | `(1, 316, 8, 8)` | `(1, 12, 8, 8)` | 316 | 12 | 8x8 |
| denseblock3 | denselayer3 (H_l: BN->ReLU->3x3) | `(1, 328, 8, 8)` | `(1, 12, 8, 8)` | 328 | 12 | 8x8 |
| denseblock3 | denselayer4 (H_l: BN->ReLU->3x3) | `(1, 340, 8, 8)` | `(1, 12, 8, 8)` | 340 | 12 | 8x8 |
| denseblock3 | denselayer5 (H_l: BN->ReLU->3x3) | `(1, 352, 8, 8)` | `(1, 12, 8, 8)` | 352 | 12 | 8x8 |
| denseblock3 | denselayer6 (H_l: BN->ReLU->3x3) | `(1, 364, 8, 8)` | `(1, 12, 8, 8)` | 364 | 12 | 8x8 |
| denseblock3 | denselayer7 (H_l: BN->ReLU->3x3) | `(1, 376, 8, 8)` | `(1, 12, 8, 8)` | 376 | 12 | 8x8 |
| denseblock3 | denselayer8 (H_l: BN->ReLU->3x3) | `(1, 388, 8, 8)` | `(1, 12, 8, 8)` | 388 | 12 | 8x8 |
| denseblock3 | denselayer9 (H_l: BN->ReLU->3x3) | `(1, 400, 8, 8)` | `(1, 12, 8, 8)` | 400 | 12 | 8x8 |
| denseblock3 | denselayer10 (H_l: BN->ReLU->3x3) | `(1, 412, 8, 8)` | `(1, 12, 8, 8)` | 412 | 12 | 8x8 |
| denseblock3 | denselayer11 (H_l: BN->ReLU->3x3) | `(1, 424, 8, 8)` | `(1, 12, 8, 8)` | 424 | 12 | 8x8 |
| denseblock3 | denselayer12 (H_l: BN->ReLU->3x3) | `(1, 436, 8, 8)` | `(1, 12, 8, 8)` | 436 | 12 | 8x8 |
| denseblock3 | Full denseblock3 Concatenation | `(1, 304, 32, 32)` | `(1, 448, 8, 8)` | 304 | 448 | 8x8 |
| Pre-activation | norm_final | `(1, 448, 8, 8)` | `(1, 448, 8, 8)` | 448 | 448 | 8x8 |
| Pre-activation | relu_final | `(1, 448, 8, 8)` | `(1, 448, 8, 8)` | 448 | 448 | 8x8 |
| Head | Global Avg Pool + Flatten | `(1, 448, 8, 8)` | `(1, 448)` | 448 | 448 | 1x1 |
| Head | Linear Classifier | `(1, 448)` | `(1, 10)` | 448 | 10 | - |
