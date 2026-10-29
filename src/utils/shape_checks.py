"""
Owner: Mohammad Abdul Rehman (Team & Paper Lead)
Module: shape_checks
Reference: Huang et al. (CVPR 2017)

Purpose:
    Performs runtime shape verification for DenseNet architectures across every layer.
    Generates and saves results/shape_table.md documenting the exact tensor shape progression
    for DenseNet-40 on a (1, 3, 32, 32) input image.
"""

import os
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
from src.models.dense_layer import _DenseLayer
from src.models.dense_block import _DenseBlock
from src.models.transition import _Transition
from src.models.densenet import DenseNet, densenet40_cifar


def generate_densenet40_shape_table(
    save_path: str = "results/shape_table.md",
) -> List[Dict[str, Any]]:
    """
    Traces forward execution through DenseNet-40 layer-by-layer on a 1x3x32x32 input,
    records the exact tensor shapes, prints the table, and saves it to results/shape_table.md.
    """
    model = densenet40_cifar(growth_rate=12, num_classes=10, in_channels=3)
    model.eval()

    x = torch.zeros(1, 3, 32, 32)
    records = []

    # Step 0: Input
    records.append({
        "Stage": "Input",
        "Module": "Raw Image",
        "Input Shape": "(1, 3, 32, 32)",
        "Output Shape": "(1, 3, 32, 32)",
        "Channels In": 3,
        "Channels Out": 3,
        "Spatial Size": "32x32",
    })

    # Step 1: Initial Convolution Stem
    conv0 = model.features.conv0
    x = conv0(x)
    records.append({
        "Stage": "Stem",
        "Module": "conv0 (3x3 Conv, s=1, p=1)",
        "Input Shape": "(1, 3, 32, 32)",
        "Output Shape": f"{tuple(x.shape)}",
        "Channels In": 3,
        "Channels Out": x.size(1),
        "Spatial Size": f"{x.size(2)}x{x.size(3)}",
    })

    # Steps 2-4: DenseBlocks and Transitions
    for name, module in model.features.named_children():
        if name == "conv0":
            continue

        if isinstance(module, _DenseBlock):
            # Step inside the dense block to inspect each layer
            features = [x]
            for layer_name, layer in module.items():
                in_cat = torch.cat(features, dim=1)
                in_shape = tuple(in_cat.shape)
                new_f = layer(features)
                features.append(new_f)
                out_shape = tuple(new_f.shape)

                records.append({
                    "Stage": name,
                    "Module": f"{layer_name} (H_l: BN->ReLU->3x3)",
                    "Input Shape": f"{in_shape}",
                    "Output Shape": f"{out_shape}",
                    "Channels In": in_cat.size(1),
                    "Channels Out": new_f.size(1),
                    "Spatial Size": f"{new_f.size(2)}x{new_f.size(3)}",
                })

            x = torch.cat(features, dim=1)
            records.append({
                "Stage": name,
                "Module": f"Full {name} Concatenation",
                "Input Shape": f"(1, {module.in_channels}, 32, 32)",
                "Output Shape": f"{tuple(x.shape)}",
                "Channels In": module.in_channels,
                "Channels Out": x.size(1),
                "Spatial Size": f"{x.size(2)}x{x.size(3)}",
            })

        elif isinstance(module, _Transition):
            in_shape = tuple(x.shape)
            x = module(x)
            records.append({
                "Stage": name,
                "Module": f"{name} (BN->1x1 Conv->2x2 AvgPool)",
                "Input Shape": f"{in_shape}",
                "Output Shape": f"{tuple(x.shape)}",
                "Channels In": module.in_channels,
                "Channels Out": module.out_channels,
                "Spatial Size": f"{x.size(2)}x{x.size(3)}",
            })

        elif name in ("norm_final", "relu_final"):
            in_shape = tuple(x.shape)
            x = module(x)
            records.append({
                "Stage": "Pre-activation",
                "Module": f"{name}",
                "Input Shape": f"{in_shape}",
                "Output Shape": f"{tuple(x.shape)}",
                "Channels In": x.size(1),
                "Channels Out": x.size(1),
                "Spatial Size": f"{x.size(2)}x{x.size(3)}",
            })

    # Step 5: Global Average Pooling and Head
    in_shape = tuple(x.shape)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    records.append({
        "Stage": "Head",
        "Module": "Global Avg Pool + Flatten",
        "Input Shape": f"{in_shape}",
        "Output Shape": f"{tuple(x.shape)}",
        "Channels In": x.size(1),
        "Channels Out": x.size(1),
        "Spatial Size": "1x1",
    })

    in_shape = tuple(x.shape)
    logits = model.classifier(x)
    records.append({
        "Stage": "Head",
        "Module": "Linear Classifier",
        "Input Shape": f"{in_shape}",
        "Output Shape": f"{tuple(logits.shape)}",
        "Channels In": x.size(1),
        "Channels Out": logits.size(1),
        "Spatial Size": "-",
    })

    # Format Markdown Table
    lines = [
        "# DenseNet-40 Layer-by-Layer Tensor Shape Table",
        "",
        "Input: `1 x 3 x 32 x 32` (CIFAR-10 image)",
        "Configuration: `growth_rate k = 12`, `layers_per_block = (12, 12, 12)`, `bottleneck = False`, `theta = 1.0`",
        "",
        "| Stage | Module Name | Input Shape | Output Shape | C_in | C_out | Spatial Dim |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in records:
        lines.append(
            f"| {r['Stage']} | {r['Module']} | `{r['Input Shape']}` | `{r['Output Shape']}` | {r['Channels In']} | {r['Channels Out']} | {r['Spatial Size']} |"
        )

    content = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Shape Checks] Shape table written to {save_path} ({len(records)} entries)")
    return records


if __name__ == "__main__":
    generate_densenet40_shape_table()
