"""
Owner: Aiman Aslam (Experiments Lead)
Module: gradio_demo
Reference: Huang et al. (CVPR 2017) Figure 5 & Medical Diagnosis Demonstration

Gradio Interactive Demonstration:
1. Dataset Selector: CIFAR-10 (10 classes) or PneumoniaMNIST (Normal vs Pneumonia).
2. Input Interface: Upload an image or select from a pre-loaded gallery.
3. Real-Time Prediction: Class probability breakdown and top-1 predicted label.
4. Clinical Decision Threshold Slider (PneumoniaMNIST):
   - Interactive slider [0.01 - 0.99] allowing clinicians to adjust sensitivity.
   - Displays real-time triage outcome at the chosen clinical operating point.
5. Architectural Internals Visualization:
   - Fig. 5 Feature-Reuse Heatmap: Average absolute convolution weights from each layer
     to all its preceding layers in a dense block, demonstrating how deeper layers
     directly reuse earlier feature maps.
   - Grad-CAM Saliency Map: Spatial heatmap overlaid on the input image.
6. Colab & Local Compatibility:
   - Supports share=True for public links on Google Colab.
   - Degrades gracefully with a clear banner if trained weights are not yet present.
"""

import os
import argparse
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import gradio as gr

from src.models.densenet import densenet40_cifar, DenseNet
from src.data.cifar import CIFAR10_CLASSES

# Global cached models
_MODELS: Dict[str, Any] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_demo_models(
    cifar_checkpoint: Optional[str] = "checkpoints/cifar10_densenet40_300ep_best_val.pt",
    pneumonia_checkpoint: Optional[str] = "checkpoints/pneumonia_custom_densenet_64_50ep_best_val.pt",
) -> Dict[str, Tuple[nn.Module, bool]]:
    """Loads CIFAR and Pneumonia models, falling back gracefully to untrained weights if missing."""
    models = {}

    # 1. CIFAR-10 Model (DenseNet-40)
    cifar_model = densenet40_cifar(growth_rate=12, num_classes=10, in_channels=3).to(_DEVICE)
    cifar_trained = False
    if cifar_checkpoint and os.path.exists(cifar_checkpoint):
        try:
            ckpt = torch.load(cifar_checkpoint, map_location=_DEVICE)
            cifar_model.load_state_dict(ckpt["model_state_dict"])
            cifar_trained = True
            print(f"[Demo] Loaded trained CIFAR model from {cifar_checkpoint}")
        except Exception as e:
            print(f"[Demo] Could not load CIFAR checkpoint ({e}). Using initialized weights.")
    cifar_model.eval()
    models["CIFAR-10"] = (cifar_model, cifar_trained)

    # 2. Pneumonia Model (DenseNet-BC-64)
    pneumonia_model = DenseNet(
        growth_rate=12,
        block_config=(6, 6, 6),
        num_init_features=24,
        bottleneck=True,
        theta=0.5,
        in_channels=1,
        num_classes=2,
        small_input=True,
    ).to(_DEVICE)
    pneumonia_trained = False
    if pneumonia_checkpoint and os.path.exists(pneumonia_checkpoint):
        try:
            ckpt = torch.load(pneumonia_checkpoint, map_location=_DEVICE)
            pneumonia_model.load_state_dict(ckpt["model_state_dict"])
            pneumonia_trained = True
            print(f"[Demo] Loaded trained Pneumonia model from {pneumonia_checkpoint}")
        except Exception as e:
            print(f"[Demo] Could not load Pneumonia checkpoint ({e}). Using initialized weights.")
    pneumonia_model.eval()
    models["PneumoniaMNIST"] = (pneumonia_model, pneumonia_trained)

    return models


def generate_feature_reuse_heatmap(model: nn.Module, block_idx: int = 1) -> Image.Image:
    """
    Recreates Figure 5 of Huang et al. (CVPR 2017):
    Computes the average absolute convolution weights between each layer and all its preceding layers
    within denseblock{block_idx}.

    For layer l (1-indexed):
        It receives k0 + (l - 1)*k input channels.
        Channel indices [0 : k0] come from the block input x_0.
        Channel indices [k0 + (j-1)*k : k0 + j*k] come from layer j (for j < l).
        We compute the average absolute weight over all filters and spatial kernel positions.
    """
    block_name = f"denseblock{block_idx}"
    block = None
    for name, m in model.features.named_children():
        if name == block_name:
            block = m
            break

    if block is None:
        # Fallback empty plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, f"{block_name} not found", ha="center", va="center")
        ax.axis("off")
        fig.canvas.draw()
        return Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    num_layers = block.num_layers
    growth_rate = block.growth_rate
    in_channels = block.in_channels

    # Matrix: rows = target layer l (1..num_layers), columns = source layer (0..l-1)
    # Col 0: Block input (x_0). Cols 1..l-1: Outputs of preceding layers.
    matrix = np.zeros((num_layers, num_layers))

    with torch.no_grad():
        for l_idx, (layer_name, layer) in enumerate(block.items()):
            # Weight shape of the first conv in the layer
            # If bottleneck: conv1 is 1x1 conv (4k, in_c, 1, 1)
            # If standard: conv1 is 3x3 conv (k, in_c, 3, 3)
            conv_weight = layer.conv1.weight.abs().cpu().numpy()  # (C_out, C_in, K, K)

            # Average over filters and spatial kernel
            avg_per_channel = np.mean(conv_weight, axis=(0, 2, 3))  # shape: (C_in,)

            # Source 0: block input (in_channels)
            matrix[l_idx, 0] = np.mean(avg_per_channel[:in_channels])

            # Source j: preceding layer j (growth_rate channels)
            for j in range(1, l_idx + 1):
                start_c = in_channels + (j - 1) * growth_rate
                end_c = in_channels + j * growth_rate
                matrix[l_idx, j] = np.mean(avg_per_channel[start_c:end_c])

    # Plot lower-triangular feature-reuse heatmap (Fig. 5 style)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="plasma", origin="upper")
    plt.colorbar(im, ax=ax, label="Mean Absolute Weight")

    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{i}" if i > 0 else "In" for i in range(num_layers)], fontsize=8)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {i+1}" for i in range(num_layers)], fontsize=8)

    ax.set_xlabel("Source Layer (Feature Supplier)", fontsize=10)
    ax.set_ylabel("Target Layer (Feature Consumer)", fontsize=10)
    ax.set_title(f"DenseNet Feature-Reuse Heatmap ({block_name})\n(Huang et al., CVPR 2017 Fig. 5)", fontsize=11)
    plt.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close()
    return Image.fromarray(rgba[:, :, :3])


def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, target_class: int) -> Image.Image:
    """
    Computes Grad-CAM heatmap on the final dense block.
    """
    model.eval()

    # Find last dense block
    target_module = None
    for name, module in reversed(list(model.features.named_children())):
        if "denseblock" in name:
            target_module = module
            break

    if target_module is None:
        target_module = model.features[-1]

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_module.register_forward_hook(forward_hook)
    h2 = target_module.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        logits = model(input_tensor)
        score = logits[0, target_class]
        score.backward()

        act = activations[0].detach()  # (1, C, H, W)
        grad = gradients[0].detach()   # (1, C, H, W)

        # Global average pool gradients along channels
        weights = torch.mean(grad, dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * act, dim=1, keepdim=True)    # (1, 1, H, W)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Apply colormap
        cmap = plt.cm.jet
        heatmap = cmap(cam)[:, :, :3]  # (H, W, 3)

        # Base image
        base_img = input_tensor.squeeze().cpu().numpy()
        if base_img.ndim == 3:
            base_img = base_img.transpose(1, 2, 0)
            base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
        else:
            base_img = np.stack([base_img] * 3, axis=-1)
            base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)

        overlay = 0.5 * base_img + 0.5 * heatmap
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    finally:
        h1.remove()
        h2.remove()


def predict(
    image: Optional[Image.Image],
    dataset_choice: str,
    threshold: float,
    internals_choice: str,
) -> Tuple[str, Dict[str, float], Optional[Image.Image], str]:
    """Main Gradio prediction callback."""
    if image is None:
        return "Please upload an image.", {}, None, ""

    global _MODELS
    if not _MODELS:
        _MODELS = load_demo_models()

    model, is_trained = _MODELS[dataset_choice]

    status_note = ""
    if not is_trained:
        status_note = "⚠️ Running in DEMO MODE (Model has untrained weights). Train via Colab/scripts to save checkpoints."

    if dataset_choice == "CIFAR-10":
        # Preprocess 32x32 RGB
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
        input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        class_names = CIFAR10_CLASSES
        prob_dict = {class_names[i]: float(probs[i]) for i in range(10)}
        top_idx = int(np.argmax(probs))
        summary_text = f"**Predicted Class:** `{class_names[top_idx]}` ({probs[top_idx] * 100:.2f}%)"

    else:
        # PneumoniaMNIST: 64x64 Grayscale
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5719], std=[0.1684]),
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        p_pneumonia = float(probs[1])
        prob_dict = {"Normal": float(probs[0]), "Pneumonia": p_pneumonia}

        diagnosed = "Pneumonia" if p_pneumonia >= threshold else "Normal"
        summary_text = (
            f"**Triage Diagnosis (Threshold = {threshold:.2f}):** `{diagnosed}`\n\n"
            f"- Normal Probability: {probs[0] * 100:.1f}%\n"
            f"- Pneumonia Probability: {probs[1] * 100:.1f}%\n"
            f"- Operating Point: High-Sensitivity Calibrated Rule"
        )

    # Internals visualization
    internals_img = None
    if internals_choice == "Feature-Reuse Heatmap (Fig. 5)":
        internals_img = generate_feature_reuse_heatmap(model, block_idx=1)
    elif internals_choice == "Grad-CAM Saliency Map":
        target_c = 1 if dataset_choice == "PneumoniaMNIST" else int(np.argmax(probs))
        try:
            internals_img = compute_gradcam(model, input_tensor, target_class=target_c)
        except Exception as e:
            print(f"[Grad-CAM Error] {e}")

    return summary_text, prob_dict, internals_img, status_note


def build_app() -> gr.Blocks:
    """Builds the Gradio user interface."""
    with gr.Blocks(title="CS-320 DenseNet Diagnostic Explorer") as demo:
        gr.Markdown(
            """
            # CS-320 Project P03: DenseNet Diagnostic Explorer
            ### *Densely Connected Convolutional Networks (Huang et al., CVPR 2017)*
            **Team:** Mohammad Abdul Rehman (Lead) | Muhammad Wajahat | Aiman Aslam
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                dataset_choice = gr.Radio(
                    choices=["CIFAR-10", "PneumoniaMNIST"],
                    value="PneumoniaMNIST",
                    label="Select Target Dataset",
                )
                threshold_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.99,
                    value=0.50,
                    step=0.01,
                    label="Clinical Sensitivity Threshold (PneumoniaMNIST)",
                    info="Adjust operating threshold for >=95% sensitivity in clinical triage",
                )
                internals_choice = gr.Radio(
                    choices=["Feature-Reuse Heatmap (Fig. 5)", "Grad-CAM Saliency Map", "None"],
                    value="Feature-Reuse Heatmap (Fig. 5)",
                    label="Internal Network Visualization",
                )
                submit_btn = gr.Button("Run Inference & Diagnostic View", variant="primary")

            with gr.Column(scale=1):
                status_banner = gr.Markdown("")
                summary_output = gr.Markdown("### Prediction Results will appear here.")
                prob_output = gr.Label(num_top_classes=5, label="Predicted Probabilities")
                internals_output = gr.Image(type="pil", label="Internals: Feature-Reuse / Saliency Map")

        submit_btn.click(
            fn=predict,
            inputs=[input_image, dataset_choice, threshold_slider, internals_choice],
            outputs=[summary_output, prob_output, internals_output, status_banner],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Colab Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    app = build_app()
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
