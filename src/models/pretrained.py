"""
Owner: Aiman Aslam (Experiments Lead)
Module: pretrained
Reference: Huang et al. (CVPR 2017) & Torchvision Pretrained Models

Purpose:
    Provides a factory helper to load torchvision's pretrained DenseNet-121 (ImageNet weights)
    and adapt it for transfer learning on PneumoniaMNIST.

Features:
    1. Grayscale to 3-channel adaptation:
       - Either by prepending a 1x1 conv / channel replication layer, or adapting conv0.
    2. Head replacement:
       - Replaces 1000-class classifier with 2-class (or single-logit) linear head.
    3. Input resolution handling:
       - Optional internal upsampling to 224x224 when resize_to_224=True.
    4. Two-stage fine-tuning schedule:
       - freeze_features=True freezes feature extractor for initial head warmup.
       - unfreeze_features() unfreezes all parameters for joint end-to-end training.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class PretrainedDenseNet121(nn.Module):
    """
    Wrapper around torchvision.models.densenet121 with options for:
    - 1-channel grayscale input (adapted conv0 or 3-channel replication)
    - Optional bilinear resize to 224x224
    - Configurable classification head (default 2 classes)
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        resize_to_224: bool = False,
        freeze_features: bool = False,
    ) -> None:
        super(PretrainedDenseNet121, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.resize_to_224 = resize_to_224

        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = tv_models.densenet121(weights=weights)

        # Handle 1-channel input by adapting the first convolution layer
        if in_channels == 1:
            orig_conv = base_model.features.conv0
            # Initialize 1-channel conv by summing weights across RGB channels to preserve luminance
            new_conv = nn.Conv2d(
                1,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(orig_conv.weight.sum(dim=1, keepdim=True))
            base_model.features.conv0 = new_conv
        elif in_channels != 3:
            orig_conv = base_model.features.conv0
            base_model.features.conv0 = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )

        # Replace classification head
        in_features = base_model.classifier.in_features
        base_model.classifier = nn.Linear(in_features, num_classes)

        self.features = base_model.features
        self.classifier = base_model.classifier

        if freeze_features:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freezes all feature extractor weights for head warmup."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all feature extractor weights for end-to-end fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize_to_224 and (x.shape[-2] != 224 or x.shape[-1] != 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits


def get_pretrained_densenet121(
    num_classes: int = 2,
    in_channels: int = 1,
    pretrained: bool = True,
    resize_to_224: bool = False,
    freeze_features: bool = False,
) -> PretrainedDenseNet121:
    """Factory function for adapted DenseNet-121."""
    return PretrainedDenseNet121(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        resize_to_224=resize_to_224,
        freeze_features=freeze_features,
    )
