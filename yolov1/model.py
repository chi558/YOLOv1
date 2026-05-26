from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50


BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
}


class YOLOv1(nn.Module):
    """YOLOv1-style detector with a configurable ResNet feature extractor."""

    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        use_conv_head: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {sorted(BACKBONES)}")
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        out_dim = grid_size * grid_size * (num_classes + num_boxes * 5)

        builder, default_weights, feature_channels = BACKBONES[backbone]
        weights = default_weights if pretrained_backbone else None
        backbone_model = builder(weights=weights)
        self.features = nn.Sequential(*list(backbone_model.children())[:-2])
        self.use_conv_head = use_conv_head
        self.detector = nn.Sequential(
            nn.Conv2d(feature_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if use_conv_head:
            self.head = nn.Conv2d(1024, num_classes + num_boxes * 5, kernel_size=1)
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((grid_size, grid_size)),
                nn.Flatten(),
                nn.Linear(1024 * grid_size * grid_size, 4096),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, out_dim),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        features = self.features(images)
        features = self.detector(features)
        output = self.head(features)
        if self.use_conv_head:
            if output.shape[-2:] != (self.grid_size, self.grid_size):
                output = nn.functional.adaptive_avg_pool2d(output, (self.grid_size, self.grid_size))
            return output.permute(0, 2, 3, 1).contiguous()
        return output.view(batch_size, self.grid_size, self.grid_size, self.num_classes + self.num_boxes * 5)
