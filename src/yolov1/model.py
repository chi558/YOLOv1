from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class YOLOv1(nn.Module):
    """YOLOv1-style detector with a ResNet-50 feature extractor."""

    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        out_dim = grid_size * grid_size * (num_classes + num_boxes * 5)

        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
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
        output = self.head(features)
        return output.view(batch_size, self.grid_size, self.grid_size, self.num_classes + self.num_boxes * 5)
