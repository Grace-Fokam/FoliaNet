"""The image classifier, with an optional branch for extra inputs.

With tabular_dim=0, the model only uses images. Weather is added to the final
risk score later in folianet/inference.py.

With tabular_dim>0, the model also needs extra input values. The current trainer
and predictor do not pass those values, so keep tabular_dim=0 for now."""

import torch
import torch.nn as nn
from torchvision import models


class FusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        tabular_dim: int = 0,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.backbone_name = backbone

        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.image_features = net.features
            self.pool = net.avgpool
            img_feat_dim = net.classifier[1].in_features  # 1280
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            self.image_features = nn.Sequential(*list(net.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            img_feat_dim = net.fc.in_features  # 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        fusion_in = img_feat_dim
        self.tabular_net = None
        if tabular_dim and tabular_dim > 0:
            self.tabular_net = nn.Sequential(
                nn.Linear(tabular_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(),
            )
            fusion_in += 32

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        x = self.image_features(image)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        if self.tabular_net is not None and tabular is not None:
            x = torch.cat([x, self.tabular_net(tabular)], dim=1)
        return self.head(x)
