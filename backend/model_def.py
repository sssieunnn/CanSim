# cansim_model.py
# -*- coding: utf-8 -*-
"""
CanSimModel (v1):
- Backbone: ResNet18 (ImageNet pretrained), 3-channel input (no conv1 override)
- Tabular: 3-D feature (default: [modality, cancer_type, effectiveness]) -> MLP
- Fusion: concat([img_feat(512), tab_feat(hidden_dim)]) -> MLP -> logits(num_drugs)
- Output: raw logits (use torch.nn.CrossEntropyLoss during training)

Recommended transforms (train/infer must match):
    transforms.Grayscale(num_output_channels=3)
    transforms.Resize((224, 224))
    transforms.ToTensor()
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # (optional use)
import torchvision.models as models


class CanSimModel(nn.Module):
    def __init__(self, tabular_dim: int = 3, hidden_dim: int = 64, num_drugs: int = 4):
        super().__init__()
        # ResNet18 pretrained (new/old torchvision API 모두 호환)
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:
            self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # 512-d feature

        # Tabular encoder
        self.tabular = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(512 + hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_drugs),
        )

    def forward(self, img: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        img: (B, 3, H, W)  # e.g., 224x224 after preprocessing
        tabular: (B, tabular_dim)
        return: (B, num_drugs) logits (apply softmax outside if you need probabilities)
        """
        img_feat = self.backbone(img)            # (B, 512)
        tab_feat = self.tabular(tabular)         # (B, hidden_dim)
        concat = torch.cat([img_feat, tab_feat], dim=1)
        return self.fusion(concat)  # logits


if __name__ == "__main__":
    # Quick shape test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanSimModel().to(device)
    x_img = torch.randn(2, 3, 224, 224, device=device)
    x_tab = torch.randn(2, 3, device=device)
    with torch.no_grad():
        out = model(x_img, x_tab)
    print("Output shape:", tuple(out.shape))  # (2, 4)
