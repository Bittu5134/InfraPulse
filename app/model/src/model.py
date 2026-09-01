import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights,
    resnet18, ResNet18_Weights
)

CLASS_NAMES = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]

CATEGORY_MAP = {
    "spalling": "Structural",
    "stagnant_water": "Functional",
    "cracked_tiles": "Performance",
    "paint_peeling": "Performance",
}

# -------------------------------------------------------------------------
# 1. Baseline: EfficientNet-B0 Architecture
# -------------------------------------------------------------------------
class InfraPulseNet(nn.Module):
    """EfficientNet-B0 Backbone with 2-layer Dropout Classifier Head."""
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks=2):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        blocks = list(self.backbone.features.children())
        num_blocks = max(1, min(num_blocks, len(blocks)))
        for block in blocks[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def get_gradcam_target_layer(self):
        return self.backbone.features[-1]


# -------------------------------------------------------------------------
# 2. ConvNeXt-Tiny: Modern Pure CNN with 7x7 Depthwise Kernels & LayerNorm
# -------------------------------------------------------------------------
class ConvNeXtInfraPulse(nn.Module):
    """ConvNeXt-Tiny Backbone for high-frequency fracture & texture analysis."""
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features  # 768

        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],  # Global AvgPool
            self.backbone.classifier[1],  # LayerNorm
            nn.Dropout(p=0.30),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self, num_stages=1):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        stages = list(self.backbone.features.children())
        for stage in stages[-num_stages:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def get_gradcam_target_layer(self):
        return self.backbone.features[-1]


# -------------------------------------------------------------------------
# 3. Swin Transformer (Swin-T): Shifted Window Self-Attention
# -------------------------------------------------------------------------
class SwinInfraPulse(nn.Module):
    """Swin-T Transformer for long-range context (water reflections & wet surfaces)."""
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = swin_t(weights=weights)
        in_features = self.backbone.head.in_features  # 768

        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.backbone.head.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self, num_stages=1):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        stages = list(self.backbone.features.children())
        for stage in stages[-num_stages:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in self.backbone.head.parameters():
            p.requires_grad = True

    def get_gradcam_target_layer(self):
        # Last norm layer before pooling
        return self.backbone.norm



# -------------------------------------------------------------------------
# 5. Multi-Task Learning (MTL) Dual-Branch Architecture
#    Single Shared Backbone + Branch 1 (Classification) + Branch 2 (Area Extractor)
# -------------------------------------------------------------------------
class MultiTaskInfraPulse(nn.Module):
    """
    Multi-Task Learning (MTL) Core Vision Model:
    - Backbone: Single shared generic feature extractor (ImageNet-1K pretrained).
    - Branch 1 (Classification Head): Identifies visible physical defect (Spalling, Water, Tiles, Paint).
    - Branch 2 (Area Extractor Head): Derives spatial defect density mask and scalar extent ratio (%) directly from shared feature maps.
    - Pure Computer Vision: Strictly zero text dependence, unified confidence calibration, low latency.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = resnet18(weights=weights)

        # Shared Generic Feature Extractor
        self.shared_backbone = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4  # Outputs: [B, 512, 7, 7]
        )

        # Branch 1: Defect Classification Head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes)
        )

        # Branch 2: Visible Defect Area & Spatial Extent Extractor
        # Generates a 2D spatial defect density mask and computes extent area ratio
        self.extent_conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Activation intensity per 7x7 spatial patch
        )
        self.extent_fc = nn.Sequential(
            nn.Linear(7 * 7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Relative extent ratio in [0, 1]
        )

    def forward(self, x, return_all=False):
        features = self.shared_backbone(x)  # [B, 512, 7, 7]

        # Branch 1: Classification Logits
        pooled = self.cls_pool(features).flatten(1)
        cls_logits = self.cls_head(pooled)

        # Branch 2: Spatial Area Extractor
        spatial_mask = self.extent_conv(features)  # [B, 1, 7, 7]
        flat_mask = spatial_mask.flatten(1)
        extent_ratio = self.extent_fc(flat_mask) * 100.0  # Scale to [0, 100%]

        if return_all:
            return {
                "logits": cls_logits,
                "extent_ratio": extent_ratio,
                "spatial_mask": spatial_mask
            }
        return cls_logits

    def freeze_backbone(self):
        for p in self.shared_backbone.parameters():
            p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = True
        for p in self.extent_conv.parameters():
            p.requires_grad = True
        for p in self.extent_fc.parameters():
            p.requires_grad = True

    def get_gradcam_target_layer(self):
        return self.shared_backbone[-1][-1].conv2


# -------------------------------------------------------------------------
# 6. Multi-Class Focal Loss for Imbalanced Defect Datasets
# -------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss (Lin et al.) to focus learning on hard / minority classes."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor of class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
