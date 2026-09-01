import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

CLASS_NAMES = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]

CATEGORY_MAP = {
    "spalling": "Structural",
    "stagnant_water": "Functional",
    "cracked_tiles": "Performance",
    "paint_peeling": "Performance",
}

class InfraPulseNet(nn.Module):
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
        # Freeze everything first
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
        # Last feature block before pooling/classifier
        return self.backbone.features[-1]
