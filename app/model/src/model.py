import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights
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
# 4. Multi-Modal Fusion Model: Vision Embeddings + Text Token Features
# -------------------------------------------------------------------------
class MultiModalInfraPulse(nn.Module):
    """Bi-Encoder Cross-Modal Fusion combining image features and text description embeddings."""
    def __init__(self, num_classes=4, vocab_size=500, text_embed_dim=128, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_cnn = efficientnet_b0(weights=weights)
        self.vision_features = base_cnn.features
        self.vision_pool = nn.AdaptiveAvgPool2d(1)
        self.vision_proj = nn.Linear(1280, 256)

        # Lightweight Text Embedding Stream (BoW / Token Embeddings)
        self.text_embed = nn.EmbeddingBag(vocab_size, text_embed_dim, mode="mean")
        self.text_proj = nn.Linear(text_embed_dim, 256)

        # Cross-Attention Gate
        self.attn_gate = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_x, text_tokens=None, text_offsets=None):
        # 1. Vision stream
        v_feat = self.vision_features(img_x)
        v_feat = self.vision_pool(v_feat).flatten(1)
        v_emb = F.relu(self.vision_proj(v_feat))

        # 2. Text stream (if text provided, else fallback to zero tensor)
        if text_tokens is not None and len(text_tokens) > 0:
            if text_offsets is None:
                text_offsets = torch.tensor([0], device=img_x.device)
            t_raw = self.text_embed(text_tokens, text_offsets)
            t_emb = F.relu(self.text_proj(t_raw))
        else:
            t_emb = torch.zeros_like(v_emb)

        # 3. Dynamic Attention Gating
        combined = torch.cat([v_emb, t_emb], dim=-1)
        gates = self.attn_gate(combined)
        v_weight = gates[:, 0:1]
        t_weight = gates[:, 1:2]

        fused = (v_emb * v_weight) + (t_emb * t_weight)
        logits = self.classifier(fused)
        return logits


# -------------------------------------------------------------------------
# 5. Multi-Class Focal Loss for Imbalanced Defect Datasets
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
