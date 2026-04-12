"""
model.py – EfficientNet-V2-S backbone with a lightweight binary classification head.

Why EfficientNet-V2-S over MobileNetV2?
  • ~25% higher ImageNet top-1 accuracy (83.9% vs 71.8%)
  • Better training efficiency thanks to Fused-MBConv blocks
  • Still fits easily in 12 GB VRAM with batch size ≥ 64
  • Available out-of-the-box in torchvision ≥ 0.14

Alternative: ConvNeXt-Tiny (similar accuracy, slightly larger).  Swap the
`build_model` call to use ConvNeXt if you prefer – no other changes needed.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    ConvNeXt_Tiny_Weights,
)


class GolfSwingClassifier(nn.Module):
    """
    Binary classifier: 1 = frame is between Toe-up and Mid-backswing.

    The head is intentionally small – the pretrained backbone already
    extracts rich spatial features; we just need a robust linear readout
    with enough regularisation to prevent over-fitting on a small dataset.

    Args:
        backbone_name – "efficientnet_v2_s" | "convnext_tiny"
        pretrained    – Load ImageNet weights.
        dropout       – Dropout probability before the final linear layer.
        freeze_until  – Freeze the first N named children of the backbone.
                        Set to 0 to fine-tune everything from the start.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_v2_s",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_until: int = 5,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # ── Backbone ────────────────────────────────────────────────────────
        if backbone_name == "efficientnet_v2_s":
            weights  = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_s(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()   # strip original head

        elif backbone_name == "convnext_tiny":
            weights  = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            in_features = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone_name!r}")

        # Freeze early layers to preserve low-level features
        named = list(backbone.named_children())
        for i, (_, child) in enumerate(named):
            if i < freeze_until:
                for p in child.parameters():
                    p.requires_grad = False

        self.backbone = backbone

        # ── Classification head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),   # raw logit – use BCEWithLogitsLoss
        )

        # Sensible weight initialisation for the head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → logits: (B,)"""
        features = self.backbone(x)          # (B, in_features)
        return self.head(features).squeeze(1)  # (B,)

    def unfreeze_all(self) -> None:
        """Call after initial warm-up to fine-tune the full network."""
        for p in self.parameters():
            p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Convenience factory ───────────────────────────────────────────────────────

def build_model(
    backbone: str = "efficientnet_v2_s",
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_until: int = 5,
) -> GolfSwingClassifier:
    model = GolfSwingClassifier(
        backbone_name=backbone,
        pretrained=pretrained,
        dropout=dropout,
        freeze_until=freeze_until,
    )
    print(
        f"[Model] {backbone} | "
        f"trainable={model.trainable_params():,} / {model.total_params():,} params"
    )
    return model
