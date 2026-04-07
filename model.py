"""
model.py — Transfer Learning model for DeepFake Detection.

Uses EfficientNetB0 pretrained on ImageNet as the feature extractor,
with a custom classifier head for binary classification (Real vs Fake).
"""

import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DROPOUT_RATE


def build_model(pretrained=True):
    """
    Build the EfficientNetB0 transfer learning model.

    Strategy:
      1. Load EfficientNetB0 with ImageNet weights.
      2. FREEZE all convolutional layers (we keep their learned features).
      3. REPLACE the final classifier head with our own small network.

    This means we only train the classifier head (~6 K parameters),
    which is very fast and works well even with 20 K images.

    Args:
        pretrained (bool): If True, load ImageNet-pretrained weights.

    Returns:
        model (nn.Module): Ready-to-train PyTorch model.
    """

    # Step 1 — Load EfficientNetB0 with pretrained weights
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)

    # Step 2 — Freeze ALL convolutional (feature extractor) layers
    # This prevents their weights from changing during training.
    for param in model.features.parameters():
        param.requires_grad = False

    # Step 3 — Replace the classifier head
    # Original head: Linear(1280, 1000)  (1000 ImageNet classes)
    # Our head:      BN → Linear(1280, 512) → ReLU → Dropout → Linear(512, 256) → ReLU → Dropout → Linear(256, 2)
    in_features = model.classifier[1].in_features  # 1280

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(256, NUM_CLASSES),  # 2 classes: Real, Fake
    )

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] EfficientNetB0 loaded")
    print(f"        Total params:     {total_params:,}")
    print(f"        Trainable params: {trainable_params:,}  "
          f"({trainable_params / total_params * 100:.1f}%)")

    return model


def unfreeze_model(model, num_layers_to_unfreeze=3):
    """
    Optionally unfreeze the last N convolutional blocks for fine-tuning.

    Call this AFTER the initial training if you want to squeeze out a few
    extra percent of accuracy. Use a LOWER learning rate (e.g. 1e-5) when
    fine-tuning to avoid destroying the pretrained features.

    Args:
        model:  The EfficientNetB0 model.
        num_layers_to_unfreeze (int): Number of feature blocks (from the end)
                                       to unfreeze.
    """
    # EfficientNetB0 has 9 feature blocks (indexed 0–8)
    total_blocks = len(model.features)
    unfreeze_from = max(0, total_blocks - num_layers_to_unfreeze)

    for i in range(unfreeze_from, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Unfroze last {num_layers_to_unfreeze} blocks  "
          f"→ {trainable:,} trainable params")


# ──────────────────────────────────────────────
# Quick test: run `python model.py` to see the model summary
# ──────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model(pretrained=True)
    print(m.classifier)

