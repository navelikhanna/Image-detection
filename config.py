
"""
config.py — Central configuration for the DeepFake Detection project.
All hyperparameters, paths, and settings live here so every other
script can simply `from config import *`.
"""

import os

# ──────────────────────────────────────────────
# 📂  PATHS
# ──────────────────────────────────────────────
# Root of the project (wherever THIS file lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset folder — expected structure:
#   dataset/
#   ├── real/   (6000 real images)
#   └── fake/   (6000 AI-generated images)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
REAL_DIR    = os.path.join(DATASET_DIR, "real")
FAKE_DIR    = os.path.join(DATASET_DIR, "fake")

# Where to save the trained model weights
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "best_model.pth")

# Where to save the training curves plot
CURVES_SAVE_PATH = os.path.join(PROJECT_ROOT, "training_curves.png")

# ──────────────────────────────────────────────
# 🖼️  IMAGE SETTINGS
# ──────────────────────────────────────────────
# EfficientNetB0 expects 224×224 input
IMAGE_SIZE = 224

# ImageNet normalization stats (used by all pretrained torchvision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# ⚙️  TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────
BATCH_SIZE    = 32        # images per mini-batch
NUM_EPOCHS    = 85        # maximum total training epochs (Phase 1 + Phase 2)
LEARNING_RATE = 1e-3      # initial learning rate (Phase 1 — head only)
VAL_SPLIT     = 0.2       # 80% train, 20% validation
EARLY_STOP_PATIENCE = 7   # stop if val loss doesn't improve for N epochs
DROPOUT_RATE  = 0.25      # dropout in the classifier head
NUM_WORKERS   = 2         # DataLoader workers (set 0 on Windows if issues)

# ── Phase 2: Fine-tuning (unfreeze backbone) ──
PHASE1_EPOCHS   = 35      # epochs for Phase 1 (head-only training)
FINETUNE_LR     = 1e-5    # much lower LR for fine-tuning backbone
FINETUNE_EPOCHS = 50      # max epochs for Phase 2
UNFREEZE_BLOCKS = 4       # unfreeze last N feature blocks

# ──────────────────────────────────────────────
# 🏷️  CLASS INFO
# ──────────────────────────────────────────────
CLASS_NAMES = ["Real", "Fake"]   # index 0 = Real, 1 = Fake
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────
# 🔥  GRAD-CAM SETTINGS
# ──────────────────────────────────────────────
# How strongly the face-region mask suppresses background in Grad-CAM.
# 0.0 = no masking (original behaviour), 1.0 = only face region shown.
GRADCAM_FACE_WEIGHT = 0.7