"""
predict.py — Inference script for the DeepFake Detection model.

Usage:
    python predict.py path/to/image.jpg

Loads the trained model (best_model.pth) and classifies a single image
as "Real" or "Fake" with a confidence percentage.
"""

import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import (
    MODEL_SAVE_PATH, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
)
from model import build_model


# Preprocessing — same as validation (no augmentation)
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_trained_model(device):
    """Load the trained model from disk."""
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_image(image_input, model=None, device=None):
    """
    Predict whether an image is Real or Fake.

    Args:
        image_input: Either a file path (str) or a PIL Image object.
        model:       (optional) Pre-loaded model. If None, loads from disk.
        device:      (optional) torch device. If None, auto-detects.

    Returns:
        (label, confidence)  e.g. ("Fake", 94.3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_trained_model(device)

    # Accept file path, PIL Image, or numpy array (webcam)
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    # Preprocess
    img_tensor = inference_transform(image).unsqueeze(0).to(device)  # add batch dim

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    label = CLASS_NAMES[predicted_class.item()]
    conf  = confidence.item() * 100  # percentage

    return label, conf


# ──────────────────────────────────────────────
# CLI usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python predict.py <image_path>")
        print("Example: python predict.py dataset/real/image001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    label, confidence = predict_image(image_path)

    print(f"\n{'=' * 40}")
    print(f"  Image:      {image_path}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'=' * 40}")
