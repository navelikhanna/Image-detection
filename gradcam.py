

"""
gradcam.py — Grad-CAM (Gradient-weighted Class Activation Mapping)
              for EfficientNetB0-based DeepFake Detection.

Generates heatmap overlays showing which image regions the model
focuses on when making a Real/Fake prediction.

Includes face-aware masking: detected face regions are emphasised
so the heatmap concentrates on the person rather than the background.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks",
           ICCV 2017 — https://arxiv.org/abs/1610.02391
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, GRADCAM_FACE_WEIGHT


# Same preprocessing used for inference
_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class GradCAM:
    """
    Grad-CAM implementation for EfficientNetB0.

    How it works:
      1. Forward pass — capture the feature maps from the target layer.
      2. Backward pass — compute gradients of the predicted class score
         w.r.t. those feature maps.
      3. Weight each feature map channel by its mean gradient (global avg pooling).
      4. Sum the weighted maps and apply ReLU → heatmap.
      5. Overlay the heatmap on the original image.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: EfficientNetB0 model (from model.py).
            target_layer: The convolutional layer to visualise.
                          Default: last feature block (model.features[-1]).
        """
        self.model = model
        self.target_layer = target_layer or model.features[-1]

        # Storage for hooks
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook — store the feature maps."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook — store the gradients."""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Run forward + backward pass and produce a normalised heatmap.

        Args:
            input_tensor: Preprocessed image tensor [1, 3, 224, 224].
            class_idx:    Class index to explain. If None, uses the
                          model's predicted class.

        Returns:
            heatmap: numpy array (H, W) in [0, 1] range.
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero all gradients, then backprop the target class score
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()

        # Pool gradients across spatial dims → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)  # only keep positive influence

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() != 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


# ──────────────────────────────────────────────────────
#  FACE-AWARE MASKING
# ──────────────────────────────────────────────────────

def _detect_faces(image_np):
    """
    Detect faces in an RGB numpy image using OpenCV Haar cascades.

    Returns:
        List of (x, y, w, h) bounding boxes, or empty list if none found.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Try multiple cascade classifiers for robustness
    cascade_names = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
        cv2.data.haarcascades + "haarcascade_profileface.xml",
    ]

    for cascade_path in cascade_names:
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )
        if len(faces) > 0:
            return faces.tolist()

    return []


def _create_face_mask(h, w, face_boxes, sigma_scale=0.6):
    """
    Create a soft Gaussian attention mask from detected face bounding boxes.

    Each face box produces a Gaussian blob; overlapping blobs are combined
    via element-wise max so the mask stays in [0, 1].

    Args:
        h, w:         Output mask dimensions.
        face_boxes:   List of (x, y, bw, bh) bounding boxes.
        sigma_scale:  Gaussian spread relative to face size (larger = softer).

    Returns:
        mask: (h, w) numpy array in [0, 1].
    """
    mask = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]

    for (fx, fy, fw, fh) in face_boxes:
        # Centre of the face box
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0

        # Sigma proportional to face size (use the larger dimension)
        sigma = max(fw, fh) * sigma_scale

        # 2D Gaussian centred on the face
        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        mask = np.maximum(mask, gaussian)

    # Normalise to [0, 1]
    if mask.max() > 0:
        mask = mask / mask.max()

    return mask


def _create_center_fallback_mask(h, w):
    """
    Fallback mask when no face is detected: a Gaussian centred on the image.
    This works because portrait images typically have the person centred.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    sigma = max(h, w) * 0.4
    mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    mask = mask / mask.max()
    return mask


def _apply_face_mask(heatmap, image_np, face_weight=GRADCAM_FACE_WEIGHT):
    """
    Modulate the GradCAM heatmap with a face-aware attention mask.

    Args:
        heatmap:      (H, W) raw heatmap in [0, 1].
        image_np:     Original RGB image as numpy array.
        face_weight:  Blending strength (0 = no masking, 1 = full masking).

    Returns:
        masked_heatmap: (H, W) face-biased heatmap in [0, 1].
    """
    if face_weight <= 0:
        return heatmap

    h, w = image_np.shape[:2]

    # Detect faces in the original image
    faces = _detect_faces(image_np)

    if faces:
        mask = _create_face_mask(h, w, faces)
    else:
        # No face found — use centre-weighted fallback
        mask = _create_center_fallback_mask(h, w)

    # Resize mask to match heatmap dimensions (heatmap may be 7×7 from model)
    hm_h, hm_w = heatmap.shape
    if (hm_h, hm_w) != (h, w):
        mask = cv2.resize(mask, (hm_w, hm_h))

    # Blend: masked = heatmap * ((1 - weight) + weight * mask)
    # This keeps face regions at full intensity and dims background.
    blend = (1.0 - face_weight) + face_weight * mask
    masked = heatmap * blend

    # Re-normalise to [0, 1]
    if masked.max() > 0:
        masked = (masked - masked.min()) / (masked.max() - masked.min())

    return masked


# ──────────────────────────────────────────────────────
#  HIGH-LEVEL OVERLAY FUNCTION
# ──────────────────────────────────────────────────────

def generate_gradcam_overlay(pil_image, model, device, alpha=0.5):
    """
    High-level function: take a PIL image, produce a Grad-CAM overlay.

    Args:
        pil_image: Input PIL Image (any size).
        model:     Loaded EfficientNetB0 model.
        device:    torch device (cpu / cuda).
        alpha:     Heatmap transparency (0 = original only, 1 = heatmap only).

    Returns:
        overlay_pil: PIL Image with heatmap overlay.
        label:       Predicted class name ("Real" / "Fake").
        confidence:  Confidence percentage (float).
    """
    from config import CLASS_NAMES

    # Preprocess
    input_tensor = _preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)

    # Enable gradients for this forward pass
    input_tensor.requires_grad_(True)

    # Generate heatmap
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_heatmap(input_tensor)

    # Get the prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        label = CLASS_NAMES[predicted.item()]
        conf = confidence.item() * 100

    # Resize heatmap to match original image
    original_np = np.array(pil_image.convert("RGB"))
    h, w = original_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # ── Face-aware masking ──
    heatmap_resized = _apply_face_mask(heatmap_resized, original_np)

    # Apply colourmap (blue→green→red)
    heatmap_coloured = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

    # Blend original image with heatmap
    overlay = np.uint8(original_np * (1 - alpha) + heatmap_coloured * alpha)

    # Add label text on the overlay
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    text = f"{label} ({conf:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w, h) / 500)
    thickness = max(1, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Background rectangle for text
    cv2.rectangle(overlay_bgr, (5, 5), (tw + 15, th + 15), (0, 0, 0), -1)
    cv2.putText(overlay_bgr, text, (10, th + 10), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)

    return overlay_pil, label, conf