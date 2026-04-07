"""
app.py — Gradio web UI for DeepFake Detection.

Features:
  • Single Image Detection with Grad-CAM heatmap (upload OR webcam)
  • Batch Image Analysis with summary statistics
  • Prediction History with session logging
  • Model Info dashboard with training curves and hyperparameters

Usage:
    python app.py
"""

import os
import gradio as gr
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from predict import predict_image, load_trained_model
from gradcam import generate_gradcam_overlay
from config import (
    CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, VAL_SPLIT, EARLY_STOP_PATIENCE, DROPOUT_RATE, NUM_WORKERS,
    CURVES_SAVE_PATH, MODEL_SAVE_PATH,
)


# ── Load model once at startup ────────────────────────
print("[*] Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model(device)
print(f"[OK] Model loaded on {device}!\n")


# ── Session state ─────────────────────────────────────
prediction_history = []  # list of dicts for the history table


# ══════════════════════════════════════════════════════
#  TAB 1 — SINGLE IMAGE DETECTION
# ══════════════════════════════════════════════════════

def _ensure_pil_image(image):
    """
    Convert any Gradio image input to a PIL Image.
    Handles: PIL Image, numpy array, file path string, and dict (Gradio 5+/6+).
    Returns None if conversion fails.
    """
    if image is None:
        return None

    # Gradio 5+/6+ may pass a dict with 'path' or 'url' key
    if isinstance(image, dict):
        path = image.get("path") or image.get("url") or image.get("name")
        if path:
            return Image.open(path).convert("RGB")
        return None

    # File path string
    if isinstance(image, str):
        return Image.open(image).convert("RGB")

    # Numpy array (webcam frames)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")

    # PIL Image
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # Last resort: try to use it directly
    print(f"[WARN] Unexpected image type: {type(image)}")
    return None


def classify_single(image):
    """
    Classify a single image (from upload or webcam).
    Returns prediction label dict, Grad-CAM overlay, and analysis text.
    """
    print(f"[DEBUG] classify_single called with type: {type(image)}")
    if isinstance(image, dict):
        print(f"[DEBUG] dict keys: {image.keys()}")

    pil_image = _ensure_pil_image(image)
    if pil_image is None:
        return {"Error": 1.0}, None, "⚠️ No image provided. Please capture a photo from the webcam or upload an image first."

    try:
        # Get prediction
        label, confidence = predict_image(pil_image, model=model, device=device)

        # Generate Grad-CAM overlay
        gradcam_img, _, _ = generate_gradcam_overlay(
            pil_image, model, device, alpha=0.5
        )

        # Build label dict for Gradio
        conf_decimal = confidence / 100.0
        if label == "Real":
            label_dict = {"Real ✅": conf_decimal, "Fake ❌": 1 - conf_decimal}
        else:
            label_dict = {"Fake ❌": conf_decimal, "Real ✅": 1 - conf_decimal}

        # Analysis text
        analysis = (
            f"### 🔍 Analysis Result\n\n"
            f"**Prediction:** {'✅ Real' if label == 'Real' else '❌ Fake'}\n\n"
            f"**Confidence:** {confidence:.1f}%\n\n"
            f"**Interpretation:** "
        )
        if confidence > 90:
            analysis += "The model is **highly confident** in this prediction."
        elif confidence > 70:
            analysis += "The model is **moderately confident**. Consider examining the Grad-CAM heatmap for regions of interest."
        else:
            analysis += "The model shows **low confidence**. The image may contain ambiguous features. Manual inspection recommended."

        # Log to history
        prediction_history.append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Prediction": label,
            "Confidence (%)": f"{confidence:.1f}",
            "Source": "Single Image",
        })

        return label_dict, gradcam_img, analysis

    except Exception as e:
        import traceback
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return {"Error": 1.0}, None, f"❌ Error: {str(e)}"


# ══════════════════════════════════════════════════════
#  TAB 2 — BATCH IMAGE ANALYSIS
# ══════════════════════════════════════════════════════

def classify_batch(files):
    """
    Classify multiple images at once. Returns a results table + summary.
    """
    if files is None or len(files) == 0:
        return None, "⚠️ No images uploaded."

    results = []
    real_count = 0
    fake_count = 0
    total_conf = 0.0

    for file_path in files:
        try:
            img = Image.open(file_path).convert("RGB")
            label, confidence = predict_image(img, model=model, device=device)

            filename = os.path.basename(file_path) if isinstance(file_path, str) else f"Image"
            results.append([filename, label, f"{confidence:.1f}%"])

            if label == "Real":
                real_count += 1
            else:
                fake_count += 1
            total_conf += confidence

            # Log to history
            prediction_history.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Prediction": label,
                "Confidence (%)": f"{confidence:.1f}",
                "Source": "Batch",
            })

        except Exception as e:
            filename = os.path.basename(file_path) if isinstance(file_path, str) else "Unknown"
            results.append([filename, "Error", str(e)])

    total = len(results)
    avg_conf = total_conf / max(total, 1)

    summary = (
        f"### 📊 Batch Analysis Summary\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total Images | {total} |\n"
        f"| Real Images | {real_count} ({real_count/total*100:.0f}%) |\n"
        f"| Fake Images | {fake_count} ({fake_count/total*100:.0f}%) |\n"
        f"| Avg. Confidence | {avg_conf:.1f}% |\n"
    )

    return results, summary


# ══════════════════════════════════════════════════════
#  TAB 3 — PREDICTION HISTORY
# ══════════════════════════════════════════════════════

def get_history():
    """Return the current prediction history as a list of lists."""
    if not prediction_history:
        return [["—", "—", "—", "—"]]
    return [[h["Timestamp"], h["Prediction"], h["Confidence (%)"], h["Source"]]
            for h in prediction_history]


def clear_history():
    """Clear all prediction history."""
    prediction_history.clear()
    return [["—", "—", "—", "—"]], "✅ History cleared."


# ══════════════════════════════════════════════════════
#  TAB 4 — MODEL INFO
# ══════════════════════════════════════════════════════

def get_model_info():
    """Return model architecture and training config as markdown."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024) if os.path.exists(MODEL_SAVE_PATH) else 0

    info = f"""### 🏗️ Model Architecture

| Property | Value |
|----------|-------|
| Base Model | EfficientNetB0 (ImageNet pretrained) |
| Strategy | Transfer Learning (frozen backbone + custom head) |
| Input Size | {IMAGE_SIZE} × {IMAGE_SIZE} px |
| Output Classes | {', '.join(CLASS_NAMES)} |
| Total Parameters | {total_params:,} |
| Trainable Parameters | {trainable_params:,} ({trainable_params/total_params*100:.1f}%) |
| Frozen Parameters | {frozen_params:,} |
| Model File Size | {model_size_mb:.1f} MB |
| Device | {device} |

### 📐 Classifier Head

```
BatchNorm1d(1280) → Dropout({DROPOUT_RATE}) → Linear(1280, 512) → ReLU → Dropout(0.3) → Linear(512, 2)
```

### ⚙️ Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | {BATCH_SIZE} |
| Max Epochs | {NUM_EPOCHS} |
| Learning Rate | {LEARNING_RATE} |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Validation Split | {VAL_SPLIT*100:.0f}% |
| Early Stopping Patience | {EARLY_STOP_PATIENCE} epochs |
| Dropout Rate | {DROPOUT_RATE} |
| Data Workers | {NUM_WORKERS} |

### 📝 Data Augmentation (Training)

- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation, hue)
- Resize to {IMAGE_SIZE}×{IMAGE_SIZE}
- ImageNet Normalisation
"""
    return info


# ══════════════════════════════════════════════════════
#  BUILD THE GRADIO UI
# ══════════════════════════════════════════════════════

custom_css = """
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.gradio-container {
    max-width: 1200px !important;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="DeepFake Image Detector",
) as demo:

    # ── Header ────────────────────────────────────────
    gr.Markdown(
        """
        # 🔎 DeepFake Image Detector
        **AI-powered detection of real vs. AI-generated images using EfficientNetB0 with Grad-CAM explainability.**

        Upload an image or use your webcam to check if it's real or AI-generated.
        The model highlights suspicious regions using **Grad-CAM heatmaps** for full transparency.
        """
    )

    with gr.Tabs():

        # ━━━━ TAB 1: SINGLE IMAGE ━━━━━━━━━━━━━━━━━━━
        with gr.TabItem("🖼️ Single Image Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="pil",
                        label="Upload Image or Use Webcam",
                        sources=["upload", "webcam"],
                        height=350,
                        streaming=False,
                    )
                    detect_btn = gr.Button(
                        "🔍 Analyse Image",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    output_label = gr.Label(
                        num_top_classes=2,
                        label="Prediction",
                    )
                    output_gradcam = gr.Image(
                        label="Grad-CAM Heatmap (Model Focus Areas)",
                        height=300,
                    )

            analysis_md = gr.Markdown(label="Analysis")

            detect_btn.click(
                fn=classify_single,
                inputs=[input_image],
                outputs=[output_label, output_gradcam, analysis_md],
            )

            gr.Examples(
                examples=[],  # add sample image paths here if desired
                inputs=input_image,
                label="Try these examples",
            ) if False else None  # placeholder — no sample images bundled

        # ━━━━ TAB 2: BATCH ANALYSIS ━━━━━━━━━━━━━━━━━
        with gr.TabItem("📁 Batch Analysis"):
            gr.Markdown(
                """
                ### Batch Image Analysis
                Upload **multiple images** at once to get predictions for each one,
                along with aggregate statistics.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    batch_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Images",
                    )
                    batch_btn = gr.Button(
                        "📊 Analyse All",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    batch_results = gr.Dataframe(
                        headers=["Filename", "Prediction", "Confidence"],
                        label="Results",
                        interactive=False,
                        wrap=True,
                    )

            batch_summary = gr.Markdown(label="Summary")

            batch_btn.click(
                fn=classify_batch,
                inputs=[batch_input],
                outputs=[batch_results, batch_summary],
            )

        # ━━━━ TAB 3: HISTORY ━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.TabItem("📜 Prediction History"):
            gr.Markdown(
                """
                ### Session Prediction Log
                All predictions from this session are recorded here.
                """
            )
            history_table = gr.Dataframe(
                headers=["Timestamp", "Prediction", "Confidence (%)", "Source"],
                value=get_history,
                label="History",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh History", size="sm")
                clear_btn = gr.Button("🗑️ Clear History", variant="stop", size="sm")

            history_status = gr.Markdown("")

            refresh_btn.click(fn=get_history, outputs=[history_table])
            clear_btn.click(fn=clear_history, outputs=[history_table, history_status])

        # ━━━━ TAB 4: MODEL INFO ━━━━━━━━━━━━━━━━━━━━━
        with gr.TabItem("🧠 Model Info"):
            gr.Markdown(get_model_info())

            if os.path.exists(CURVES_SAVE_PATH):
                gr.Markdown("### 📈 Training Curves")
                gr.Image(
                    value=CURVES_SAVE_PATH,
                    label="Loss & Accuracy over Epochs",
                    show_label=True,
                    height=400,
                )

    # ── Footer ────────────────────────────────────────
    gr.Markdown(
        """
        ---
        **DeepFake Detection System** —Built by Sahil and Aayushi
        """
    )


# ── Launch ────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=custom_css,
    )
