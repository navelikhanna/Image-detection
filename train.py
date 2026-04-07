"""
train.py — Two-Phase Training script for the DeepFake Detection model.

Usage:
    python train.py

Phase 1: Train classifier head only (frozen backbone) — fast convergence.
Phase 2: Unfreeze last N backbone blocks + lower LR — fine-tune for higher accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import (
    MODEL_SAVE_PATH, CURVES_SAVE_PATH, LEARNING_RATE, CLASS_NAMES,
    EARLY_STOP_PATIENCE, PHASE1_EPOCHS, FINETUNE_LR,
    FINETUNE_EPOCHS, UNFREEZE_BLOCKS,
)
from dataset import get_data_loaders
from model import build_model, unfreeze_model


def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    """Run a single epoch of training or validation."""
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                # Gradient clipping to prevent explosion during fine-tuning
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, num_epochs, patience, phase_name,
                history, best_val_loss):
    """
    Train for a given phase (head-only or fine-tuning).
    Returns updated history and best_val_loss.
    """
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        global_epoch = len(history["train_loss"]) + 1

        # ── Train ──
        pbar_train = tqdm(train_loader, desc=f"[{phase_name}] Epoch {epoch}/{num_epochs} [Train]")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in pbar_train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar_train.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[{phase_name}] Epoch {epoch}/{num_epochs} [Val]  "):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\n📊 [{phase_name}] Epoch {epoch}/{num_epochs}  —  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}\n")

        # Step the LR scheduler
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"   📈 Current LR: {current_lr:.2e}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   ✅ Best model saved → {MODEL_SAVE_PATH}  (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
                break

    # Print classification report for this phase
    print(f"\n{'='*60}")
    print(f"📋 CLASSIFICATION REPORT ({phase_name} — last epoch)")
    print("=" * 60)
    print(classification_report(all_labels_list, all_preds, target_names=CLASS_NAMES))

    return history, best_val_loss


def train():
    # ── Device setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    print("\n📂 Loading dataset...")
    train_loader, val_loader = get_data_loaders()

    # ── Model ──
    print("\n🏗️  Building model...")
    model = build_model(pretrained=True)
    model = model.to(device)

    # ── Loss with label smoothing for better generalisation ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    # ════════════════════════════════════════════
    # PHASE 1: Head-only training (frozen backbone)
    # ════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("🔒 PHASE 1: Training classifier head (backbone frozen)")
    print("═" * 60)

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode="min", factor=0.5, patience=2,
    )

    history, best_val_loss = train_phase(
        model, train_loader, val_loader, criterion, optimizer1,
        scheduler1, device, num_epochs=PHASE1_EPOCHS,
        patience=EARLY_STOP_PATIENCE, phase_name="Phase 1",
        history=history, best_val_loss=best_val_loss,
    )

    # ════════════════════════════════════════════
    # PHASE 2: Fine-tuning (unfreeze last N backbone blocks)
    # ════════════════════════════════════════════
    print("\n" + "═" * 60)
    print(f"🔓 PHASE 2: Fine-tuning (unfreezing last {UNFREEZE_BLOCKS} backbone blocks)")
    print("═" * 60)

    # Reload best weights from Phase 1
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("   📦 Loaded best Phase 1 weights")

    # Unfreeze last N blocks
    unfreeze_model(model, num_layers_to_unfreeze=UNFREEZE_BLOCKS)
    model = model.to(device)

    # New optimizer with MUCH lower LR for backbone, slightly higher for head
    optimizer2 = optim.Adam([
        {"params": model.features.parameters(), "lr": FINETUNE_LR},
        {"params": model.classifier.parameters(), "lr": FINETUNE_LR * 5},
    ])
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=FINETUNE_EPOCHS, eta_min=1e-7,
    )
    # Wrap CosineAnnealing in a compatible way for our train_phase function
    # We use ReduceLROnPlateau as a wrapper that also monitors val_loss
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="min", factor=0.5, patience=3,
    )

    history, best_val_loss = train_phase(
        model, train_loader, val_loader, criterion, optimizer2,
        scheduler2, device, num_epochs=FINETUNE_EPOCHS,
        patience=EARLY_STOP_PATIENCE + 1,  # Extra patience for fine-tuning
        phase_name="Phase 2",
        history=history, best_val_loss=best_val_loss,
    )

    # ── Plot training curves ──
    plot_curves(history)
    print(f"\n📈 Training curves saved → {CURVES_SAVE_PATH}")
    print(f"💾 Best model weights    → {MODEL_SAVE_PATH}")
    print("\n✅ Two-phase training complete!")


def plot_curves(history):
    """Plot and save loss + accuracy curves with phase separator."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Phase separator line
    phase1_end = PHASE1_EPOCHS
    if phase1_end < len(history["train_loss"]):
        for ax in (ax1, ax2):
            ax.axvline(x=phase1_end + 0.5, color='red', linestyle='--',
                       alpha=0.5, label='Phase 1→2')

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "o-", label="Val Loss", markersize=4)
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"], "o-", label="Val Acc", markersize=4)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVES_SAVE_PATH, dpi=150)
    plt.close()


if __name__ == "__main__":
    train()
