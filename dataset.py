"""
dataset.py — Dataset loading and preprocessing pipeline.

This script:
  1. Scans the real/ and fake/ folders to build an image-label list.
  2. Applies data augmentation (training) or simple resize+normalize (validation).
  3. Splits into 80/20 train/val sets and returns DataLoaders.
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from config import (
    REAL_DIR, FAKE_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, VAL_SPLIT, NUM_WORKERS,
)


# ──────────────────────────────────────────────
# DATA AUGMENTATION / TRANSFORMS
# ──────────────────────────────────────────────

# Training transforms — stronger augmentation for better generalisation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),  # random crop + resize
    transforms.RandomHorizontalFlip(p=0.5),       # 50 % chance to flip
    transforms.RandomRotation(degrees=15),        # slight rotation
    transforms.RandomAffine(                      # perspective variation
        degrees=0, translate=(0.05, 0.05), shear=5,
    ),
    transforms.ColorJitter(                       # colour variation
        brightness=0.3, contrast=0.3,
        saturation=0.3, hue=0.1,
    ),
    transforms.RandomGrayscale(p=0.05),           # occasional grayscale
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # blur augmentation
    transforms.ToTensor(),                        # convert to tensor [0, 1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ImageNet stats
])

# Validation transforms — NO augmentation, only resize & normalize
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ──────────────────────────────────────────────
# CUSTOM DATASET CLASS
# ──────────────────────────────────────────────

class DeepFakeDataset(Dataset):
    """
    Reads images from `real/` and `fake/` folders.
    Labels:  0 = Real,  1 = Fake
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, transform=None):
        """
        Args:
            transform: torchvision transforms to apply to each image.
        """
        self.transform = transform
        self.image_paths = []   # list of file paths
        self.labels = []        # corresponding labels

        # Scan real/ folder → label 0
        self._scan_folder(REAL_DIR, label=0)
        # Scan fake/ folder → label 1
        self._scan_folder(FAKE_DIR, label=1)

        print(f"[Dataset] Loaded {len(self.image_paths)} images  "
              f"(Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})")

    def _scan_folder(self, folder_path, label):
        """Walk through a folder and collect image paths + labels."""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Expected folder not found: {folder_path}\n"
                f"Make sure your dataset is organised as dataset/real/ and dataset/fake/"
            )
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.VALID_EXTENSIONS:
                self.image_paths.append(os.path.join(folder_path, fname))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return (image_tensor, label) for a given index."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image and convert to RGB (handles grayscale/RGBA edge cases)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ──────────────────────────────────────────────
# HELPER: GET DATA LOADERS
# ──────────────────────────────────────────────

def get_data_loaders():
    """
    Build the full dataset, split into train/val, and return DataLoaders.

    Returns:
        train_loader, val_loader  (torch DataLoader objects)
    """
    # Build the complete dataset (no transforms yet — they'll be set below)
    full_dataset = DeepFakeDataset(transform=None)

    # Calculate split sizes
    total = len(full_dataset)
    val_size = int(total * VAL_SPLIT)
    train_size = total - val_size

    # Random split (reproducible with generator seed)
    import torch
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Wrap subsets so they use the correct transforms
    train_dataset = TransformSubset(train_subset, train_transforms)
    val_dataset   = TransformSubset(val_subset, val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {train_size} images | Val: {val_size} images")
    return train_loader, val_loader


class TransformSubset(Dataset):
    """
    Wraps a torch Subset so we can apply a DIFFERENT transform to
    train vs. val splits (since random_split shares the same Dataset).
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # If the image hasn't been transformed yet (it's still a PIL Image
        # because we passed transform=None to the parent dataset)
        if self.transform:
            image = self.transform(image)
        return image, label

