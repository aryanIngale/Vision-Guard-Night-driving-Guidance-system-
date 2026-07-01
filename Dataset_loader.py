"""
dataset_loader.py  —  Night Guidance System
BDD100K semantic segmentation dataset loader.

Mask convention:
  BDD100K ships *_train_id.png masks where each pixel = class train-id.
  For binary mode  → road (train-id 0) becomes 1, everything else 0.
  The mask file is named  <image_stem>_train_id.png
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_BACKUP import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR,
    VAL_IMG_DIR,   VAL_MASK_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    ROAD_TRAIN_ID,
    AUG_HFLIP, AUG_VFLIP, AUG_ROTATE90,
    AUG_COLOR_JITTER, AUG_BLUR,
)



# Dataset


class BDDDataset(Dataset):
    """
    BDD100K semantic segmentation dataset.

    Args:
        image_dir  : Path to folder of .jpg images
        mask_dir   : Path to folder of *_train_id.png masks
        size       : Square resize target
        augment    : Enable data augmentation (True only for training)
        road_id    : The train-id integer that represents road pixels
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        size    = IMAGE_SIZE,
        augment = False,
        road_id = ROAD_TRAIN_ID,
    ):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.size      = size
        self.augment   = augment
        self.road_id   = road_id

        # Only keep images that have a matching mask
        all_imgs = sorted(os.listdir(image_dir))
        self.images = []
        skipped = 0

        for img in all_imgs:
            stem      = os.path.splitext(img)[0]
            mask_name = stem + "_train_id.png"
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                self.images.append(img)
            else:
                skipped += 1

        if skipped:
            print(f"  [Dataset] Skipped {skipped} image(s) — no matching mask found.")

        print(f"  [Dataset] {len(self.images)} paired samples  |  aug={augment}  |  dir={image_dir}")

    # ── length ──────────────────────────────────

    def __len__(self):
        return len(self.images)

    # ── augmentation ────────────────────────────

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        """Apply identical spatial transforms to image + mask.
           Color transforms applied to image only."""

        # Horizontal flip
        if AUG_HFLIP and np.random.rand() > 0.5:
            img  = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # Vertical flip
        if AUG_VFLIP and np.random.rand() > 0.85:
            img  = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        # 90° rotation
        if AUG_ROTATE90 and np.random.rand() > 0.75:
            k    = np.random.randint(1, 4)
            img  = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        # Color jitter  (image only)
        if AUG_COLOR_JITTER and np.random.rand() > 0.4:
            hsv         = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.65, 1.35), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.60, 1.40), 0, 255)
            img         = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Gaussian blur  (simulates motion / lens blur)
        if AUG_BLUR and np.random.rand() > 0.75:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img, mask

    # ── item ────────────────────────────────────

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        stem      = os.path.splitext(img_name)[0]

        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir,  stem + "_train_id.png")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # -------- MULTICLASS MAPPING --------
        mask_new = np.zeros_like(mask)

        # Road
        mask_new[mask == self.road_id] = 1  

        # Vehicles
        vehicle_ids = [13, 14, 15, 16]
        mask_new[np.isin(mask, vehicle_ids)] = 2  

        # Pedestrians
        person_ids = [11, 12]
        mask_new[np.isin(mask, person_ids)] = 3  

        mask = mask_new.astype(np.int64)

        # Augment (AFTER mapping is OK)
        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalize image
        img = img.astype(np.float32) / 255.0

        # Convert to tensor
        img_tensor  = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).long()
        
        return img_tensor, mask_tensor



# DataLoader factory


def get_loaders():
    """Return (train_loader, val_loader) with config defaults."""

    train_ds = BDDDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, augment=True)
    val_ds   = BDDDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )

    return train_loader, val_loader



# Quick sanity test

if __name__ == "__main__":
    tl, vl = get_loaders()
    imgs, masks = next(iter(tl))
    print("Train batch — images:", imgs.shape, " masks:", masks.shape)
    print("Image range:", imgs.min().item(), "→", imgs.max().item())
    print("Unique mask values:", torch.unique(masks).tolist())