"""
scripts/datacheck.py  —  Night Guidance System
Validates that all image-mask pairs are present and readable.

Run:
    python scripts/datacheck.py
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_BACKUP import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR,
    VAL_IMG_DIR,   VAL_MASK_DIR,
    ROAD_TRAIN_ID,
)


def check_split(img_dir, mask_dir, name):
    print(f"\n── {name} ──────────────────────────────")

    if not os.path.exists(img_dir):
        print(f"  ERROR: Image dir missing: {img_dir}")
        return 0
    if not os.path.exists(mask_dir):
        print(f"  ERROR: Mask dir missing:  {mask_dir}")
        return 0

    images     = sorted(os.listdir(img_dir))
    mask_files = set(os.listdir(mask_dir))

    missing_masks  = []
    orphan_masks   = []
    corrupt_imgs   = []
    corrupt_masks  = []
    class_counts   = {}

    for img in tqdm(images, desc=f"  Checking {name}"):
        stem = os.path.splitext(img)[0]
        mask_name = stem + "_train_id.png"

        if mask_name not in mask_files:
            missing_masks.append(img)
            continue

        img_path  = os.path.join(img_dir,  img)
        mask_path = os.path.join(mask_dir, mask_name)

        frame = cv2.imread(img_path)
        if frame is None:
            corrupt_imgs.append(img)
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            corrupt_masks.append(mask_name)
            continue

        for val in np.unique(mask):
            class_counts[int(val)] = class_counts.get(int(val), 0) + 1

    # Check orphan masks
    img_stems = {os.path.splitext(i)[0] for i in images}
    for mf in mask_files:
        stem = mf.replace("_train_id.png", "")
        if stem not in img_stems:
            orphan_masks.append(mf)

    usable = len(images) - len(missing_masks) - len(corrupt_imgs) - len(corrupt_masks)

    print(f"  Total images    : {len(images)}")
    print(f"  Usable pairs    : {usable}")
    print(f"  Missing masks   : {len(missing_masks)}")
    print(f"  Corrupt images  : {len(corrupt_imgs)}")
    print(f"  Corrupt masks   : {len(corrupt_masks)}")
    print(f"  Orphan masks    : {len(orphan_masks)}")

    if missing_masks:
        print(f"  Sample missing  : {missing_masks[:3]}")
    if corrupt_imgs:
        print(f"  Sample corrupt  : {corrupt_imgs[:3]}")

    # Class distribution
    print(f"\n  Class distribution (top 10 by frequency):")
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
    for cls, cnt in sorted_classes:
        road_marker = " ← ROAD" if cls == ROAD_TRAIN_ID else ""
        print(f"    class {cls:3d}  : {cnt:6d} images{road_marker}")

    return usable


if __name__ == "__main__":
    print("\n========================================")
    print("  Night Guidance System — Data Check")
    print("========================================")

    train_ok = check_split(TRAIN_IMG_DIR, TRAIN_MASK_DIR, "TRAIN")
    val_ok   = check_split(VAL_IMG_DIR,   VAL_MASK_DIR,   "VAL")

    print(f"\n  Total usable pairs: {train_ok + val_ok}")
    print("========================================\n")